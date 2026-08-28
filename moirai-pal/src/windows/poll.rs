//! Windows readiness reactor backed by `WSAPoll`.
//!
//! The IOCP completion model signals completions of *posted overlapped
//! operations*, not socket *readiness*, so it cannot drive the readiness-based
//! futures in [`crate::net`] (which do a non-blocking syscall and register a
//! waker on `WouldBlock`). This reactor uses `WSAPoll` — the Windows analogue of
//! `poll(2)` — to report which registered sockets are readable/writable, which is
//! exactly the readiness signal those futures need.
//!
//! Self-cleaning: a socket closed by its owner (without an explicit
//! `unregister_fd`) surfaces as `POLLNVAL` and is dropped from the interest map,
//! so a stale entry cannot wedge the poll loop.

use std::collections::HashMap;
use std::io;
use std::net::UdpSocket;
use std::os::windows::io::AsRawSocket;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use windows::Win32::Networking::WinSock::{
    WSAPoll, POLLERR, POLLHUP, POLLNVAL, POLLRDNORM, POLLWRNORM, SOCKET, SOCKET_ERROR, WSAPOLLFD,
    WSAPOLL_EVENT_FLAGS,
};

use crate::{Event, Interest, RawFd, Reactor};

/// `WSAPoll`-based readiness reactor.
pub struct WsaPollReactor {
    /// Registered sockets and the generation that distinguishes reused raw
    /// `SOCKET` values.
    registrations: Mutex<RegistrationTable>,
    /// Loopback UDP socket used to interrupt a blocking `WSAPoll`: `wake()` sends
    /// a datagram to `wake_addr`, making this socket readable so the poll returns
    /// promptly (e.g. after a new registration or on shutdown).
    wake: UdpSocket,
    wake_addr: std::net::SocketAddr,
    /// Reused `WSAPoll` snapshot, so the hot poll loop does not allocate fd or
    /// generation arrays per iteration. Lock order: `poll_buffer` before
    /// `registrations`; every other path takes at most `registrations`.
    poll_buffer: Mutex<PollBuffer>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RegistrationGeneration(u64);

#[derive(Clone, Copy)]
struct Registration {
    interest: Interest,
    generation: RegistrationGeneration,
}

#[derive(Default)]
struct RegistrationTable {
    entries: HashMap<usize, Registration>,
    next_generation: u64,
}

impl RegistrationTable {
    fn register(&mut self, socket: usize, interest: Interest) -> io::Result<()> {
        let next_generation = self
            .next_generation
            .checked_add(1)
            .ok_or_else(|| io::Error::other("WSAPoll registration generation exhausted"))?;
        self.next_generation = next_generation;
        self.entries.insert(
            socket,
            Registration {
                interest,
                generation: RegistrationGeneration(next_generation),
            },
        );
        Ok(())
    }

    fn is_current(&self, socket: usize, generation: RegistrationGeneration) -> bool {
        self.entries
            .get(&socket)
            .is_some_and(|registration| registration.generation == generation)
    }

    fn remove_if_current(&mut self, socket: usize, generation: RegistrationGeneration) -> bool {
        if !self.is_current(socket, generation) {
            return false;
        }
        self.entries.remove(&socket).is_some()
    }
}

#[derive(Default)]
struct PollBuffer {
    fds: Vec<WSAPOLLFD>,
    generations: Vec<RegistrationGeneration>,
}

/// Readiness paired with the registration generation represented by the poll
/// snapshot. The public [`Event`] remains platform-neutral; central dispatch
/// uses this sidecar to reject readiness from a replaced socket registration.
pub(crate) struct PolledEvent {
    event: Event,
    generation: RegistrationGeneration,
}

impl PolledEvent {
    pub(crate) fn event(&self) -> &Event {
        &self.event
    }

    #[cfg(test)]
    pub(crate) fn descriptor(&self) -> RawFd {
        self.event.fd
    }
}

// Safety: all shared state is behind the `registrations` and `poll_buffer` `Mutex`es;
// the `wake` `UdpSocket` supports concurrent `send_to` (any thread) and `recv`
// (the poll thread), which winsock permits for UDP.
unsafe impl Send for WsaPollReactor {}
unsafe impl Sync for WsaPollReactor {}

impl WsaPollReactor {
    /// Create a reactor with an empty interest set and a fresh wake socket.
    pub fn new() -> io::Result<Self> {
        let wake = UdpSocket::bind("127.0.0.1:0")?;
        wake.set_nonblocking(true)?;
        let wake_addr = wake.local_addr()?;
        Ok(Self {
            registrations: Mutex::new(RegistrationTable::default()),
            wake,
            wake_addr,
            poll_buffer: Mutex::new(PollBuffer::default()),
        })
    }

    fn wake_socket(&self) -> usize {
        self.wake.as_raw_socket() as usize
    }

    /// Drain any pending wake datagrams (the socket is non-blocking).
    fn drain_wake(&self) {
        let mut buf = [0u8; 64];
        while self.wake.recv(&mut buf).is_ok() {}
    }

    /// Return whether a polled event still names the registration represented
    /// by its snapshot generation.
    pub(crate) fn is_current_polled_event(&self, event: &PolledEvent) -> bool {
        lock_mutex(&self.registrations).is_current(event.event.fd as usize, event.generation)
    }

    /// Poll readiness while preserving each snapshot registration generation
    /// for central dispatch.
    pub(crate) fn poll_registered_events(
        &self,
        timeout: Option<Duration>,
    ) -> io::Result<Vec<PolledEvent>> {
        self.poll_events_with(timeout, |event, generation| PolledEvent {
            event,
            generation,
        })
    }

    fn poll_events_with<T>(
        &self,
        timeout: Option<Duration>,
        mut make_event: impl FnMut(Event, RegistrationGeneration) -> T,
    ) -> io::Result<Vec<T>> {
        // Reuse the persistent fd array; the mutex serializes concurrent
        // pollers. The generation sidecar remains paired with each returned
        // event even after this buffer is reused by a later poll.
        let mut poll_buffer = lock_mutex(&self.poll_buffer);
        poll_buffer.fds.clear();
        poll_buffer.generations.clear();
        // Slot 0 is always the wake socket, so `nfds >= 1` (WSAPoll rejects 0).
        poll_buffer.fds.push(WSAPOLLFD {
            fd: SOCKET(self.wake_socket()),
            events: POLLRDNORM,
            revents: WSAPOLL_EVENT_FLAGS(0),
        });
        {
            let registrations = lock_mutex(&self.registrations);
            poll_buffer.fds.reserve(registrations.entries.len());
            poll_buffer.generations.reserve(registrations.entries.len());
            for (socket, registration) in &registrations.entries {
                let mut events = WSAPOLL_EVENT_FLAGS(0);
                if registration.interest.readable {
                    events |= POLLRDNORM;
                }
                if registration.interest.writable {
                    events |= POLLWRNORM;
                }
                poll_buffer.fds.push(WSAPOLLFD {
                    fd: SOCKET(*socket),
                    events,
                    revents: WSAPOLL_EVENT_FLAGS(0),
                });
                poll_buffer.generations.push(registration.generation);
            }
        }

        let timeout_ms = timeout.map_or(-1, |d| d.as_millis().min(i32::MAX as u128) as i32);

        // Safety: `fds` is a valid, correctly-sized array of `WSAPOLLFD` that
        // outlives the call; `WSAPoll` writes only into the `revents` fields.
        let n = unsafe {
            WSAPoll(
                poll_buffer.fds.as_mut_ptr(),
                poll_buffer.fds.len() as u32,
                timeout_ms,
            )
        };
        if n == SOCKET_ERROR {
            return Err(io::Error::last_os_error());
        }
        if n == 0 {
            return Ok(Vec::new());
        }

        if poll_buffer.fds[0].revents.0 != 0 {
            self.drain_wake();
        }

        let mut events_out = Vec::new();
        let mut registrations = lock_mutex(&self.registrations);
        for (pfd, generation) in poll_buffer.fds[1..].iter().zip(&poll_buffer.generations) {
            let r = pfd.revents.0;
            if r == 0 {
                continue;
            }
            let socket = pfd.fd.0;
            if !registrations.is_current(socket, *generation) {
                continue;
            }
            if r & POLLNVAL.0 != 0 {
                // Remove only the registration represented by this snapshot;
                // the raw SOCKET value may already belong to a newer socket.
                registrations.remove_if_current(socket, *generation);
                continue;
            }
            events_out.push(make_event(
                Event {
                    fd: socket as RawFd,
                    readable: r & (POLLRDNORM.0 | POLLHUP.0 | POLLERR.0) != 0,
                    writable: r & POLLWRNORM.0 != 0,
                    error: r & POLLERR.0 != 0,
                    hangup: r & POLLHUP.0 != 0,
                },
                *generation,
            ));
        }

        Ok(events_out)
    }
}

impl Reactor for WsaPollReactor {
    fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        lock_mutex(&self.registrations).register(fd as usize, interest)?;
        // Interrupt any in-flight blocking poll so the new socket is included.
        let _ = self.wake();
        Ok(())
    }

    fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        lock_mutex(&self.registrations)
            .entries
            .remove(&(fd as usize));
        Ok(())
    }

    fn poll_events(&self, timeout: Option<Duration>) -> io::Result<Vec<Event>> {
        self.poll_events_with(timeout, |event, _generation| event)
    }

    fn wake(&self) -> io::Result<()> {
        self.wake.send_to(&[1u8], self.wake_addr).map(|_| ())
    }
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn wsapoll_reactor_reports_socket_readiness() {
        let reactor = WsaPollReactor::new().expect("reactor");
        let recv = UdpSocket::bind("127.0.0.1:0").expect("recv bind");
        recv.set_nonblocking(true).expect("nonblocking");
        let recv_addr = recv.local_addr().expect("recv addr");
        let sender = UdpSocket::bind("127.0.0.1:0").expect("sender bind");
        let raw = recv.as_raw_socket() as RawFd;
        let raw_key = raw as usize;

        reactor
            .register_fd(raw, Interest::READABLE)
            .expect("register");

        // Before any datagram arrives the receiver is not readable. (register_fd
        // pinged the wake socket; poll_events drains that and reports no recv
        // readiness.)
        let events = reactor
            .poll_events(Some(Duration::from_millis(50)))
            .expect("poll");
        assert!(
            events.iter().all(|e| e.fd as usize != raw_key),
            "no readiness before data"
        );

        // After a datagram, WSAPoll reports the receiver readable.
        sender.send_to(b"x", recv_addr).expect("send");
        let events = reactor
            .poll_events(Some(Duration::from_millis(500)))
            .expect("poll");
        assert!(
            events
                .iter()
                .any(|e| e.fd as usize == raw_key && e.readable),
            "reactor must report the receiver readable, got {events:?}"
        );
    }

    #[test]
    fn wsapoll_reactor_wake_interrupts_a_blocking_poll() {
        let reactor = Arc::new(WsaPollReactor::new().expect("reactor"));
        let polling = Arc::clone(&reactor);
        let start = Arc::new(std::sync::Barrier::new(2));
        let polling_start = Arc::clone(&start);
        let (done_tx, done_rx) = std::sync::mpsc::sync_channel(1);
        let handle = std::thread::spawn(move || {
            // Would block up to 5s; wake() must interrupt it well before.
            // (Don't return the `Vec<Event>` — it is `!Send` on Windows.)
            polling_start.wait();
            let result = polling.poll_events(Some(Duration::from_secs(5))).map(drop);
            done_tx.send(result).expect("report poll completion");
        });
        start.wait();
        reactor.wake().expect("wake");
        done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("wake must complete the poll before the deadline")
            .expect("poll");
        handle.join().expect("poll thread");
    }

    #[test]
    fn wsapoll_reactor_self_cleans_closed_sockets() {
        let reactor = WsaPollReactor::new().expect("reactor");
        let sock = UdpSocket::bind("127.0.0.1:0").expect("bind");
        let raw = sock.as_raw_socket() as RawFd;
        reactor
            .register_fd(raw, Interest::READABLE)
            .expect("register");
        drop(sock); // close without unregistering -> POLLNVAL

        // The poll surfaces POLLNVAL for the closed socket and removes it; the
        // call must not error or wedge.
        let _ = reactor
            .poll_events(Some(Duration::from_millis(50)))
            .expect("poll must not error on a stale socket");
        assert!(
            lock_mutex(&reactor.registrations).entries.is_empty(),
            "closed socket must be self-cleaned from the interest set"
        );
    }

    #[test]
    fn stale_snapshot_cannot_remove_or_match_reused_socket() {
        let socket = 41;
        let mut registrations = RegistrationTable::default();
        registrations
            .register(socket, Interest::READABLE)
            .expect("first registration");
        let stale_generation = registrations
            .entries
            .get(&socket)
            .expect("first registration exists")
            .generation;

        registrations
            .register(socket, Interest::WRITABLE)
            .expect("replacement registration");
        let current_generation = registrations
            .entries
            .get(&socket)
            .expect("replacement registration exists")
            .generation;

        assert_ne!(stale_generation, current_generation);
        assert!(!registrations.is_current(socket, stale_generation));
        assert!(!registrations.remove_if_current(socket, stale_generation));
        assert!(registrations.is_current(socket, current_generation));
    }
}
