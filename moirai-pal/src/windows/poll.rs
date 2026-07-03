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
    /// Registered sockets (by raw `SOCKET` value) and their interest.
    interests: Mutex<HashMap<usize, Interest>>,
    /// Loopback UDP socket used to interrupt a blocking `WSAPoll`: `wake()` sends
    /// a datagram to `wake_addr`, making this socket readable so the poll returns
    /// promptly (e.g. after a new registration or on shutdown).
    wake: UdpSocket,
    wake_addr: std::net::SocketAddr,
    /// Reused `WSAPoll` fd array, so the hot poll loop does not rebuild a fresh
    /// vector allocation per iteration. Lock order: `pollfds` before
    /// `interests` (the only path taking both is `poll_events`; every other
    /// path takes at most `interests`), so no cycle exists.
    pollfds: Mutex<Vec<WSAPOLLFD>>,
}

// Safety: all shared state is behind the `interests` and `pollfds` `Mutex`es;
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
            interests: Mutex::new(HashMap::new()),
            wake,
            wake_addr,
            pollfds: Mutex::new(Vec::new()),
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
}

impl Reactor for WsaPollReactor {
    fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        lock_mutex(&self.interests).insert(fd as usize, interest);
        // Interrupt any in-flight blocking poll so the new socket is included.
        let _ = self.wake();
        Ok(())
    }

    fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        lock_mutex(&self.interests).remove(&(fd as usize));
        Ok(())
    }

    fn poll_events(&self, timeout: Option<Duration>) -> io::Result<Vec<Event>> {
        // Reuse the persistent fd array; the mutex serializes concurrent
        // pollers (the reactor is driven by one event-loop thread in practice).
        // The interest map is snapshotted directly into the reused buffer under
        // a short-lived lock, released before the (potentially blocking) poll
        // so `register_fd`/`unregister_fd` never wait on a poll in flight.
        let mut pollfds = lock_mutex(&self.pollfds);
        pollfds.clear();
        // Slot 0 is always the wake socket, so `nfds >= 1` (WSAPoll rejects 0).
        pollfds.push(WSAPOLLFD {
            fd: SOCKET(self.wake_socket()),
            events: POLLRDNORM,
            revents: WSAPOLL_EVENT_FLAGS(0),
        });
        {
            let map = lock_mutex(&self.interests);
            pollfds.reserve(map.len());
            for (sock, interest) in map.iter() {
                let mut events = WSAPOLL_EVENT_FLAGS(0);
                if interest.readable {
                    events |= POLLRDNORM;
                }
                if interest.writable {
                    events |= POLLWRNORM;
                }
                pollfds.push(WSAPOLLFD {
                    fd: SOCKET(*sock),
                    events,
                    revents: WSAPOLL_EVENT_FLAGS(0),
                });
            }
        }

        let timeout_ms = timeout.map_or(-1, |d| d.as_millis().min(i32::MAX as u128) as i32);

        // Safety: `pollfds` is a valid, correctly-sized array of `WSAPOLLFD` that
        // outlives the call; `WSAPoll` writes only into the `revents` fields.
        let n = unsafe { WSAPoll(pollfds.as_mut_ptr(), pollfds.len() as u32, timeout_ms) };
        if n == SOCKET_ERROR {
            return Err(io::Error::last_os_error());
        }
        if n == 0 {
            return Ok(Vec::new()); // timeout
        }

        if pollfds[0].revents.0 != 0 {
            self.drain_wake();
        }

        let mut events_out = Vec::new();
        let mut invalid: Vec<usize> = Vec::new();
        for pfd in &pollfds[1..] {
            let r = pfd.revents.0; // raw i16 flag bits
            if r == 0 {
                continue;
            }
            if r & POLLNVAL.0 != 0 {
                // Socket was closed by its owner; drop it from the interest set.
                invalid.push(pfd.fd.0);
                continue;
            }
            events_out.push(Event {
                fd: pfd.fd.0 as RawFd,
                readable: r & (POLLRDNORM.0 | POLLHUP.0 | POLLERR.0) != 0,
                writable: r & POLLWRNORM.0 != 0,
                error: r & POLLERR.0 != 0,
                hangup: r & POLLHUP.0 != 0,
            });
        }

        if !invalid.is_empty() {
            let mut map = lock_mutex(&self.interests);
            for sock in invalid {
                map.remove(&sock);
            }
        }

        Ok(events_out)
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
        let started = std::time::Instant::now();
        let handle = std::thread::spawn(move || {
            // Would block up to 5s; wake() must interrupt it well before.
            // (Don't return the `Vec<Event>` — it is `!Send` on Windows.)
            let _ = polling
                .poll_events(Some(Duration::from_secs(5)))
                .expect("poll");
        });
        std::thread::sleep(Duration::from_millis(50));
        reactor.wake().expect("wake");
        handle.join().expect("poll thread");
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "wake() must promptly interrupt a blocking poll"
        );
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
            lock_mutex(&reactor.interests).is_empty(),
            "closed socket must be self-cleaned from the interest set"
        );
    }
}
