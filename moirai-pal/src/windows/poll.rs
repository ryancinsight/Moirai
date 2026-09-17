//! Windows readiness reactor backed by `WSAPoll`.
//!
//! The IOCP completion model signals completions of *posted overlapped
//! operations*, not socket *readiness*, so it cannot drive the readiness-based
//! futures in [`crate::net`] (which do a non-blocking syscall and register a
//! waker on `WouldBlock`). This reactor uses `WSAPoll` — the Windows analogue of
//! `poll(2)` — to report which registered sockets are readable/writable, which is
//! exactly the readiness signal those futures need.
//!
//! PAL socket registrations retain a weak OS-socket owner. Each poll snapshot
//! upgrades that owner and holds the strong lease through `WSAPoll`, excluding
//! concurrent `closesocket`; an owner retired before snapshot acquisition is
//! invalidated without entering the kernel call. Raw registrations remain
//! caller-owned and a closed raw socket surfaces as `POLLNVAL`. Every
//! invalidation carries its registration generation, so a delayed event cannot
//! consume a newer registration for a reused raw socket value.

use std::io;
use std::net::UdpSocket;
use std::ops::{Deref, DerefMut};
use std::os::windows::io::AsRawSocket;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use windows::Win32::Networking::WinSock::{
    POLLERR, POLLHUP, POLLNVAL, POLLRDNORM, POLLWRNORM, SOCKET, SOCKET_ERROR, WSAGetLastError,
    WSAPOLL_EVENT_FLAGS, WSAPOLLFD, WSAPoll,
};

use crate::reactor::registration::{
    PlatformUpdateFailure, PolledEvent, RegistrationGeneration, RegistrationTable,
    WaiterRegistration,
};
use crate::reactor::socket_owner::{SocketLease, WeakSocketOwner};
use crate::{Event, Interest, RawFd, Reactor};

/// `WSAPoll`-based readiness reactor.
pub struct WsaPollReactor {
    /// Registered sockets and the generation that distinguishes reused raw
    /// `SOCKET` values.
    registrations: Mutex<RegistrationTable<usize>>,
    /// Loopback UDP socket used to interrupt a blocking `WSAPoll`: `wake()` sends
    /// a datagram to `wake_addr`, making this socket readable so the poll returns
    /// promptly (e.g. after a new registration or on shutdown).
    wake: UdpSocket,
    wake_addr: std::net::SocketAddr,
    /// Reused `WSAPoll` snapshot, so the hot poll loop does not allocate fd or
    /// generation arrays per iteration. Lock order: `poll_buffer` before
    /// `registrations`; every other path takes at most `registrations`.
    poll_buffer: Mutex<PollBuffer>,
    /// Reused strong-owner storage, kept separate so final owner release never
    /// occurs while the poll snapshot lock is held.
    lease_buffer: Mutex<Vec<SocketLease>>,
}

#[derive(Default)]
struct PollBuffer {
    fds: Vec<WSAPOLLFD>,
    generations: Vec<RegistrationGeneration>,
}

struct PollSnapshot<'a> {
    lease_source: &'a Mutex<Vec<SocketLease>>,
    buffer: Option<MutexGuard<'a, PollBuffer>>,
    leases: Vec<SocketLease>,
}

impl<'a> PollSnapshot<'a> {
    fn acquire(source: &'a Mutex<PollBuffer>, lease_source: &'a Mutex<Vec<SocketLease>>) -> Self {
        let buffer = lock_mutex(source);
        let leases = std::mem::take(&mut *lock_mutex(lease_source));
        Self {
            lease_source,
            buffer: Some(buffer),
            leases,
        }
    }

    fn finish(mut self) {
        drop(self.buffer.take());
        self.leases.clear();
        std::mem::swap(&mut *lock_mutex(self.lease_source), &mut self.leases);
    }
}

impl Deref for PollSnapshot<'_> {
    type Target = PollBuffer;

    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_deref()
            .expect("poll snapshot owns its buffer until release")
    }
}

impl DerefMut for PollSnapshot<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer
            .as_deref_mut()
            .expect("poll snapshot owns its buffer until release")
    }
}

impl Drop for PollSnapshot<'_> {
    fn drop(&mut self) {
        drop(self.buffer.take());
        self.leases.clear();
    }
}

// SAFETY: all shared state is behind the `registrations` and `poll_buffer` `Mutex`es;
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
            lease_buffer: Mutex::new(Vec::new()),
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
        lock_mutex(&self.registrations).is_current(event.event().fd as usize, event.generation())
    }

    #[cfg(test)]
    pub(crate) fn has_registration(&self, fd: RawFd) -> bool {
        lock_mutex(&self.registrations).get(fd as usize).is_some()
    }

    pub(crate) fn update_registration(
        &self,
        fd: RawFd,
        interest: Interest,
    ) -> Result<(), PlatformUpdateFailure> {
        let socket = fd as usize;
        let mut registrations = lock_mutex(&self.registrations);
        let Some(current) = registrations.get(socket) else {
            return Err(PlatformUpdateFailure::new(
                io::Error::new(io::ErrorKind::NotFound, "WSAPoll registration is absent"),
                None,
            ));
        };
        if interest.readable || interest.writable {
            let updated = registrations.update_interest(socket, current.generation, interest);
            debug_assert!(updated, "registration remained locked during update");
        } else {
            registrations.remove(socket);
        }
        Ok(())
    }

    pub(crate) fn replace_registration(
        &self,
        fd: RawFd,
        interest: Interest,
    ) -> Result<(), PlatformUpdateFailure> {
        self.replace_waiter_registration(fd, interest, interest)
            .map(drop)
    }

    pub(crate) fn register_waiter(
        &self,
        fd: RawFd,
        interest: Interest,
    ) -> Result<RegistrationGeneration, PlatformUpdateFailure> {
        self.replace_waiter_registration(fd, interest, interest)
            .map(|registration| registration.generation)
    }

    pub(crate) fn register_owned_waiter(
        &self,
        fd: RawFd,
        interest: Interest,
        owner: WeakSocketOwner,
    ) -> Result<RegistrationGeneration, PlatformUpdateFailure> {
        self.replace_waiter_registration_with_owner(fd, interest, interest, Some(owner))
            .map(|registration| registration.generation)
    }

    pub(crate) fn replace_waiter_registration(
        &self,
        fd: RawFd,
        retained_interest: Interest,
        fresh_interest: Interest,
    ) -> Result<WaiterRegistration, PlatformUpdateFailure> {
        self.replace_waiter_registration_with_owner(fd, retained_interest, fresh_interest, None)
    }

    pub(crate) fn replace_owned_waiter_registration(
        &self,
        fd: RawFd,
        retained_interest: Interest,
        fresh_interest: Interest,
        owner: WeakSocketOwner,
    ) -> Result<WaiterRegistration, PlatformUpdateFailure> {
        self.replace_waiter_registration_with_owner(
            fd,
            retained_interest,
            fresh_interest,
            Some(owner),
        )
    }

    fn replace_waiter_registration_with_owner(
        &self,
        fd: RawFd,
        retained_interest: Interest,
        fresh_interest: Interest,
        owner: Option<WeakSocketOwner>,
    ) -> Result<WaiterRegistration, PlatformUpdateFailure> {
        let socket = fd as usize;
        let mut registrations = lock_mutex(&self.registrations);
        let previous = registrations.get(socket);
        let interest = if previous.is_some() {
            retained_interest
        } else {
            fresh_interest
        };
        let generation = registrations.issue_generation().map_err(|error| {
            PlatformUpdateFailure::new(error, previous.as_ref().map(|entry| entry.interest))
        })?;
        if let Some(owner) = owner {
            registrations.commit_owned(socket, interest, generation, owner);
        } else {
            registrations.commit(socket, interest, generation);
        }
        if let Err(error) = self.wake() {
            let armed_interest = if let Some(previous) = previous {
                if let Some(owner) = previous.owner {
                    registrations.commit_owned(
                        socket,
                        previous.interest,
                        previous.generation,
                        owner,
                    );
                } else {
                    registrations.commit(socket, previous.interest, previous.generation);
                }
                Some(previous.interest)
            } else {
                let removed = registrations.remove_if_current(socket, generation);
                debug_assert!(removed, "registration remained locked during wake rollback");
                None
            };
            return Err(PlatformUpdateFailure::new(error, armed_interest));
        }
        Ok(WaiterRegistration {
            generation,
            replaced_existing: previous.is_some(),
        })
    }

    /// Poll readiness while preserving each snapshot registration generation
    /// for central dispatch.
    pub(crate) fn poll_registered_events(
        &self,
        timeout: Option<Duration>,
    ) -> io::Result<Vec<PolledEvent>> {
        self.poll_events_after_snapshot(
            timeout,
            |event, generation, invalidated| {
                if invalidated {
                    PolledEvent::invalidated(event, generation)
                } else {
                    PolledEvent::new(event, generation)
                }
            },
            || {},
        )
    }

    #[cfg(test)]
    pub(crate) fn poll_registered_events_after_snapshot(
        &self,
        timeout: Option<Duration>,
        after_snapshot: impl FnOnce(),
    ) -> io::Result<Vec<PolledEvent>> {
        self.poll_events_after_snapshot(
            timeout,
            |event, generation, invalidated| {
                if invalidated {
                    PolledEvent::invalidated(event, generation)
                } else {
                    PolledEvent::new(event, generation)
                }
            },
            after_snapshot,
        )
    }

    #[cfg(test)]
    pub(crate) fn poll_registered_events_after_snapshot_error(
        &self,
        after_snapshot: impl FnOnce(),
    ) -> io::Result<Vec<PolledEvent>> {
        self.poll_events_after_snapshot_with(
            Some(Duration::ZERO),
            |event, generation, invalidated| {
                if invalidated {
                    PolledEvent::invalidated(event, generation)
                } else {
                    PolledEvent::new(event, generation)
                }
            },
            after_snapshot,
            |fds, timeout| {
                let registered = fds
                    .get_mut(1)
                    .expect("test poll includes one owned registration");
                registered.events |= WSAPOLL_EVENT_FLAGS(0x4000);
                // SAFETY: the test passes a valid array and deliberately asks
                // Winsock to reject an unsupported event flag.
                unsafe { WSAPoll(fds.as_mut_ptr(), fds.len() as u32, timeout) }
            },
        )
    }

    fn poll_events_with<T>(
        &self,
        timeout: Option<Duration>,
        make_event: impl FnMut(Event, RegistrationGeneration, bool) -> T,
    ) -> io::Result<Vec<T>> {
        self.poll_events_after_snapshot(timeout, make_event, || {})
    }

    fn poll_events_after_snapshot<T>(
        &self,
        timeout: Option<Duration>,
        make_event: impl FnMut(Event, RegistrationGeneration, bool) -> T,
        after_snapshot: impl FnOnce(),
    ) -> io::Result<Vec<T>> {
        self.poll_events_after_snapshot_with(timeout, make_event, after_snapshot, |fds, timeout| {
            // SAFETY: `fds` is a valid, correctly-sized array of `WSAPOLLFD`
            // that outlives the call; `WSAPoll` writes only `revents`.
            unsafe { WSAPoll(fds.as_mut_ptr(), fds.len() as u32, timeout) }
        })
    }

    fn poll_events_after_snapshot_with<T>(
        &self,
        timeout: Option<Duration>,
        mut make_event: impl FnMut(Event, RegistrationGeneration, bool) -> T,
        after_snapshot: impl FnOnce(),
        poll: impl FnOnce(&mut [WSAPOLLFD], i32) -> i32,
    ) -> io::Result<Vec<T>> {
        // Reuse the persistent fd array; the mutex serializes concurrent
        // pollers. The generation sidecar remains paired with each returned
        // event even after this buffer is reused by a later poll.
        let mut poll_buffer = PollSnapshot::acquire(&self.poll_buffer, &self.lease_buffer);
        poll_buffer.fds.clear();
        poll_buffer.generations.clear();
        // Slot 0 is always the wake socket, so `nfds >= 1` (WSAPoll rejects 0).
        poll_buffer.fds.push(WSAPOLLFD {
            fd: SOCKET(self.wake_socket()),
            events: POLLRDNORM,
            revents: WSAPOLL_EVENT_FLAGS(0),
        });
        let mut events_out = Vec::new();
        {
            let mut registrations = lock_mutex(&self.registrations);
            poll_buffer.fds.reserve(registrations.len());
            poll_buffer.generations.reserve(registrations.len());
            poll_buffer.leases.reserve(registrations.len());
            registrations.retain(|socket, registration| {
                let polled_socket = if let Some(owner) = registration.owner.as_ref() {
                    let Some(lease) = owner.upgrade() else {
                        events_out.push(make_event(
                            Event {
                                fd: socket as RawFd,
                                readable: false,
                                writable: false,
                                error: true,
                                hangup: true,
                            },
                            registration.generation,
                            true,
                        ));
                        return false;
                    };
                    let raw_socket = lease.raw_socket() as usize;
                    debug_assert_eq!(raw_socket, socket, "socket owner matches registration key");
                    poll_buffer.leases.push(lease);
                    raw_socket
                } else {
                    socket
                };
                let mut events = WSAPOLL_EVENT_FLAGS(0);
                if registration.interest.readable {
                    events |= POLLRDNORM;
                }
                if registration.interest.writable {
                    events |= POLLWRNORM;
                }
                poll_buffer.fds.push(WSAPOLLFD {
                    fd: SOCKET(polled_socket),
                    events,
                    revents: WSAPOLL_EVENT_FLAGS(0),
                });
                poll_buffer.generations.push(registration.generation);
                true
            });
        }
        // Test synchronization can close a socket only after its exact
        // registration generation has entered this snapshot. This point is
        // immediately before `WSAPoll`, but does not claim that the kernel call
        // has already begun.
        after_snapshot();

        let timeout_ms = timeout.map_or(-1, |d| d.as_millis().min(i32::MAX as u128) as i32);

        let n = poll(&mut poll_buffer.fds, timeout_ms);
        if n == SOCKET_ERROR {
            // Winsock requires its thread-local error to be read immediately
            // after the failed call. Release snapshot leases only afterward.
            // SAFETY: `WSAGetLastError` has no preconditions.
            let error = io::Error::from_raw_os_error(unsafe { WSAGetLastError() }.0);
            poll_buffer.finish();
            return Err(error);
        }
        if n == 0 {
            poll_buffer.finish();
            return Ok(events_out);
        }

        if poll_buffer.fds[0].revents.0 != 0 {
            self.drain_wake();
        }

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
                events_out.push(make_event(
                    Event {
                        fd: socket as RawFd,
                        readable: false,
                        writable: false,
                        error: true,
                        hangup: true,
                    },
                    *generation,
                    true,
                ));
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
                false,
            ));
        }

        drop(registrations);
        poll_buffer.finish();
        Ok(events_out)
    }
}

impl Reactor for WsaPollReactor {
    fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        self.replace_registration(fd, interest)
            .map_err(PlatformUpdateFailure::into_error)
    }

    fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        lock_mutex(&self.registrations).remove(fd as usize);
        Ok(())
    }

    fn poll_events(&self, timeout: Option<Duration>) -> io::Result<Vec<Event>> {
        self.poll_events_with(timeout, |event, _generation, _invalidated| event)
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
    use crate::reactor::socket_owner::SocketLease;
    use std::io::Read;
    use std::net::{TcpListener, TcpStream};
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
            lock_mutex(&reactor.registrations).is_empty(),
            "closed socket must be self-cleaned from the interest set"
        );
    }

    #[test]
    fn stale_snapshot_cannot_remove_or_match_reused_socket() {
        let socket = 41;
        let mut registrations = RegistrationTable::default();
        let first_generation = registrations.issue_generation().expect("first generation");
        registrations.commit(socket, Interest::READABLE, first_generation);
        let stale_generation = registrations
            .get(socket)
            .expect("first registration exists")
            .generation;

        let replacement_generation = registrations
            .issue_generation()
            .expect("replacement generation");
        registrations.commit(socket, Interest::WRITABLE, replacement_generation);
        let current_generation = registrations
            .get(socket)
            .expect("replacement registration exists")
            .generation;

        assert_ne!(stale_generation, current_generation);
        assert!(!registrations.is_current(socket, stale_generation));
        assert!(!registrations.remove_if_current(socket, stale_generation));
        assert!(registrations.is_current(socket, current_generation));
    }

    #[test]
    fn owned_snapshot_defers_close_until_kernel_return() {
        let reactor = WsaPollReactor::new().expect("reactor");
        let mut socket = Some(Arc::new(
            UdpSocket::bind("127.0.0.1:0").expect("owned socket bind"),
        ));
        socket
            .as_ref()
            .expect("owned socket")
            .set_nonblocking(true)
            .expect("owned socket nonblocking");
        let weak = Arc::downgrade(socket.as_ref().expect("owned socket"));
        let owner = SocketLease::from(socket.as_ref().expect("owned socket"));
        let fd = owner.raw_socket() as RawFd;
        reactor
            .register_owned_waiter(fd, Interest::READABLE, owner.downgrade())
            .map_err(PlatformUpdateFailure::into_error)
            .expect("owned waiter registration");
        drop(owner);

        let events = reactor
            .poll_registered_events_after_snapshot(Some(Duration::ZERO), || {
                drop(socket.take());
                assert!(
                    weak.upgrade().is_some(),
                    "snapshot lease must retain the real socket during WSAPoll"
                );
            })
            .expect("owned snapshot poll");
        assert!(events.iter().all(|event| event.descriptor() != fd));
        assert!(
            weak.upgrade().is_none(),
            "last snapshot lease must release after WSAPoll"
        );

        let invalidated = reactor
            .poll_registered_events(Some(Duration::ZERO))
            .expect("expired owner cleanup");
        assert!(
            invalidated
                .iter()
                .any(|event| { event.descriptor() == fd && event.was_invalidated() })
        );
        assert!(!reactor.has_registration(fd));
    }

    #[test]
    fn tcp_snapshot_defers_peer_eof_until_kernel_return() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener bind");
        let mut peer = TcpStream::connect(listener.local_addr().expect("listener address"))
            .expect("peer connect");
        peer.set_nonblocking(true).expect("peer nonblocking");
        let (owned, _) = listener.accept().expect("owned accept");
        owned.set_nonblocking(true).expect("owned nonblocking");
        let mut owned = Some(Arc::new(owned));
        let weak = Arc::downgrade(owned.as_ref().expect("owned stream"));
        let owner = SocketLease::from(owned.as_ref().expect("owned stream"));
        let fd = owner.raw_socket() as RawFd;
        let reactor = WsaPollReactor::new().expect("reactor");
        reactor
            .register_owned_waiter(fd, Interest::READABLE, owner.downgrade())
            .map_err(PlatformUpdateFailure::into_error)
            .expect("owned stream registration");
        drop(owner);

        reactor
            .poll_registered_events_after_snapshot(Some(Duration::ZERO), || {
                drop(owned.take());
                assert!(weak.upgrade().is_some());
                let mut byte = [0_u8; 1];
                assert_eq!(
                    peer.read(&mut byte)
                        .expect_err("snapshot lease must defer peer EOF")
                        .kind(),
                    io::ErrorKind::WouldBlock
                );
            })
            .expect("owned stream snapshot");
        assert!(weak.upgrade().is_none());

        peer.set_nonblocking(false).expect("peer blocking mode");
        peer.set_read_timeout(Some(Duration::from_secs(2)))
            .expect("peer EOF deadline");
        let mut byte = [0_u8; 1];
        assert_eq!(
            peer.read(&mut byte)
                .expect("peer EOF after snapshot release"),
            0
        );
    }

    #[test]
    fn retired_owner_does_not_hide_live_socket_readiness() {
        let reactor = WsaPollReactor::new().expect("reactor");
        let mut retired = Some(Arc::new(
            UdpSocket::bind("127.0.0.1:0").expect("retired socket bind"),
        ));
        let live = Arc::new(UdpSocket::bind("127.0.0.1:0").expect("live socket bind"));
        live.set_nonblocking(true).expect("live socket nonblocking");
        let retired_owner = SocketLease::from(retired.as_ref().expect("retired socket"));
        let retired_fd = retired_owner.raw_socket() as RawFd;
        let retired_weak = Arc::downgrade(retired.as_ref().expect("retired socket"));
        let live_owner = SocketLease::from(&live);
        let live_fd = live_owner.raw_socket() as RawFd;
        reactor
            .register_owned_waiter(retired_fd, Interest::READABLE, retired_owner.downgrade())
            .map_err(PlatformUpdateFailure::into_error)
            .expect("retired registration");
        reactor
            .register_owned_waiter(live_fd, Interest::READABLE, live_owner.downgrade())
            .map_err(PlatformUpdateFailure::into_error)
            .expect("live registration");
        drop(retired_owner);
        drop(live_owner);
        let sender = UdpSocket::bind("127.0.0.1:0").expect("sender bind");
        assert_eq!(
            sender
                .send_to(b"ready", live.local_addr().expect("live address"))
                .expect("readiness datagram"),
            5
        );

        let events = reactor
            .poll_registered_events_after_snapshot(Some(Duration::from_secs(1)), || {
                drop(retired.take());
                assert!(retired_weak.upgrade().is_some());
            })
            .expect("mixed snapshot poll");
        assert!(
            events
                .iter()
                .any(|event| { event.descriptor() == live_fd && event.event().readable })
        );
        assert!(retired_weak.upgrade().is_none());
        let mut payload = [0_u8; 5];
        assert_eq!(live.recv(&mut payload).expect("live payload"), 5);
        assert_eq!(&payload, b"ready");
    }

    #[test]
    fn snapshot_error_releases_owner_after_unlock() {
        let reactor = WsaPollReactor::new().expect("reactor");
        let mut socket = Some(Arc::new(
            UdpSocket::bind("127.0.0.1:0").expect("owned socket bind"),
        ));
        let weak = Arc::downgrade(socket.as_ref().expect("owned socket"));
        let owner = SocketLease::from(socket.as_ref().expect("owned socket"));
        let fd = owner.raw_socket() as RawFd;
        reactor
            .register_owned_waiter(fd, Interest::READABLE, owner.downgrade())
            .map_err(PlatformUpdateFailure::into_error)
            .expect("owned waiter registration");
        drop(owner);

        let result = reactor.poll_registered_events_after_snapshot_error(|| {
            drop(socket.take());
            assert!(weak.upgrade().is_some());
        });
        let Err(error) = result else {
            panic!("unsupported WSAPoll event flags must fail");
        };
        assert_eq!(error.raw_os_error(), Some(10022));
        assert!(
            weak.upgrade().is_none(),
            "error path releases the final snapshot owner"
        );
        assert!(reactor.poll_buffer.try_lock().is_ok());
    }
}
