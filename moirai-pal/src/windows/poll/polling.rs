//! `WSAPoll` execution: snapshotting registrations, invoking the kernel call,
//! and translating `revents` into readiness events.

use std::io;
use std::time::Duration;

use windows::Win32::Networking::WinSock::{
    POLLERR, POLLHUP, POLLNVAL, POLLRDNORM, POLLWRNORM, SOCKET, SOCKET_ERROR, WSAGetLastError,
    WSAPOLL_EVENT_FLAGS, WSAPOLLFD, WSAPoll,
};

use crate::reactor::registration::{PolledEvent, RegistrationGeneration};
use crate::{Event, RawFd};

use super::types::{PollSnapshot, WsaPollReactor, lock_mutex};

impl WsaPollReactor {
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

    pub(super) fn poll_events_with<T>(
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
