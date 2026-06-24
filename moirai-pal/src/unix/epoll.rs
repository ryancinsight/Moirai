//! Linux epoll-based async I/O reactor implementation.
//!
//! This module provides high-performance async I/O using Linux's epoll system call,
//! which is the most efficient I/O multiplexing mechanism on Linux systems.

use std::collections::HashMap;
use std::io;
use std::os::unix::io::RawFd;
use std::time::Duration;

use crate::{Event, Interest, Reactor};

/// Linux epoll-based I/O reactor.
pub struct EpollReactor {
    /// epoll file descriptor
    epoll_fd: RawFd,
    /// eventfd used to wake `epoll_wait`
    wake_fd: RawFd,
    /// Event buffer for epoll_wait
    #[allow(dead_code)] // Used for future event processing per ADR requirements
    event_buffer: Vec<libc::epoll_event>,
    /// Mapping of file descriptors to their registered interests
    #[allow(dead_code)] // Used for future interest tracking per ADR requirements
    fd_interests: HashMap<RawFd, Interest>,
}

// Constants for epoll operations
const MAX_EVENTS: usize = 1024;
const EPOLL_CREATE_FLAGS: libc::c_int = libc::EPOLL_CLOEXEC;

impl EpollReactor {
    /// Create a new epoll-based reactor.
    pub fn new() -> io::Result<Self> {
        let epoll_fd = unsafe { libc::epoll_create1(EPOLL_CREATE_FLAGS) };

        if epoll_fd < 0 {
            return Err(io::Error::last_os_error());
        }

        let wake_fd = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        if wake_fd < 0 {
            let error = io::Error::last_os_error();
            unsafe {
                libc::close(epoll_fd);
            }
            return Err(error);
        }

        let mut wake_event = libc::epoll_event {
            events: libc::EPOLLIN as u32,
            u64: wake_fd as u64,
        };
        let register_wake = unsafe {
            libc::epoll_ctl(
                epoll_fd,
                libc::EPOLL_CTL_ADD,
                wake_fd,
                &mut wake_event as *mut libc::epoll_event,
            )
        };
        if register_wake < 0 {
            let error = io::Error::last_os_error();
            unsafe {
                libc::close(wake_fd);
                libc::close(epoll_fd);
            }
            return Err(error);
        }

        Ok(Self {
            epoll_fd,
            wake_fd,
            event_buffer: Vec::with_capacity(MAX_EVENTS),
            fd_interests: HashMap::new(),
        })
    }

    /// Convert Interest to epoll events.
    fn interest_to_epoll_events(interest: Interest) -> u32 {
        // Level-triggered (no EPOLLET): the reactor registers a waker only after
        // a `WouldBlock`, so an edge-triggered registration would lose any
        // readiness that arrived between the failed syscall and registration (or
        // any readiness already present at registration, which emits no edge),
        // hanging the task. Level-triggered re-reports readiness on every
        // `epoll_wait` until the I/O actually consumes it, which self-heals that
        // race and the `unregister`+`register` interest-widening window.
        let mut events = 0u32;

        if interest.readable {
            events |= libc::EPOLLIN as u32;
        }

        if interest.writable {
            events |= libc::EPOLLOUT as u32;
        }

        if interest.error {
            events |= libc::EPOLLERR as u32 | libc::EPOLLHUP as u32;
        }

        events
    }

    /// Convert epoll events to Event.
    fn epoll_events_to_event(fd: RawFd, events: u32) -> Event {
        Event {
            fd,
            readable: (events & libc::EPOLLIN as u32) != 0,
            writable: (events & libc::EPOLLOUT as u32) != 0,
            error: (events & libc::EPOLLERR as u32) != 0,
            hangup: (events & libc::EPOLLHUP as u32) != 0,
        }
    }
}

impl Reactor for EpollReactor {
    fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        let events = Self::interest_to_epoll_events(interest);

        let mut event = libc::epoll_event {
            events,
            u64: fd as u64,
        };

        let result = unsafe {
            libc::epoll_ctl(
                self.epoll_fd,
                libc::EPOLL_CTL_ADD,
                fd,
                &mut event as *mut libc::epoll_event,
            )
        };

        if result < 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(())
    }

    fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        let result = unsafe {
            libc::epoll_ctl(self.epoll_fd, libc::EPOLL_CTL_DEL, fd, std::ptr::null_mut())
        };

        if result < 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(())
    }

    fn poll_events(&self, timeout: Option<Duration>) -> io::Result<Vec<Event>> {
        // Ensure event buffer has the right capacity
        let mut events = vec![libc::epoll_event { events: 0, u64: 0 }; MAX_EVENTS];

        let timeout_ms = match timeout {
            // Saturate: `as_millis()` is u128; a multi-week timeout would wrap a
            // raw `as c_int` to a negative value (turning a finite wait into an
            // infinite block or a garbage short timeout).
            Some(duration) => duration.as_millis().min(libc::c_int::MAX as u128) as libc::c_int,
            None => -1, // Block indefinitely
        };

        let num_events = unsafe {
            libc::epoll_wait(
                self.epoll_fd,
                events.as_mut_ptr(),
                MAX_EVENTS as libc::c_int,
                timeout_ms,
            )
        };

        if num_events < 0 {
            return Err(io::Error::last_os_error());
        }

        let mut result = Vec::with_capacity(num_events as usize);

        // Use iterator pattern per Rust Book Ch.13 for better performance
        for event in events.iter().take(num_events as usize) {
            let fd = event.u64 as RawFd;
            if fd == self.wake_fd {
                drain_eventfd(self.wake_fd)?;
                continue;
            }

            let reactor_event = Self::epoll_events_to_event(fd, event.events);
            result.push(reactor_event);
        }

        Ok(result)
    }

    fn wake(&self) -> io::Result<()> {
        let value = 1_u64;
        let written = unsafe {
            libc::write(
                self.wake_fd,
                (&value as *const u64).cast::<libc::c_void>(),
                std::mem::size_of::<u64>(),
            )
        };

        if written == std::mem::size_of::<u64>() as isize {
            return Ok(());
        }

        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::WouldBlock {
            Ok(())
        } else {
            Err(error)
        }
    }
}

impl Drop for EpollReactor {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.wake_fd);
            libc::close(self.epoll_fd);
        }
    }
}

fn drain_eventfd(wake_fd: RawFd) -> io::Result<()> {
    let mut value = 0_u64;
    let read = unsafe {
        libc::read(
            wake_fd,
            (&mut value as *mut u64).cast::<libc::c_void>(),
            std::mem::size_of::<u64>(),
        )
    };

    if read == std::mem::size_of::<u64>() as isize {
        return Ok(());
    }

    let error = io::Error::last_os_error();
    if error.kind() == io::ErrorKind::WouldBlock {
        Ok(())
    } else {
        Err(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoll_reactor_creation() {
        let reactor = EpollReactor::new();
        assert!(reactor.is_ok());
    }

    #[test]
    fn test_interest_conversion() {
        let interest = Interest::READ_WRITE;
        let events = EpollReactor::interest_to_epoll_events(interest);

        assert_ne!(events & libc::EPOLLIN as u32, 0);
        assert_ne!(events & libc::EPOLLOUT as u32, 0);
        // Level-triggered: EPOLLET must NOT be set (see interest_to_epoll_events).
        assert_eq!(events & libc::EPOLLET as u32, 0);
    }

    #[test]
    fn test_event_conversion() {
        let fd = 5;
        let events = libc::EPOLLIN as u32 | libc::EPOLLERR as u32;
        let event = EpollReactor::epoll_events_to_event(fd, events);

        assert_eq!(event.fd, fd);
        assert!(event.readable);
        assert!(!event.writable);
        assert!(event.error);
        assert!(!event.hangup);
    }

    #[test]
    fn test_epoll_wake_returns_no_user_events() {
        let reactor = EpollReactor::new().expect("epoll reactor must be created");
        reactor.wake().expect("epoll wake must succeed");
        let events = reactor
            .poll_events(Some(Duration::from_millis(10)))
            .expect("epoll poll must succeed");
        assert!(events.is_empty());
    }
}
