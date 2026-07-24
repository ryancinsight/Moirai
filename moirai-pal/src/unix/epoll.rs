//! Linux epoll-based async I/O reactor.
//!
//! [`EpollReactor`] multiplexes readiness for the async executor: callers
//! register descriptors with [`Reactor::register_fd`], drive
//! [`Reactor::poll_events`] on an event-loop thread, and interrupt a blocking
//! wait with [`Reactor::wake`]. Everything here is raw `libc` syscalls, so the
//! invariants that keep it sound are stated once below and referenced by the
//! per-site `SAFETY` comments.
//!
//! # Descriptor ownership
//!
//! The reactor owns exactly two descriptors — `epoll_fd` and the `eventfd` in
//! `wake_fd` — created in `new` and closed once in `Drop`. Both are `CLOEXEC`.
//! Every failure path in `new` closes what it already created before returning,
//! so a failed construction leaks nothing. Because `Drop` takes `&mut self`, no
//! other method can be running against those descriptors when they are closed.
//!
//! Registered descriptors are *not* owned: they belong to the sockets and pipes
//! the caller registers. A closed descriptor is removed from the interest list
//! by the kernel, and passing a stale one to `epoll_ctl` yields `EBADF` — an
//! error, never memory unsafety.
//!
//! # Syscall buffers
//!
//! `epoll_wait` writes up to `MAX_EVENTS` entries into `event_buffer`, which is
//! allocated at exactly that length and reused, so the pointer/length pair the
//! kernel receives always describes live, initialized memory. Only the first
//! `num_events` entries are read back. The mutex serializes concurrent pollers;
//! the reactor is driven by one event-loop thread in practice, and a blocking
//! wait is interrupted by `wake` rather than by a competing poller.
//!
//! `libc::epoll_event` is `repr(packed)` on x86-64, so its fields are read by
//! value (`event.u64`, `event.events`) and never borrowed — taking a reference
//! to a packed field is undefined behavior, and the compiler rejects it.
//!
//! # errno discipline
//!
//! `errno` is meaningful only after a syscall reports failure, and is otherwise
//! whatever the last failing call left behind. Every wrapper here therefore
//! tests the return value for the failure sentinel *first* and calls
//! `io::Error::last_os_error` only on that branch.
//!
//! # Readiness model
//!
//! Registrations are level-triggered; `interest_to_epoll_events` documents why
//! edge-triggered would lose readiness that arrives before a waker is installed.

use std::io;
use std::os::unix::io::RawFd;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use crate::{Event, Interest, Reactor};

/// Linux epoll-based I/O reactor.
pub struct EpollReactor {
    /// epoll file descriptor
    epoll_fd: RawFd,
    /// eventfd used to wake `epoll_wait`
    wake_fd: RawFd,
    /// Reused `epoll_wait` output buffer (`MAX_EVENTS` entries), so the hot
    /// poll loop does not allocate a fresh 1024-entry vector per iteration.
    event_buffer: Mutex<Vec<libc::epoll_event>>,
}

// Constants for epoll operations
const MAX_EVENTS: usize = 1024;
const EPOLL_CREATE_FLAGS: libc::c_int = libc::EPOLL_CLOEXEC;

impl EpollReactor {
    /// Create a new epoll-based reactor.
    pub fn new() -> io::Result<Self> {
        // SAFETY: `epoll_create1` takes only a flag word and returns a fresh
        // descriptor or -1; it touches no caller memory.
        let epoll_fd = unsafe { libc::epoll_create1(EPOLL_CREATE_FLAGS) };

        if epoll_fd < 0 {
            return Err(io::Error::last_os_error());
        }

        // SAFETY: as above — `eventfd` takes an initial count and flags only.
        let wake_fd = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        if wake_fd < 0 {
            let error = io::Error::last_os_error();
            // SAFETY: `epoll_fd` was created above, is still open, and is
            // unreachable from anywhere else because `self` does not exist yet.
            unsafe {
                libc::close(epoll_fd);
            }
            return Err(error);
        }

        let mut wake_event = libc::epoll_event {
            events: libc::EPOLLIN as u32,
            u64: wake_fd as u64,
        };
        // SAFETY: both descriptors are open, and the event pointer refers to
        // `wake_event`, a live initialized local that outlives the call. The
        // kernel copies the struct; it does not retain the pointer.
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
            // SAFETY: both descriptors were created above and are still open;
            // `self` does not exist, so nothing else can reach them.
            unsafe {
                libc::close(wake_fd);
                libc::close(epoll_fd);
            }
            return Err(error);
        }

        Ok(Self {
            epoll_fd,
            wake_fd,
            event_buffer: Mutex::new(vec![libc::epoll_event { events: 0, u64: 0 }; MAX_EVENTS]),
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

        // SAFETY: `epoll_fd` is open for the lifetime of `self`, and the event
        // pointer refers to `event`, a live initialized local that outlives the
        // call. `fd` is the caller's; if it is stale the kernel reports `EBADF`.
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
        // SAFETY: `epoll_fd` is open for the lifetime of `self`. `EPOLL_CTL_DEL`
        // reads no event struct, so a null pointer is the documented argument.
        let result = unsafe {
            libc::epoll_ctl(self.epoll_fd, libc::EPOLL_CTL_DEL, fd, std::ptr::null_mut())
        };

        if result < 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(())
    }

    fn poll_events(&self, timeout: Option<Duration>) -> io::Result<Vec<Event>> {
        // Reuse the persistent output buffer; the mutex serializes concurrent
        // pollers (the reactor is driven by one event-loop thread in practice).
        let mut events = lock_mutex(&self.event_buffer);

        let timeout_ms = match timeout {
            // Saturate: `as_millis()` is u128; a multi-week timeout would wrap a
            // raw `as c_int` to a negative value (turning a finite wait into an
            // infinite block or a garbage short timeout).
            Some(duration) => duration.as_millis().min(libc::c_int::MAX as u128) as libc::c_int,
            None => -1, // Block indefinitely
        };

        // SAFETY: `epoll_fd` is open for the lifetime of `self`, and the buffer
        // is held under the mutex guard for the whole call. It was allocated
        // with exactly `MAX_EVENTS` initialized entries, so the kernel may fill
        // up to that many without running past the allocation.
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
        // SAFETY: `wake_fd` is the reactor's own eventfd, open for the lifetime
        // of `self`, and the pointer/length pair describes `value`, a live local
        // `u64`, for exactly its own size.
        let written = unsafe {
            libc::write(
                self.wake_fd,
                (&value as *const u64).cast::<libc::c_void>(),
                std::mem::size_of::<u64>(),
            )
        };

        if written < 0 {
            let error = io::Error::last_os_error();
            // The eventfd is non-blocking: a full counter means a wake is
            // already pending, which is the state this call wanted anyway.
            if error.kind() == io::ErrorKind::WouldBlock {
                return Ok(());
            }
            return Err(error);
        }

        // An eventfd write transfers all 8 bytes or fails; there is no short write.
        debug_assert_eq!(written, std::mem::size_of::<u64>() as isize);
        Ok(())
    }
}

impl Drop for EpollReactor {
    fn drop(&mut self) {
        // SAFETY: both descriptors are owned by this reactor and closed exactly
        // once here. `&mut self` is exclusive, so no poll, registration, or wake
        // can be using them concurrently.
        unsafe {
            libc::close(self.wake_fd);
            libc::close(self.epoll_fd);
        }
    }
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn drain_eventfd(wake_fd: RawFd) -> io::Result<()> {
    let mut value = 0_u64;
    // SAFETY: `wake_fd` is the reactor's eventfd, and the pointer/length pair
    // describes `value`, a live local `u64`, for exactly its own size. The read
    // resets the counter, so a wake that arrives afterwards re-reports.
    let read = unsafe {
        libc::read(
            wake_fd,
            (&mut value as *mut u64).cast::<libc::c_void>(),
            std::mem::size_of::<u64>(),
        )
    };

    if read < 0 {
        let error = io::Error::last_os_error();
        // Counter already zero — another drain consumed this wake.
        if error.kind() == io::ErrorKind::WouldBlock {
            return Ok(());
        }
        return Err(error);
    }

    // An eventfd read returns all 8 bytes or fails; there is no short read.
    debug_assert_eq!(read, std::mem::size_of::<u64>() as isize);
    Ok(())
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
