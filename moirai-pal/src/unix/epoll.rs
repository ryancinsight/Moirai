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

use crate::reactor::registration::{
    PlatformUpdateFailure, PolledEvent, RegistrationGeneration, RegistrationTable,
};
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
    /// Current descriptor generations. `epoll_event::u64` carries the same
    /// generation so a result remains attributable after descriptor reuse.
    registrations: Mutex<RegistrationTable<RawFd>>,
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
            // Registration generations start at one; zero is reserved for the
            // reactor's own wake descriptor.
            u64: 0,
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
            registrations: Mutex::new(RegistrationTable::default()),
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

    fn update_epoll_interest(
        &self,
        operation: libc::c_int,
        fd: RawFd,
        interest: Interest,
        generation: RegistrationGeneration,
    ) -> io::Result<()> {
        let mut event = libc::epoll_event {
            events: Self::interest_to_epoll_events(interest),
            u64: generation.get() as u64,
        };
        let event_ptr = if operation == libc::EPOLL_CTL_DEL {
            std::ptr::null_mut()
        } else {
            &mut event as *mut libc::epoll_event
        };
        // SAFETY: `epoll_fd` is live, `fd` is caller-owned, and non-delete
        // operations receive a pointer to an initialized event that outlives
        // the syscall. Delete ignores its event argument.
        let result = unsafe { libc::epoll_ctl(self.epoll_fd, operation, fd, event_ptr) };
        if result < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    pub(crate) fn is_current_polled_event(&self, event: &PolledEvent) -> bool {
        lock_mutex(&self.registrations).is_current(event.event().fd, event.generation())
    }

    pub(crate) fn update_registration(
        &self,
        fd: RawFd,
        interest: Interest,
    ) -> Result<(), PlatformUpdateFailure> {
        let mut registrations = lock_mutex(&self.registrations);
        let Some(current) = registrations.get(fd) else {
            return Err(PlatformUpdateFailure::new(
                io::Error::new(io::ErrorKind::NotFound, "epoll registration is absent"),
                None,
            ));
        };
        let operation = if interest.readable || interest.writable {
            libc::EPOLL_CTL_MOD
        } else {
            libc::EPOLL_CTL_DEL
        };
        match self.update_epoll_interest(operation, fd, interest, current.generation) {
            Ok(()) => {
                if operation == libc::EPOLL_CTL_DEL {
                    registrations.remove(fd);
                } else {
                    let updated = registrations.update_interest(fd, current.generation, interest);
                    debug_assert!(updated, "registration remained locked during update");
                }
                Ok(())
            }
            Err(error) => {
                let registration_absent = registration_is_absent(&error);
                let armed_interest = if registration_absent {
                    registrations.remove(fd);
                    None
                } else {
                    Some(current.interest)
                };
                Err(PlatformUpdateFailure::new(error, armed_interest))
            }
        }
    }

    pub(crate) fn replace_registration(
        &self,
        fd: RawFd,
        interest: Interest,
    ) -> Result<(), PlatformUpdateFailure> {
        let mut registrations = lock_mutex(&self.registrations);
        let current = registrations.get(fd);
        let generation = registrations.issue_generation().map_err(|error| {
            PlatformUpdateFailure::new(error, current.map(|entry| entry.interest))
        })?;
        let operation = if current.is_some() {
            libc::EPOLL_CTL_MOD
        } else {
            libc::EPOLL_CTL_ADD
        };
        match self.update_epoll_interest(operation, fd, interest, generation) {
            Ok(()) => {
                registrations.commit(fd, interest, generation);
                Ok(())
            }
            Err(error) if operation == libc::EPOLL_CTL_MOD && registration_is_absent(&error) => {
                registrations.remove(fd);
                self.update_epoll_interest(libc::EPOLL_CTL_ADD, fd, interest, generation)
                    .map(|()| registrations.commit(fd, interest, generation))
                    .map_err(|error| PlatformUpdateFailure::new(error, None))
            }
            Err(error) => Err(PlatformUpdateFailure::new(
                error,
                current.map(|entry| entry.interest),
            )),
        }
    }

    pub(crate) fn poll_registered_events(
        &self,
        timeout: Option<Duration>,
    ) -> io::Result<Vec<PolledEvent>> {
        self.poll_events_with(timeout, PolledEvent::new)
    }

    fn poll_events_with<T>(
        &self,
        timeout: Option<Duration>,
        mut make_event: impl FnMut(Event, RegistrationGeneration) -> T,
    ) -> io::Result<Vec<T>> {
        // Reuse the persistent output buffer; the mutex serializes concurrent
        // pollers (the reactor is driven by one event-loop thread in practice).
        let mut events = lock_mutex(&self.event_buffer);

        let timeout_ms = match timeout {
            Some(duration) => duration.as_millis().min(libc::c_int::MAX as u128) as libc::c_int,
            None => -1,
        };

        // SAFETY: `epoll_fd` is live and `events` provides initialized writable
        // storage for exactly `MAX_EVENTS` entries for the duration of the call.
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
        let registrations = lock_mutex(&self.registrations);
        for event in events.iter().take(num_events as usize) {
            let raw_generation = event.u64 as usize;
            let Some(generation) = RegistrationGeneration::from_raw(raw_generation) else {
                drain_eventfd(self.wake_fd)?;
                continue;
            };
            let Some(fd) = registrations.key_for_generation(generation) else {
                continue;
            };
            result.push(make_event(
                Self::epoll_events_to_event(fd, event.events),
                generation,
            ));
        }
        Ok(result)
    }
}

impl Reactor for EpollReactor {
    fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        let mut registrations = lock_mutex(&self.registrations);
        let generation = registrations.issue_generation()?;
        self.update_epoll_interest(libc::EPOLL_CTL_ADD, fd, interest, generation)?;
        registrations.commit(fd, interest, generation);
        Ok(())
    }

    fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        let mut registrations = lock_mutex(&self.registrations);
        let Some(current) = registrations.get(fd) else {
            return Ok(());
        };
        self.update_epoll_interest(
            libc::EPOLL_CTL_DEL,
            fd,
            current.interest,
            current.generation,
        )?;
        registrations.remove(fd);
        Ok(())
    }

    fn poll_events(&self, timeout: Option<Duration>) -> io::Result<Vec<Event>> {
        self.poll_events_with(timeout, |event, _generation| event)
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

fn registration_is_absent(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(code) if code == libc::EBADF || code == libc::ENOENT || code == libc::EPERM
    )
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
