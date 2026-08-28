//! kqueue reactor for macOS and BSD targets.

use std::{
    cell::RefCell,
    io, ptr,
    sync::{Mutex, MutexGuard},
    time::Duration,
};

use crate::reactor::registration::{
    PlatformUpdateFailure, PolledEvent, RegistrationGeneration, RegistrationTable,
};
use crate::{Event, Interest, RawFd, Reactor};

const EVENT_CAPACITY: usize = 1024;
const WAKE_IDENT: usize = usize::MAX;
const CHANGE_ATTEMPTS: usize = 2;

thread_local! {
    /// Per-poller output storage keeps the hot poll path allocation-free while
    /// confining libc's pointer-bearing `kevent` representation to its thread.
    static EVENT_BUFFER: RefCell<Vec<libc::kevent>> =
        RefCell::new(vec![zeroed_event(); EVENT_CAPACITY]);
}

/// kqueue-backed reactor for BSD-style event notification.
pub struct KqueueReactor {
    kqueue_fd: RawFd,
    registrations: Mutex<RegistrationTable<RawFd>>,
}

impl KqueueReactor {
    /// Create a new kqueue reactor.
    pub fn new() -> io::Result<Self> {
        // SAFETY: `kqueue` has no preconditions and returns a new descriptor or
        // `-1` with errno set.
        let kqueue_fd = unsafe { libc::kqueue() };
        if kqueue_fd < 0 {
            return Err(io::Error::last_os_error());
        }

        let reactor = Self {
            kqueue_fd,
            registrations: Mutex::new(RegistrationTable::default()),
        };
        reactor.register_waker()?;
        Ok(reactor)
    }

    fn register_waker(&self) -> io::Result<()> {
        let mut change = user_event(WAKE_IDENT, libc::EV_ADD | libc::EV_CLEAR, 0);
        submit_changes(self.kqueue_fd, std::slice::from_mut(&mut change))
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
                io::Error::new(io::ErrorKind::NotFound, "kqueue registration is absent"),
                None,
            ));
        };
        let (armed_interest, error) =
            self.transition_interest(fd, current.interest, interest, current.generation);
        if armed_interest.readable || armed_interest.writable {
            let updated = registrations.update_interest(fd, current.generation, armed_interest);
            debug_assert!(updated, "registration remained locked during update");
        } else {
            registrations.remove(fd);
        }
        error.map_or(Ok(()), |error| {
            let armed =
                (armed_interest.readable || armed_interest.writable).then_some(armed_interest);
            Err(PlatformUpdateFailure::new(error, armed))
        })
    }

    pub(crate) fn replace_registration(
        &self,
        fd: RawFd,
        interest: Interest,
    ) -> Result<(), PlatformUpdateFailure> {
        let mut registrations = lock_mutex(&self.registrations);
        if let Some(current) = registrations.get(fd) {
            let empty = Interest {
                readable: false,
                writable: false,
                error: current.interest.error,
            };
            let (residual, error) =
                self.transition_interest(fd, current.interest, empty, current.generation);
            if residual.readable || residual.writable {
                let updated = registrations.update_interest(fd, current.generation, residual);
                debug_assert!(updated, "registration remained locked during replacement");
            } else {
                registrations.remove(fd);
            }
            if let Some(error) = error {
                let armed = (residual.readable || residual.writable).then_some(residual);
                return Err(PlatformUpdateFailure::new(error, armed));
            }
        }

        let generation = registrations
            .issue_generation()
            .map_err(|error| PlatformUpdateFailure::new(error, None))?;
        let empty = Interest {
            readable: false,
            writable: false,
            error: interest.error,
        };
        let (actual, error) = self.transition_interest(fd, empty, interest, generation);
        if actual.readable || actual.writable {
            registrations.commit(fd, actual, generation);
        }
        error.map_or(Ok(()), |error| {
            let armed = (actual.readable || actual.writable).then_some(actual);
            Err(PlatformUpdateFailure::new(error, armed))
        })
    }

    pub(crate) fn poll_registered_events(
        &self,
        timeout: Option<Duration>,
    ) -> io::Result<Vec<PolledEvent>> {
        self.poll_events_with(timeout, PolledEvent::new)
    }

    fn transition_interest(
        &self,
        fd: RawFd,
        current: Interest,
        desired: Interest,
        generation: RegistrationGeneration,
    ) -> (Interest, Option<io::Error>) {
        let mut actual = current;
        let mut first_error = None;
        for (filter, current_enabled, desired_enabled) in [
            (libc::EVFILT_READ, current.readable, desired.readable),
            (libc::EVFILT_WRITE, current.writable, desired.writable),
        ] {
            if current_enabled == desired_enabled {
                continue;
            }
            let flags = if desired_enabled {
                libc::EV_ADD
            } else {
                libc::EV_DELETE
            };
            let mut change = fd_event(fd, filter, flags, generation);
            match submit_interest_change(self.kqueue_fd, &mut change) {
                Ok(()) => set_interest_filter(&mut actual, filter, desired_enabled),
                Err(error) if !desired_enabled && error.raw_os_error() == Some(libc::ENOENT) => {
                    // The filter is already absent, which is the requested
                    // postcondition; reconcile the sidecar without failing.
                    set_interest_filter(&mut actual, filter, false);
                }
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }
        (actual, first_error)
    }

    fn poll_events_with<T>(
        &self,
        timeout: Option<Duration>,
        mut make_event: impl FnMut(Event, RegistrationGeneration) -> T,
    ) -> io::Result<Vec<T>> {
        EVENT_BUFFER.with(|buffer| {
            let mut events = buffer.try_borrow_mut().map_err(|_| {
                io::Error::other("kqueue polling cannot re-enter on the same thread")
            })?;
            let timeout_spec = timeout.map(duration_to_timespec);
            let timeout_ptr = timeout_spec
                .as_ref()
                .map_or(ptr::null(), |spec| spec as *const libc::timespec);

            // SAFETY: `events` points to writable storage for `EVENT_CAPACITY`
            // entries. `timeout_ptr` is null or points to a live timespec.
            let ready = unsafe {
                libc::kevent(
                    self.kqueue_fd,
                    ptr::null(),
                    0,
                    events.as_mut_ptr(),
                    EVENT_CAPACITY as i32,
                    timeout_ptr,
                )
            };
            if ready < 0 {
                return Err(io::Error::last_os_error());
            }

            let mut output = Vec::with_capacity(ready as usize);
            let registrations = lock_mutex(&self.registrations);
            for event in events.iter().take(ready as usize) {
                if event.ident == WAKE_IDENT {
                    continue;
                }
                let Some(generation) = RegistrationGeneration::from_raw(event.udata.addr()) else {
                    continue;
                };
                let fd = event.ident as RawFd;
                if !registrations.is_current(fd, generation) {
                    continue;
                }
                output.push(make_event(
                    Event {
                        fd,
                        readable: event.filter == libc::EVFILT_READ,
                        writable: event.filter == libc::EVFILT_WRITE,
                        error: (event.flags & libc::EV_ERROR) != 0,
                        hangup: (event.flags & libc::EV_EOF) != 0,
                    },
                    generation,
                ));
            }
            Ok(output)
        })
    }
}

impl Reactor for KqueueReactor {
    fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        if fd < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid file descriptor for kqueue registration",
            ));
        }

        // Level-triggered (no EV_CLEAR): the reactor registers a waker only after
        // a `WouldBlock`, so an edge-triggered filter would drop readiness that
        // arrived in the register window (or was already present at registration),
        // hanging the task. Without EV_CLEAR, kevent re-reports readiness until
        // the I/O consumes it — self-healing the race. (The WAKE_IDENT user event
        // keeps EV_CLEAR: it is a one-shot self-wake, not fd readiness.)
        let mut registrations = lock_mutex(&self.registrations);
        let generation = registrations.issue_generation()?;
        let empty = Interest {
            readable: false,
            writable: false,
            error: interest.error,
        };
        let (actual, error) = self.transition_interest(fd, empty, interest, generation);
        if let Some(error) = error {
            // Registration is transactional at the trait boundary. If one
            // filter was added before another failed, remove the successful
            // subset before reporting failure. Retain the exact residual only
            // if that compensating removal itself fails, so later polling can
            // still identify and reconcile it.
            let (residual, cleanup_error) = self.transition_interest(fd, actual, empty, generation);
            if residual.readable || residual.writable {
                registrations.commit(fd, residual, generation);
            }
            return Err(cleanup_error.unwrap_or(error));
        }
        if actual.readable || actual.writable {
            registrations.commit(fd, actual, generation);
        }
        Ok(())
    }

    fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        let mut registrations = lock_mutex(&self.registrations);
        let Some(current) = registrations.get(fd) else {
            return Ok(());
        };
        let empty = Interest {
            readable: false,
            writable: false,
            error: current.interest.error,
        };
        let (actual, error) =
            self.transition_interest(fd, current.interest, empty, current.generation);
        if actual.readable || actual.writable {
            let updated = registrations.update_interest(fd, current.generation, actual);
            debug_assert!(updated, "registration remained locked during removal");
        } else {
            registrations.remove(fd);
        }
        error.map_or(Ok(()), Err)
    }

    fn poll_events(&self, timeout: Option<Duration>) -> io::Result<Vec<Event>> {
        self.poll_events_with(timeout, |event, _generation| event)
    }

    fn wake(&self) -> io::Result<()> {
        let mut change = user_event(WAKE_IDENT, 0, libc::NOTE_TRIGGER);
        submit_changes(self.kqueue_fd, std::slice::from_mut(&mut change))
    }
}

impl Drop for KqueueReactor {
    fn drop(&mut self) {
        // SAFETY: `kqueue_fd` is owned by this reactor and closed once in drop.
        let _ = unsafe { libc::close(self.kqueue_fd) };
    }
}

fn submit_changes(kqueue_fd: RawFd, changes: &mut [libc::kevent]) -> io::Result<()> {
    if changes.is_empty() {
        return Ok(());
    }

    // SAFETY: `changes` points to initialized kevent entries. No event output is
    // requested, so the event list pointer is null.
    let result = unsafe {
        libc::kevent(
            kqueue_fd,
            changes.as_ptr(),
            changes.len() as i32,
            ptr::null_mut(),
            0,
            ptr::null(),
        )
    };

    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn submit_interest_change(kqueue_fd: RawFd, change: &mut libc::kevent) -> io::Result<()> {
    change.flags |= libc::EV_RECEIPT;
    for attempt in 0..CHANGE_ATTEMPTS {
        let mut receipt = zeroed_event();
        // SAFETY: `change` and `receipt` each point to one initialized kevent.
        // EV_RECEIPT makes the kernel return the per-change status in `receipt`.
        let result = unsafe { libc::kevent(kqueue_fd, change, 1, &mut receipt, 1, ptr::null()) };
        if result < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted && attempt == 0 {
                // The kernel may have applied the change before EINTR. One
                // identical retry resolves that ambiguity: EV_ADD updates an
                // existing filter, and a repeated EV_DELETE reports ENOENT,
                // which the caller treats as the requested absent state.
                continue;
            }
            return Err(error);
        }
        if result != 1 || (receipt.flags & libc::EV_ERROR) == 0 {
            return Err(io::Error::other(
                "kqueue did not return the requested change receipt",
            ));
        }
        if receipt.data == 0 {
            return Ok(());
        }
        let errno = i32::try_from(receipt.data)
            .map_err(|_| io::Error::other("kqueue receipt returned an invalid errno"))?;
        return Err(io::Error::from_raw_os_error(errno));
    }
    unreachable!("bounded kqueue change attempts return from every iteration")
}

fn fd_event(
    fd: RawFd,
    filter: i16,
    flags: u16,
    generation: RegistrationGeneration,
) -> libc::kevent {
    let mut event = zeroed_event();
    event.ident = fd as usize;
    event.filter = filter;
    event.flags = flags;
    event.udata = ptr::without_provenance_mut(generation.get());
    event
}

fn set_interest_filter(interest: &mut Interest, filter: i16, enabled: bool) {
    if filter == libc::EVFILT_READ {
        interest.readable = enabled;
    } else {
        debug_assert_eq!(filter, libc::EVFILT_WRITE);
        interest.writable = enabled;
    }
}

fn user_event(ident: usize, flags: u16, fflags: u32) -> libc::kevent {
    let mut event = zeroed_event();
    event.ident = ident;
    event.filter = libc::EVFILT_USER;
    event.flags = flags;
    event.fflags = fflags;
    event
}

fn zeroed_event() -> libc::kevent {
    // SAFETY: `kevent` is a plain C struct where all-zero is a valid baseline
    // before fields are assigned.
    unsafe { std::mem::zeroed() }
}

fn duration_to_timespec(duration: Duration) -> libc::timespec {
    libc::timespec {
        // Saturate: `as_secs()` is u64; a multi-century timeout would wrap a raw
        // `as time_t` to a negative value where `time_t` is 32-bit.
        tv_sec: duration.as_secs().min(libc::time_t::MAX as u64) as libc::time_t,
        tv_nsec: duration.subsec_nanos() as libc::c_long,
    }
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
