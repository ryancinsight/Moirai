//! kqueue reactor for macOS and BSD targets.

use std::{
    collections::HashMap,
    io, ptr,
    sync::{Mutex, MutexGuard},
    time::Duration,
};

use crate::{Event, Interest, RawFd, Reactor};

const EVENT_CAPACITY: usize = 1024;
const WAKE_IDENT: usize = usize::MAX;

/// kqueue-backed reactor for BSD-style event notification.
pub struct KqueueReactor {
    kqueue_fd: RawFd,
    interests: Mutex<HashMap<RawFd, Interest>>,
}

impl KqueueReactor {
    /// Create a new kqueue reactor.
    pub fn new() -> io::Result<Self> {
        // Safety: `kqueue` has no preconditions and returns a new descriptor or
        // `-1` with errno set.
        let kqueue_fd = unsafe { libc::kqueue() };
        if kqueue_fd < 0 {
            return Err(io::Error::last_os_error());
        }

        let reactor = Self {
            kqueue_fd,
            interests: Mutex::new(HashMap::new()),
        };
        reactor.register_waker()?;
        Ok(reactor)
    }

    fn register_waker(&self) -> io::Result<()> {
        let mut change = user_event(WAKE_IDENT, libc::EV_ADD | libc::EV_CLEAR, 0);
        submit_changes(self.kqueue_fd, std::slice::from_mut(&mut change))
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
        let mut changes = Vec::with_capacity(2);
        if interest.readable {
            changes.push(fd_event(fd, libc::EVFILT_READ, libc::EV_ADD));
        }
        if interest.writable {
            changes.push(fd_event(fd, libc::EVFILT_WRITE, libc::EV_ADD));
        }

        submit_changes(self.kqueue_fd, &mut changes)?;
        lock_mutex(&self.interests).insert(fd, interest);
        Ok(())
    }

    fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        let interest = lock_mutex(&self.interests).remove(&fd);
        let Some(interest) = interest else {
            return Ok(());
        };

        let mut changes = Vec::with_capacity(2);
        if interest.readable {
            changes.push(fd_event(fd, libc::EVFILT_READ, libc::EV_DELETE));
        }
        if interest.writable {
            changes.push(fd_event(fd, libc::EVFILT_WRITE, libc::EV_DELETE));
        }

        submit_changes(self.kqueue_fd, &mut changes)
    }

    fn poll_events(&self, timeout: Option<Duration>) -> io::Result<Vec<Event>> {
        let mut events = vec![zeroed_event(); EVENT_CAPACITY];
        let timeout_spec = timeout.map(duration_to_timespec);
        let timeout_ptr = timeout_spec
            .as_ref()
            .map_or(ptr::null(), |spec| spec as *const libc::timespec);

        // Safety: `events` points to writable storage for `EVENT_CAPACITY`
        // entries. `timeout_ptr` is either null or points to a stack timespec
        // valid for the duration of the syscall.
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
        for event in events.into_iter().take(ready as usize) {
            if event.ident == WAKE_IDENT {
                continue;
            }

            output.push(Event {
                fd: event.ident as RawFd,
                readable: event.filter == libc::EVFILT_READ,
                writable: event.filter == libc::EVFILT_WRITE,
                error: (event.flags & libc::EV_ERROR) != 0,
                hangup: (event.flags & libc::EV_EOF) != 0,
            });
        }

        Ok(output)
    }

    fn wake(&self) -> io::Result<()> {
        let mut change = user_event(WAKE_IDENT, 0, libc::NOTE_TRIGGER);
        submit_changes(self.kqueue_fd, std::slice::from_mut(&mut change))
    }
}

impl Drop for KqueueReactor {
    fn drop(&mut self) {
        // Safety: `kqueue_fd` is owned by this reactor and closed once in drop.
        let _ = unsafe { libc::close(self.kqueue_fd) };
    }
}

fn submit_changes(kqueue_fd: RawFd, changes: &mut [libc::kevent]) -> io::Result<()> {
    if changes.is_empty() {
        return Ok(());
    }

    // Safety: `changes` points to initialized kevent entries. No event output is
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

fn fd_event(fd: RawFd, filter: i16, flags: u16) -> libc::kevent {
    let mut event = zeroed_event();
    event.ident = fd as usize;
    event.filter = filter;
    event.flags = flags;
    event
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
    // Safety: `kevent` is a plain C struct where all-zero is a valid baseline
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
