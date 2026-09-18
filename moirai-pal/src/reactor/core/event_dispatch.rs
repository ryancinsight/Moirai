//! Readiness event handling: consuming platform events and waking the
//! matching registered wakers.

use std::io;
use std::sync::atomic::Ordering;

use super::super::registration::PlatformUpdateFailure;
#[cfg(any(unix, windows))]
use super::super::registration::PolledEvent;
use super::types::{FdKey, IoReactor};
#[cfg(not(any(unix, windows)))]
use crate::Reactor;
use crate::{Event, Interest, PlatformReactor, RawFd};

impl IoReactor {
    /// Handle a single I/O event.
    #[cfg(not(any(unix, windows)))]
    fn handle_event(&self, event: Event) -> io::Result<()> {
        // Update metrics
        self.metrics
            .events_processed
            .fetch_add(1, Ordering::Relaxed);

        // Consume matching one-shot interests before waking their tasks. A
        // task that still observes WouldBlock re-arms its interest on re-poll.
        self.wake_fd_waiters(event)
    }

    /// Handle readiness paired with its platform registration generation.
    #[cfg(any(unix, windows))]
    pub(in crate::reactor) fn handle_polled_event(&self, event: PolledEvent) -> io::Result<()> {
        self.metrics
            .events_processed
            .fetch_add(1, Ordering::Relaxed);

        #[cfg(windows)]
        if event.was_invalidated() {
            return self.wake_invalidated_waiters(event);
        }
        let readiness = event.event().clone();
        self.wake_fd_waiters_if_current(readiness, |platform| {
            platform.is_current_polled_event(&event)
        })
    }

    #[cfg(windows)]
    fn wake_invalidated_waiters(&self, event: PolledEvent) -> io::Result<()> {
        let key = FdKey::from(event.event().fd);
        let mut fds = self
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if !fds.contains_key(&key) {
            return Ok(());
        }
        let mut platform_generations = self
            .platform_generations
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if platform_generations.get(&key).copied() != Some(event.generation()) {
            return Ok(());
        }
        platform_generations.remove(&key);
        self.waiter_cancellations.clear_interest(key, true, true);
        let Some(mut fd_info) = fds.remove(&key) else {
            return Ok(());
        };
        drop(platform_generations);
        drop(fds);

        if let Some(waker) = fd_info.read_waker.take() {
            waker.wake();
        }
        if let Some(waker) = fd_info.write_waker.take() {
            waker.wake();
        }
        Ok(())
    }

    /// Wake tasks waiting on a specific file descriptor event.
    #[cfg(any(not(any(unix, windows)), test))]
    pub(in crate::reactor) fn wake_fd_waiters(&self, event: Event) -> io::Result<()> {
        self.wake_fd_waiters_if_current(event, |_| true)
    }

    fn wake_fd_waiters_if_current(
        &self,
        event: Event,
        is_current: impl FnOnce(&PlatformReactor) -> bool,
    ) -> io::Result<()> {
        self.wake_fd_waiters_with_platform(event, is_current, |_, fd, interest| {
            self.update_platform_registration(fd, interest)
        })
    }

    pub(super) fn update_platform_registration(
        &self,
        fd: RawFd,
        interest: Interest,
    ) -> Result<(), PlatformUpdateFailure> {
        #[cfg(any(unix, windows))]
        {
            self.platform_reactor.update_registration(fd, interest)
        }
        #[cfg(not(any(unix, windows)))]
        {
            self.platform_reactor
                .unregister_fd(fd)
                .map_err(|error| PlatformUpdateFailure::new(error, None))?;
            if interest.readable || interest.writable {
                self.platform_reactor
                    .register_fd(fd, interest)
                    .map_err(|error| PlatformUpdateFailure::new(error, None))?;
            }
            Ok(())
        }
    }

    pub(in crate::reactor) fn wake_fd_waiters_with_platform(
        &self,
        event: Event,
        is_current: impl FnOnce(&PlatformReactor) -> bool,
        update_platform: impl FnOnce(
            &PlatformReactor,
            RawFd,
            Interest,
        ) -> Result<(), PlatformUpdateFailure>,
    ) -> io::Result<()> {
        let key = FdKey::from(event.fd);
        let mut fds = self
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let Some(fd_info) = fds.get_mut(&key) else {
            return Ok(());
        };
        if !is_current(&self.platform_reactor) {
            return Ok(());
        }
        fd_info.event_count += 1;

        let consume_read =
            fd_info.interest.readable && (event.readable || event.error || event.hangup);
        let consume_write =
            fd_info.interest.writable && (event.writable || event.error || event.hangup);
        if !consume_read && !consume_write {
            return Ok(());
        }

        let remaining = Interest {
            readable: fd_info.interest.readable && !consume_read,
            writable: fd_info.interest.writable && !consume_write,
            error: fd_info.interest.error,
        };

        // Every registered readiness interest is one-shot at this layer. The
        // platform backends are level-triggered only to close the syscall-to-
        // registration race; retaining a delivered writable interest would
        // otherwise make the event loop spin indefinitely.
        let platform_result = update_platform(&self.platform_reactor, event.fd, remaining);

        let fd_info = fds
            .get_mut(&key)
            .expect("fd registration remained locked during readiness update");
        let read_waker = if consume_read {
            #[cfg(windows)]
            self.waiter_cancellations.clear_interest(key, true, false);
            fd_info.read_waker.take()
        } else {
            None
        };
        let write_waker = if consume_write {
            #[cfg(windows)]
            self.waiter_cancellations.clear_interest(key, false, true);
            fd_info.write_waker.take()
        } else {
            None
        };
        let mut stranded_read_waker = None;
        let mut stranded_write_waker = None;
        let (remove_registration, platform_error) = match platform_result {
            Ok(()) => {
                if remaining.readable || remaining.writable {
                    fd_info.interest = remaining;
                    (false, None)
                } else {
                    (true, None)
                }
            }
            Err(failure) => {
                // A delivered waiter must observe its wake even when the
                // backend transition fails. Wake every remaining waiter too:
                // each one must re-poll and republish its interest instead of
                // depending on a transition that did not complete.
                stranded_read_waker = fd_info.read_waker.take();
                stranded_write_waker = fd_info.write_waker.take();
                if let Some(armed_interest) = failure.armed_interest() {
                    fd_info.interest = armed_interest;
                    (false, Some(failure.into_error()))
                } else {
                    (true, Some(failure.into_error()))
                }
            }
        };
        if remove_registration {
            #[cfg(windows)]
            self.platform_generations
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .remove(&key);
            fds.remove(&key);
            #[cfg(windows)]
            self.waiter_cancellations.clear_interest(key, true, true);
        }
        drop(fds);

        if let Some(waker) = read_waker {
            waker.wake();
        }
        if let Some(waker) = write_waker {
            waker.wake();
        }
        if let Some(waker) = stranded_read_waker {
            waker.wake();
        }
        if let Some(waker) = stranded_write_waker {
            waker.wake();
        }
        platform_error.map_or(Ok(()), Err)
    }
}
