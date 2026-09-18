//! Task waker registration, replacement, and removal for a file descriptor.

use std::io;
use std::sync::atomic::Ordering;
use std::task::Waker;
use std::time::Instant;

#[cfg(windows)]
use super::super::registration::PlatformUpdateFailure;
#[cfg(windows)]
use super::super::socket_owner::SocketLease;
#[cfg(windows)]
use super::super::waiter_cancellation::WaiterCancellation;
use super::types::{FdInfo, FdKey, IoReactor};
#[cfg(not(windows))]
use crate::Reactor;
use crate::{Interest, RawFd};

#[cfg(windows)]
type WakerRegistration = Option<WaiterCancellation>;
#[cfg(not(windows))]
type WakerRegistration = ();

impl IoReactor {
    /// Register a task's waker for a file descriptor and interest.
    ///
    /// # Errors
    ///
    /// Returns a platform registration error or the retained terminal driver
    /// failure after a driven event loop has stopped on an error.
    pub fn register_waker(&self, fd: RawFd, interest: Interest, waker: Waker) -> io::Result<()> {
        #[cfg(windows)]
        {
            self.register_waker_with_owner(fd, interest, waker, None)
                .map(drop)
        }
        #[cfg(not(windows))]
        {
            self.register_waker_with_owner(fd, interest, waker)
        }
    }

    #[cfg(windows)]
    pub(crate) fn register_owned_waker(
        &self,
        fd: RawFd,
        interest: Interest,
        waker: Waker,
        owner: SocketLease,
    ) -> io::Result<WaiterCancellation> {
        self.register_waker_with_owner(fd, interest, waker, Some(owner.downgrade()))?
            .ok_or_else(|| io::Error::other("owned waiter cancellation was not published"))
    }

    fn register_waker_with_owner(
        &self,
        fd: RawFd,
        interest: Interest,
        waker: Waker,
        #[cfg(windows)] owner: Option<super::super::socket_owner::WeakSocketOwner>,
    ) -> io::Result<WakerRegistration> {
        #[cfg(windows)]
        let cancellation = owner
            .as_ref()
            .map(|_| self.waiter_cancellations.reserve(fd, interest))
            .transpose()?;
        let mut fds = self
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(error) = self.driver_failure.registration_error() {
            return Err(error);
        }
        if let Some(fd_info) = fds.get_mut(&FdKey::from(fd)) {
            let mut new_interest = fd_info.interest;
            if interest.readable {
                new_interest.readable = true;
            }
            if interest.writable {
                new_interest.writable = true;
            }
            #[cfg(windows)]
            let replacement = if let Some(owner) = owner.clone() {
                self.platform_reactor.replace_owned_waiter_registration(
                    fd,
                    new_interest,
                    interest,
                    owner,
                )
            } else {
                self.platform_reactor
                    .replace_waiter_registration(fd, new_interest, interest)
            };
            #[cfg(unix)]
            let replacement = self
                .platform_reactor
                .replace_registration(fd, new_interest)
                .map(|()| true);
            #[cfg(not(any(unix, windows)))]
            let replacement = self
                .update_platform_registration(fd, new_interest)
                .map(|()| true);
            let replaced_existing = match replacement {
                #[cfg(windows)]
                Ok(replacement) => {
                    self.platform_generations
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner())
                        .insert(FdKey::from(fd), replacement.generation);
                    replacement.replaced_existing
                }
                #[cfg(not(windows))]
                Ok(replaced_existing) => replaced_existing,
                Err(failure) => {
                    let read_waker = fd_info.read_waker.take();
                    let write_waker = fd_info.write_waker.take();
                    let remove_registration = if let Some(armed_interest) = failure.armed_interest()
                    {
                        fd_info.interest = armed_interest;
                        false
                    } else {
                        true
                    };
                    let error = failure.into_error();
                    if remove_registration {
                        #[cfg(windows)]
                        self.platform_generations
                            .lock()
                            .unwrap_or_else(|poison| poison.into_inner())
                            .remove(&FdKey::from(fd));
                        fds.remove(&FdKey::from(fd));
                    }
                    drop(fds);
                    if let Some(waker) = read_waker {
                        waker.wake();
                    }
                    if let Some(waker) = write_waker {
                        waker.wake();
                    }
                    return Err(error);
                }
            };
            let displaced_read_waker = (!replaced_existing || interest.readable)
                .then(|| fd_info.read_waker.take())
                .flatten();
            let displaced_write_waker = (!replaced_existing || interest.writable)
                .then(|| fd_info.write_waker.take())
                .flatten();
            fd_info.interest = if replaced_existing {
                new_interest
            } else {
                interest
            };

            if interest.readable {
                fd_info.read_waker = Some(waker.clone());
            }
            if interest.writable {
                fd_info.write_waker = Some(waker);
            }
            #[cfg(windows)]
            let registration = if let Some(cancellation) = cancellation {
                if !replaced_existing {
                    self.waiter_cancellations
                        .clear_interest(FdKey::from(fd), true, true);
                }
                self.waiter_cancellations.publish(&cancellation);
                Some(cancellation)
            } else {
                self.waiter_cancellations.clear_interest(
                    FdKey::from(fd),
                    !replaced_existing || interest.readable,
                    !replaced_existing || interest.writable,
                );
                None
            };
            drop(fds);
            if replaced_existing {
                drop(displaced_read_waker);
                drop(displaced_write_waker);
            } else {
                if let Some(waker) = displaced_read_waker {
                    waker.wake();
                }
                if let Some(waker) = displaced_write_waker {
                    waker.wake();
                }
            }
            #[cfg(windows)]
            {
                Ok(registration)
            }
            #[cfg(not(windows))]
            Ok(())
        } else {
            // Publish the waker in the same state-lock transaction as the
            // platform registration. The poll thread may observe readiness as
            // soon as registration wakes it, but it cannot consume a
            // temporarily wakerless entry before insertion completes.
            #[cfg(windows)]
            let platform_generation = if let Some(owner) = owner.clone() {
                self.platform_reactor
                    .register_owned_waiter(fd, interest, owner)
            } else {
                self.platform_reactor.register_waiter(fd, interest)
            }
            .map_err(PlatformUpdateFailure::into_error)?;
            #[cfg(not(windows))]
            self.platform_reactor.register_fd(fd, interest)?;
            let mut fd_info = FdInfo {
                interest,
                registered_at: Instant::now(),
                event_count: 0,
                read_waker: None,
                write_waker: None,
            };
            if interest.readable {
                fd_info.read_waker = Some(waker.clone());
            }
            if interest.writable {
                fd_info.write_waker = Some(waker);
            }

            let key = FdKey::from(fd);
            #[cfg(windows)]
            self.platform_generations
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .insert(key, platform_generation);
            fds.insert(key, fd_info);
            let current_count = fds.len() as u64;
            self.metrics
                .peak_fd_count
                .fetch_max(current_count, Ordering::Relaxed);
            #[cfg(windows)]
            {
                if let Some(cancellation) = cancellation {
                    self.waiter_cancellations.publish(&cancellation);
                    Ok(Some(cancellation))
                } else {
                    self.waiter_cancellations.clear_interest(
                        key,
                        interest.readable,
                        interest.writable,
                    );
                    Ok(None)
                }
            }
            #[cfg(not(windows))]
            Ok(())
        }
    }

    /// Remove wakers for a file descriptor.
    pub fn deregister_waker(&self, fd: RawFd, interest: Interest) {
        let mut read_waker = None;
        let mut write_waker = None;
        if let Ok(mut fds) = self.registered_fds.lock()
            && let Some(fd_info) = fds.get_mut(&FdKey::from(fd))
        {
            if interest.readable {
                read_waker = fd_info.read_waker.take();
            }
            if interest.writable {
                write_waker = fd_info.write_waker.take();
            }
            #[cfg(windows)]
            self.waiter_cancellations.clear_interest(
                FdKey::from(fd),
                interest.readable,
                interest.writable,
            );
        }
        drop(read_waker);
        drop(write_waker);
    }
}
