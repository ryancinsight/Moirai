//! Raw file-descriptor registration and unregistration.

use std::io;
use std::sync::atomic::Ordering;
use std::time::Instant;

#[cfg(windows)]
use super::super::registration::PlatformUpdateFailure;
use super::types::{FdInfo, FdKey, IoReactor};
use crate::{Interest, RawFd, Reactor};

impl IoReactor {
    /// Register a file descriptor for async I/O operations.
    ///
    /// # Errors
    ///
    /// Returns a platform registration error or the retained terminal driver
    /// failure after a driven event loop has stopped on an error.
    ///
    /// On Windows this raw-descriptor API requires the caller to keep the
    /// socket open until it unregisters or consumes the readiness interest.
    /// PAL network sockets use the private owner-aware registration path that
    /// retains snapshot ownership through `WSAPoll`.
    pub fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        let mut fds = self
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(error) = self.driver_failure.registration_error() {
            return Err(error);
        }

        // Hold central state across platform publication so readiness cannot be
        // dispatched before its matching central registration exists.
        #[cfg(windows)]
        let platform_generation = self
            .platform_reactor
            .register_waiter(fd, interest)
            .map_err(PlatformUpdateFailure::into_error)?;
        #[cfg(not(windows))]
        self.platform_reactor.register_fd(fd, interest)?;

        let key = FdKey::from(fd);
        #[cfg(windows)]
        self.platform_generations
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(key, platform_generation);
        #[cfg(windows)]
        self.waiter_cancellations.clear_interest(key, true, true);
        let displaced = fds.insert(
            key,
            FdInfo {
                interest,
                registered_at: Instant::now(),
                event_count: 0,
                read_waker: None,
                write_waker: None,
            },
        );

        // Update peak FD count metric
        let current_count = fds.len() as u64;
        self.metrics
            .peak_fd_count
            .fetch_max(current_count, Ordering::Relaxed);

        drop(fds);
        drop(displaced);
        Ok(())
    }

    /// Unregister a file descriptor.
    pub fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        let mut fds = self
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        self.platform_reactor.unregister_fd(fd)?;
        let key = FdKey::from(fd);
        #[cfg(windows)]
        self.platform_generations
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .remove(&key);
        #[cfg(windows)]
        self.waiter_cancellations.clear_interest(key, true, true);
        let displaced = fds.remove(&key);
        drop(fds);
        drop(displaced);
        Ok(())
    }
}
