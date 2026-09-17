//! Terminal readiness-driver failure publication.

use std::error::Error;
use std::fmt;
use std::io;
use std::sync::{Arc, Mutex, atomic::Ordering};

use super::core::IoReactor;

#[derive(Debug)]
struct RetainedDriverFailure(Arc<io::Error>);

impl fmt::Display for RetainedDriverFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("readiness driver terminated")
    }
}

impl Error for RetainedDriverFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.0.as_ref())
    }
}

fn retained_driver_error(source: &Arc<io::Error>) -> io::Error {
    io::Error::new(source.kind(), RetainedDriverFailure(Arc::clone(source)))
}

#[derive(Default)]
pub(super) struct DriverFailureState {
    retained: Mutex<Option<Arc<io::Error>>>,
    #[cfg(test)]
    next_iteration: Mutex<Option<io::Error>>,
}

impl DriverFailureState {
    pub(super) fn registration_error(&self) -> Option<io::Error> {
        self.retained
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .as_ref()
            .map(retained_driver_error)
    }

    #[cfg(test)]
    pub(super) fn inject_iteration_failure(&self, error: io::Error) {
        *self
            .next_iteration
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(error);
    }

    #[cfg(test)]
    pub(super) fn take_iteration_failure(&self) -> Option<io::Error> {
        self.next_iteration
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take()
    }
}

impl IoReactor {
    /// Publish a driven event-loop failure and release every stranded waiter.
    pub(super) fn publish_driver_failure(&self, error: io::Error) -> io::Error {
        let mut fds = self
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut driver_failure = self
            .driver_failure
            .retained
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(retained) = driver_failure.as_ref() {
            return retained_driver_error(retained);
        }

        let retained = Arc::new(error);
        *driver_failure = Some(Arc::clone(&retained));
        self.running.store(false, Ordering::Relaxed);
        #[cfg(windows)]
        self.platform_generations
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clear();
        let registrations = std::mem::take(&mut *fds);
        drop(driver_failure);
        drop(fds);

        for mut fd_info in registrations.into_values() {
            if let Some(waker) = fd_info.read_waker.take() {
                waker.wake();
            }
            if let Some(waker) = fd_info.write_waker.take() {
                waker.wake();
            }
        }

        retained_driver_error(&retained)
    }

    #[cfg(test)]
    pub(super) fn inject_driver_failure(&self, error: io::Error) -> io::Error {
        self.publish_driver_failure(error)
    }

    #[cfg(test)]
    pub(super) fn inject_iteration_failure(&self, error: io::Error) {
        self.driver_failure.inject_iteration_failure(error);
    }
}
