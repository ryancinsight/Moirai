//! Reactor construction, the driven event loop, and exposed metrics.

use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::super::driver_failure::DriverFailureState;
use super::super::metrics::ReactorMetrics;
#[cfg(windows)]
use super::super::waiter_cancellation::WaiterCancellationState;
#[cfg(all(test, windows))]
use super::types::FdKey;
use super::types::IoReactor;
#[cfg(all(test, windows))]
use crate::RawFd;
use crate::{Reactor, create_reactor};

impl IoReactor {
    /// Create a new I/O reactor with platform-optimal implementation.
    pub fn new() -> io::Result<Self> {
        let platform_reactor = create_reactor()?;

        let platform_reactor = Arc::new(platform_reactor);
        let running = Arc::new(AtomicBool::new(false));
        let registered_fds = Arc::new(Mutex::new(HashMap::new()));
        let driver_failure = DriverFailureState::default();
        #[cfg(windows)]
        let platform_generations = Arc::new(Mutex::new(HashMap::new()));
        #[cfg(windows)]
        let waiter_cancellations = WaiterCancellationState::new(
            Arc::clone(&platform_reactor),
            Arc::clone(&running),
            Arc::clone(&registered_fds),
            Arc::clone(&platform_generations),
            driver_failure.clone(),
        );

        Ok(Self {
            platform_reactor,
            running,
            registered_fds,
            driver_failure,
            #[cfg(windows)]
            platform_generations,
            #[cfg(windows)]
            waiter_cancellations,
            metrics: Arc::new(ReactorMetrics::default()),
        })
    }

    #[cfg(all(test, windows))]
    pub(in crate::reactor) fn has_platform_generation(&self, fd: RawFd) -> bool {
        self.platform_generations
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .contains_key(&FdKey::from(fd))
    }

    /// Run the event loop until stopped.
    ///
    /// A platform iteration failure is terminal for this reactor. The first
    /// failure is retained before every registered waiter is woken; later
    /// registrations return an error whose source is that original failure.
    ///
    /// # Errors
    ///
    /// Returns an error when the reactor was already started or a platform
    /// iteration fails.
    pub fn run(&self) -> io::Result<()> {
        // Relaxed: `running` is a single-location loop-control flag. It does
        // not publish reactor state; `stop` separately wakes the platform poll
        // so this loop observes the flag at its next iteration boundary.
        self.metrics
            .start_time
            .set(Instant::now())
            .map_err(|_| io::Error::other("Reactor already started"))?;
        self.running.store(true, Ordering::Relaxed);

        while self.running.load(Ordering::Relaxed) {
            if let Err(error) = self.run_iteration(Some(Duration::from_millis(10))) {
                return Err(self.publish_driver_failure(error));
            }
        }

        Ok(())
    }

    /// Run a single iteration of the event loop.
    pub fn run_iteration(&self, timeout: Option<Duration>) -> io::Result<()> {
        #[cfg(test)]
        if let Some(error) = self.driver_failure.take_iteration_failure() {
            return Err(error);
        }
        let iteration_start = Instant::now();

        #[cfg(any(unix, windows))]
        {
            let events = self.platform_reactor.poll_registered_events(timeout)?;
            for event in events {
                self.handle_polled_event(event)?;
            }
        }

        #[cfg(not(any(unix, windows)))]
        {
            let events = self.platform_reactor.poll_events(timeout)?;
            for event in events {
                self.handle_event(event)?;
            }
        }

        // Update metrics
        let iteration_time = iteration_start.elapsed().as_nanos() as u64;
        self.metrics
            .avg_event_time_ns
            .store(iteration_time, Ordering::Relaxed);

        Ok(())
    }

    /// Stop the event loop.
    pub fn stop(&self) -> io::Result<()> {
        // Relaxed: this store only requests loop termination. The platform
        // wake is the independent progress edge that releases a blocked poll;
        // no data written before this store is consumed through `running`.
        self.running.store(false, Ordering::Relaxed);
        self.platform_reactor.wake()
    }

    /// Wake up the reactor from blocking poll.
    pub fn wake(&self) -> io::Result<()> {
        self.platform_reactor.wake()
    }

    /// Get current reactor metrics.
    pub fn metrics(&self) -> ReactorMetrics {
        ReactorMetrics {
            events_processed: AtomicU64::new(self.metrics.events_processed.load(Ordering::Relaxed)),
            avg_event_time_ns: AtomicU64::new(
                self.metrics.avg_event_time_ns.load(Ordering::Relaxed),
            ),
            peak_fd_count: AtomicU64::new(self.metrics.peak_fd_count.load(Ordering::Relaxed)),
            start_time: std::sync::OnceLock::new(),
        }
    }
}
