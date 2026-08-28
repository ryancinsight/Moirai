use std::collections::HashMap;
use std::io;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::task::Waker;
use std::time::{Duration, Instant};

use super::metrics::ReactorMetrics;
#[cfg(windows)]
use crate::windows::poll::PolledEvent;
use crate::{create_reactor, Event, Interest, PlatformReactor, RawFd, Reactor};

/// Send/Sync-safe internal key for platform handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FdKey(pub(crate) usize);

impl From<RawFd> for FdKey {
    fn from(fd: RawFd) -> Self {
        Self(fd as usize)
    }
}

/// Information about registered file descriptors
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for future telemetry/debugging per ADR requirements
pub struct FdInfo {
    /// Registered readiness interest.
    pub interest: Interest,
    /// When the descriptor was registered.
    pub registered_at: Instant,
    /// Number of events observed for this descriptor.
    pub event_count: u64,
    /// Waker armed for read readiness.
    pub read_waker: Option<Waker>,
    /// Waker armed for write readiness.
    pub write_waker: Option<Waker>,
}

/// Central async I/O reactor managing all platform-specific operations.
pub struct IoReactor {
    /// Platform-specific reactor implementation
    pub(crate) platform_reactor: PlatformReactor,
    /// Event loop control
    pub(crate) running: Arc<AtomicBool>,
    /// Registered file descriptor tracking
    pub(crate) registered_fds: Arc<Mutex<HashMap<FdKey, FdInfo>>>,
    /// Performance metrics
    pub(crate) metrics: Arc<ReactorMetrics>,
}

impl IoReactor {
    /// Create a new I/O reactor with platform-optimal implementation.
    pub fn new() -> io::Result<Self> {
        let platform_reactor = create_reactor()?;

        Ok(Self {
            platform_reactor,
            running: Arc::new(AtomicBool::new(false)),
            registered_fds: Arc::new(Mutex::new(HashMap::new())),
            metrics: Arc::new(ReactorMetrics::default()),
        })
    }

    /// Register a file descriptor for async I/O operations.
    pub fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        let mut fds = self
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        // Hold central state across platform publication so readiness cannot be
        // dispatched before its matching central registration exists.
        self.platform_reactor.register_fd(fd, interest)?;

        fds.insert(
            FdKey::from(fd),
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

        Ok(())
    }

    /// Unregister a file descriptor.
    pub fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        let mut fds = self
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        self.platform_reactor.unregister_fd(fd)?;
        fds.remove(&FdKey::from(fd));
        Ok(())
    }

    /// Run the event loop until stopped.
    pub fn run(&self) -> io::Result<()> {
        // Relaxed: `running` is a single-location loop-control flag. It does
        // not publish reactor state; `stop` separately wakes the platform poll
        // so this loop observes the flag at its next iteration boundary.
        self.running.store(true, Ordering::Relaxed);
        self.metrics
            .start_time
            .set(Instant::now())
            .map_err(|_| io::Error::other("Reactor already started"))?;

        while self.running.load(Ordering::Relaxed) {
            self.run_iteration(Some(Duration::from_millis(10)))?;
        }

        Ok(())
    }

    /// Run a single iteration of the event loop.
    pub fn run_iteration(&self, timeout: Option<Duration>) -> io::Result<()> {
        let iteration_start = Instant::now();

        #[cfg(windows)]
        {
            let events = self.platform_reactor.poll_registered_events(timeout)?;
            for event in events {
                self.handle_polled_event(event)?;
            }
        }

        #[cfg(not(windows))]
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

    /// Handle a single I/O event.
    #[cfg(not(windows))]
    fn handle_event(&self, event: Event) -> io::Result<()> {
        // Update metrics
        self.metrics
            .events_processed
            .fetch_add(1, Ordering::Relaxed);

        // Consume matching one-shot interests before waking their tasks. A
        // task that still observes WouldBlock re-arms its interest on re-poll.
        self.wake_fd_waiters(event)
    }

    /// Handle readiness paired with its Windows registration generation.
    #[cfg(windows)]
    pub(super) fn handle_polled_event(&self, event: PolledEvent) -> io::Result<()> {
        self.metrics
            .events_processed
            .fetch_add(1, Ordering::Relaxed);

        let readiness = event.event().clone();
        self.wake_fd_waiters_if_current(readiness, |platform| {
            platform.is_current_polled_event(&event)
        })
    }

    /// Wake tasks waiting on a specific file descriptor event.
    #[cfg(any(not(windows), test))]
    pub(super) fn wake_fd_waiters(&self, event: Event) -> io::Result<()> {
        self.wake_fd_waiters_if_current(event, |_| true)
    }

    fn wake_fd_waiters_if_current(
        &self,
        event: Event,
        is_current: impl FnOnce(&PlatformReactor) -> bool,
    ) -> io::Result<()> {
        self.wake_fd_waiters_with_platform(event, is_current, |platform, fd, remaining| {
            platform.unregister_fd(fd).and_then(|()| {
                if remaining.readable || remaining.writable {
                    platform.register_fd(fd, remaining)
                } else {
                    Ok(())
                }
            })
        })
    }

    pub(super) fn wake_fd_waiters_with_platform(
        &self,
        event: Event,
        is_current: impl FnOnce(&PlatformReactor) -> bool,
        update_platform: impl FnOnce(&PlatformReactor, RawFd, Interest) -> io::Result<()>,
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
            fd_info.read_waker.take()
        } else {
            None
        };
        let write_waker = if consume_write {
            fd_info.write_waker.take()
        } else {
            None
        };
        let (stranded_read_waker, stranded_write_waker) = if platform_result.is_err() {
            (fd_info.read_waker.take(), fd_info.write_waker.take())
        } else {
            (None, None)
        };
        if platform_result.is_ok() && (remaining.readable || remaining.writable) {
            fd_info.interest = remaining;
        } else {
            fds.remove(&key);
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
        platform_result
    }

    /// Register a task's waker for a file descriptor and interest.
    pub fn register_waker(&self, fd: RawFd, interest: Interest, waker: Waker) -> io::Result<()> {
        let mut fds = self
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(fd_info) = fds.get_mut(&FdKey::from(fd)) {
            let mut new_interest = fd_info.interest;
            if interest.readable {
                new_interest.readable = true;
            }
            if interest.writable {
                new_interest.writable = true;
            }
            // Always re-register with the platform reactor: the socket may have
            // been pruned from the platform-level interest map via POLLNVAL (when
            // the previous socket with the same FD number was closed), while the
            // `registered_fds` entry was left behind. Blindly trusting the cached
            // interest and skipping re-registration leaves the new socket invisible
            // to `poll_events`, so its readiness is never signalled and reads/writes
            // block until the debug timeout fires.
            let _ = self.platform_reactor.unregister_fd(fd); // best-effort; may already be absent
            self.platform_reactor.register_fd(fd, new_interest)?;
            fd_info.interest = new_interest;

            if interest.readable {
                fd_info.read_waker = Some(waker.clone());
            }
            if interest.writable {
                fd_info.write_waker = Some(waker);
            }
            Ok(())
        } else {
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

            // Publish the waker in the same state-lock transaction as the
            // platform registration. The poll thread may observe readiness as
            // soon as `register_fd` wakes it, but it cannot consume a
            // temporarily wakerless entry before this insertion completes.
            self.platform_reactor.register_fd(fd, interest)?;
            fds.insert(FdKey::from(fd), fd_info);
            let current_count = fds.len() as u64;
            self.metrics
                .peak_fd_count
                .fetch_max(current_count, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Remove wakers for a file descriptor.
    pub fn deregister_waker(&self, fd: RawFd, interest: Interest) {
        if let Ok(mut fds) = self.registered_fds.lock() {
            if let Some(fd_info) = fds.get_mut(&FdKey::from(fd)) {
                if interest.readable {
                    fd_info.read_waker = None;
                }
                if interest.writable {
                    fd_info.write_waker = None;
                }
            }
        }
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
