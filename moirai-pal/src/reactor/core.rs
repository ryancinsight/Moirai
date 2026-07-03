use std::collections::HashMap;
use std::io;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::task::Waker;
use std::time::{Duration, Instant};

use super::metrics::ReactorMetrics;
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
        // Register with platform reactor
        self.platform_reactor.register_fd(fd, interest)?;

        // Track registration
        let mut fds = self.registered_fds.lock().unwrap();
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
        self.platform_reactor.unregister_fd(fd)?;
        self.registered_fds.lock().unwrap().remove(&FdKey::from(fd));
        Ok(())
    }

    /// Run the event loop until stopped.
    pub fn run(&self) -> io::Result<()> {
        self.running.store(true, Ordering::SeqCst);
        self.metrics
            .start_time
            .set(Instant::now())
            .map_err(|_| io::Error::other("Reactor already started"))?;

        while self.running.load(Ordering::SeqCst) {
            self.run_iteration(Some(Duration::from_millis(10)))?;
        }

        Ok(())
    }

    /// Run a single iteration of the event loop.
    pub fn run_iteration(&self, timeout: Option<Duration>) -> io::Result<()> {
        let iteration_start = Instant::now();

        // Poll for I/O events
        let events = self.platform_reactor.poll_events(timeout)?;

        // Process I/O events
        for event in events {
            self.handle_event(event)?;
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
        self.running.store(false, Ordering::SeqCst);
        self.platform_reactor.wake()
    }

    /// Handle a single I/O event.
    fn handle_event(&self, event: Event) -> io::Result<()> {
        // Update FD event count
        if let Ok(mut fds) = self.registered_fds.lock() {
            if let Some(fd_info) = fds.get_mut(&FdKey::from(event.fd)) {
                fd_info.event_count += 1;
            }
        }

        // Update metrics
        self.metrics
            .events_processed
            .fetch_add(1, Ordering::Relaxed);

        // Wake any tasks waiting on this file descriptor
        self.wake_fd_waiters(event);

        Ok(())
    }

    /// Wake tasks waiting on a specific file descriptor event.
    fn wake_fd_waiters(&self, event: Event) {
        let mut read_waker = None;
        let mut write_waker = None;

        if let Ok(mut fds) = self.registered_fds.lock() {
            if let Some(fd_info) = fds.get_mut(&FdKey::from(event.fd)) {
                if event.readable || event.error || event.hangup {
                    read_waker = fd_info.read_waker.take();
                }
                if event.writable || event.error || event.hangup {
                    write_waker = fd_info.write_waker.take();
                }
            }
        }

        if let Some(waker) = read_waker {
            waker.wake();
        }
        if let Some(waker) = write_waker {
            waker.wake();
        }
    }

    /// Register a task's waker for a file descriptor and interest.
    pub fn register_waker(&self, fd: RawFd, interest: Interest, waker: Waker) -> io::Result<()> {
        let mut fds = self.registered_fds.lock().unwrap();
        if let Some(fd_info) = fds.get_mut(&FdKey::from(fd)) {
            let mut new_interest = fd_info.interest;
            let mut modified = false;
            if interest.readable && !new_interest.readable {
                new_interest.readable = true;
                modified = true;
            }
            if interest.writable && !new_interest.writable {
                new_interest.writable = true;
                modified = true;
            }
            if modified {
                self.platform_reactor.unregister_fd(fd)?;
                self.platform_reactor.register_fd(fd, new_interest)?;
                fd_info.interest = new_interest;
            }

            if interest.readable {
                fd_info.read_waker = Some(waker.clone());
            }
            if interest.writable {
                fd_info.write_waker = Some(waker);
            }
            Ok(())
        } else {
            drop(fds);
            self.register_fd(fd, interest)?;
            let mut fds = self.registered_fds.lock().unwrap();
            let fd_info = fds.get_mut(&FdKey::from(fd)).unwrap();
            if interest.readable {
                fd_info.read_waker = Some(waker.clone());
            }
            if interest.writable {
                fd_info.write_waker = Some(waker);
            }
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
