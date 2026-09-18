//! Core reactor types: the platform-handle key, per-descriptor bookkeeping,
//! and the [`IoReactor`] struct itself.

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::task::Waker;
use std::time::Instant;

use super::super::driver_failure::DriverFailureState;
use super::super::metrics::ReactorMetrics;
#[cfg(windows)]
use super::super::registration::RegistrationGeneration;
#[cfg(windows)]
use super::super::waiter_cancellation::WaiterCancellationState;
use crate::{Interest, PlatformReactor, RawFd};

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
    pub(crate) platform_reactor: Arc<PlatformReactor>,
    /// Event loop control
    pub(crate) running: Arc<AtomicBool>,
    /// Registered file descriptor tracking
    pub(crate) registered_fds: Arc<Mutex<HashMap<FdKey, FdInfo>>>,
    /// First terminal failure from a driven event loop.
    pub(in crate::reactor) driver_failure: DriverFailureState,
    /// Windows platform generation paired with each central registration.
    #[cfg(windows)]
    pub(in crate::reactor) platform_generations: Arc<Mutex<HashMap<FdKey, RegistrationGeneration>>>,
    /// Reactor-bound identity for owned Windows waiter cancellation.
    #[cfg(windows)]
    pub(in crate::reactor) waiter_cancellations: Arc<WaiterCancellationState>,
    /// Performance metrics
    pub(crate) metrics: Arc<ReactorMetrics>,
}
