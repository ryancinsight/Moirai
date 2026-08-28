//! Central fd-readiness reactor over the platform backends.
//!
//! The live path is fd readiness: I/O futures register wakers for a file
//! descriptor and interest, the platform backend (epoll/kqueue/`WSAPoll`)
//! reports readiness, and [`IoReactor`] wakes the registered executor wakers.

/// The [`IoReactor`] driving waker registration and event dispatch.
pub mod core;
#[cfg(any(
    test,
    target_os = "macos",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd"
))]
pub(crate) mod kqueue_transition;
/// Reactor performance counters.
pub mod metrics;
pub(crate) mod registration;
/// Thread-local / process-global active-reactor installation.
pub mod tls;

/// Reactor unit tests.
#[cfg(test)]
pub mod tests;

pub use self::core::IoReactor;
pub use self::metrics::ReactorMetrics;
