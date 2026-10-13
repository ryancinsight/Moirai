#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]
//! Reactor unit tests, grouped by the behavior each area exercises.

#[cfg(any(unix, windows))]
mod harness;

#[cfg(any(unix, windows))]
mod backend_update_failure;
mod lifecycle_and_metrics;
#[cfg(windows)]
mod owned_waiter_cancellation;
#[cfg(any(unix, windows))]
mod readiness_dispatch;
#[cfg(windows)]
mod socket_generation;
#[cfg(any(unix, windows))]
mod terminal_failure;
