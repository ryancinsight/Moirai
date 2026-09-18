#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]
//! Reactor unit tests, grouped by the behavior each area exercises.

mod harness;

mod backend_update_failure;
mod lifecycle_and_metrics;
#[cfg(windows)]
mod owned_waiter_cancellation;
mod readiness_dispatch;
#[cfg(windows)]
mod socket_generation;
mod terminal_failure;
