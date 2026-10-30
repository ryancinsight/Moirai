#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

mod admission_ownership;
mod cancel;
mod completion_callbacks;
mod completion_cleanup;
mod construction;
mod fixtures;
mod future_destruction;
mod priority;
mod settlement_ownership;
mod shutdown;
mod shutdown_callbacks;
mod shutdown_deadlines;
mod shutdown_ownership;
mod shutdown_reentry;
mod spawn;
mod sync_completion_callbacks;
mod wait;
