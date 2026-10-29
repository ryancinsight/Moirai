#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

mod admission_ownership;
mod cancel;
mod construction;
mod fixtures;
mod future_destruction;
mod priority;
mod shutdown;
mod shutdown_ownership;
mod spawn;
mod wait;
