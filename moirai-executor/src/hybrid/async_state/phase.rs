//! Phase values of the `AsyncFutureState::state` atomic.

pub(super) const ASYNC_IDLE: u8 = 0;
pub(super) const ASYNC_QUEUED: u8 = 1;
pub(super) const ASYNC_POLLING: u8 = 2;
pub(super) const ASYNC_NOTIFIED: u8 = 3;
pub(super) const ASYNC_COMPLETED: u8 = 4;
