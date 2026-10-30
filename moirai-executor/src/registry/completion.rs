//! Published lifecycle state and its deferred waiter notification.

use std::{task::Waker, time::Duration};

/// Terminal metadata is visible and its state lease is inactive before this
/// receipt invokes arbitrary waiter code.
#[must_use]
pub(crate) struct TaskCompletion {
    elapsed: Duration,
    waker: Option<Waker>,
}

impl TaskCompletion {
    pub(super) const fn new(elapsed: Duration, waker: Option<Waker>) -> Self {
        Self { elapsed, waker }
    }

    pub(crate) fn notify(self) -> Duration {
        if let Some(waker) = self.waker {
            waker.wake();
        }
        self.elapsed
    }
}
