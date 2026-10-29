//! Published lifecycle state and its deferred waiter notification.

use std::{task::Waker, time::Duration};

/// Metadata is visible before notifying code that may collect the task result.
#[must_use]
pub(crate) struct TaskCompletion {
    elapsed: Duration,
    wake: Option<Waker>,
}

impl TaskCompletion {
    pub(super) fn new(elapsed: Duration, wake: Option<Waker>) -> Self {
        Self { elapsed, wake }
    }

    pub(crate) fn duration(&self) -> Duration {
        self.elapsed
    }

    pub(crate) fn notify(self) -> Duration {
        if let Some(wake) = self.wake {
            wake.wake();
        }
        self.elapsed
    }
}
