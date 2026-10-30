//! Published lifecycle state and its deferred waiter notification.

use std::time::Duration;

use super::waiter::WaiterNotifications;

/// Metadata is visible before notifying code that may collect the task result.
#[must_use]
pub(crate) struct TaskCompletion {
    elapsed: Duration,
    notifications: WaiterNotifications,
}

impl TaskCompletion {
    pub(super) fn new(elapsed: Duration, notifications: WaiterNotifications) -> Self {
        Self {
            elapsed,
            notifications,
        }
    }

    pub(crate) fn duration(&self) -> Duration {
        self.elapsed
    }

    pub(crate) fn notify(self) -> Duration {
        self.notifications.notify();
        self.elapsed
    }
}
