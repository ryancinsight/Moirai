//! Browser GPU device-loss observation state.

use std::cell::Cell;
use std::rc::Rc;

pub(crate) struct DeviceLossStatus {
    observed: Rc<Cell<bool>>,
}

pub(crate) struct DeviceLossNotification {
    observed: Rc<Cell<bool>>,
}

impl DeviceLossStatus {
    pub(crate) fn channel() -> (Self, DeviceLossNotification) {
        let observed = Rc::new(Cell::new(false));
        (
            Self {
                observed: Rc::clone(&observed),
            },
            DeviceLossNotification { observed },
        )
    }

    pub(crate) fn is_observed(&self) -> bool {
        self.observed.get()
    }
}

impl DeviceLossNotification {
    pub(crate) fn observe(self) {
        self.observed.set(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notification_marks_its_device_as_lost() {
        let (status, notification) = DeviceLossStatus::channel();

        assert!(!status.is_observed());
        notification.observe();
        assert!(status.is_observed());
    }

    #[test]
    fn prior_device_notification_cannot_mark_replacement_lost() {
        let (prior_status, prior_notification) = DeviceLossStatus::channel();
        let (replacement_status, _replacement_notification) = DeviceLossStatus::channel();

        prior_notification.observe();

        assert!(prior_status.is_observed());
        assert!(!replacement_status.is_observed());
    }
}
