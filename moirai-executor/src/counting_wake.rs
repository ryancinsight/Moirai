//! Wake target shared by the executor's tests: counts how often it is woken.

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Wake, Waker},
};

/// Wake target that counts how many times it is woken.
#[derive(Default)]
pub(crate) struct CountingWake(pub(crate) AtomicUsize);

impl CountingWake {
    /// Number of wakes observed so far.
    pub(crate) fn wakes(&self) -> usize {
        self.0.load(Ordering::Acquire)
    }
}

impl Wake for CountingWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }
}

/// A fresh wake target and a waker that wakes it.
pub(crate) fn counting_waker() -> (Arc<CountingWake>, Waker) {
    let target = Arc::new(CountingWake::default());
    let waker = Waker::from(Arc::clone(&target));
    (target, waker)
}
