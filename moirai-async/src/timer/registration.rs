use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::task::Waker;

pub(super) struct TimerRegistration {
    waker: Mutex<Option<Waker>>,
    cancelled: AtomicBool,
}

impl TimerRegistration {
    pub(super) fn new(waker: Waker) -> Arc<Self> {
        Arc::new(Self {
            waker: Mutex::new(Some(waker)),
            cancelled: AtomicBool::new(false),
        })
    }

    pub(super) fn replace_waker(&self, waker: &Waker) {
        let mut stored = self.waker.lock().unwrap();
        match stored.as_ref() {
            Some(current) if current.will_wake(waker) => {}
            _ => *stored = Some(waker.clone()),
        }
    }

    pub(super) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub(super) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub(super) fn wake(&self) {
        if let Some(waker) = self.waker.lock().unwrap().take() {
            waker.wake();
        }
    }
}
