use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::task::Waker;

pub(super) struct TimerRegistration {
    waker: Mutex<Option<Waker>>,
    cancelled: AtomicBool,
    /// `true` while a heap entry for this registration is resident in the
    /// driver's timer heap. Written only while the driver's state mutex is
    /// held (`schedule` sets it, entry removal clears it), which serializes
    /// all accesses; the field is atomic only because the registration lives
    /// in a shared `Arc`, so `Relaxed` ordering suffices.
    in_heap: AtomicBool,
}

impl TimerRegistration {
    pub(super) fn new(waker: Waker) -> Arc<Self> {
        Arc::new(Self {
            waker: Mutex::new(Some(waker)),
            cancelled: AtomicBool::new(false),
            in_heap: AtomicBool::new(false),
        })
    }

    pub(super) fn replace_waker(&self, waker: &Waker) {
        let mut stored = self.waker.lock().unwrap();
        match stored.as_ref() {
            Some(current) if current.will_wake(waker) => {}
            _ => *stored = Some(waker.clone()),
        }
    }

    /// Mark the registration cancelled. Returns `true` if this call performed
    /// the cancellation (i.e. it was not already cancelled), so the driver
    /// counts each dead heap entry exactly once.
    pub(super) fn cancel(&self) -> bool {
        !self.cancelled.swap(true, Ordering::AcqRel)
    }

    pub(super) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// See `in_heap` field docs: called only under the driver state mutex.
    pub(super) fn mark_in_heap(&self) {
        self.in_heap.store(true, Ordering::Relaxed);
    }

    /// See `in_heap` field docs: called only under the driver state mutex.
    pub(super) fn clear_in_heap(&self) {
        self.in_heap.store(false, Ordering::Relaxed);
    }

    /// See `in_heap` field docs: read only under the driver state mutex.
    pub(super) fn is_in_heap(&self) -> bool {
        self.in_heap.load(Ordering::Relaxed)
    }

    pub(super) fn wake(&self) {
        if let Some(waker) = self.waker.lock().unwrap().take() {
            waker.wake();
        }
    }
}
