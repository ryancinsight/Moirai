//! Wake target shared by the executor's tests: counts how often it is woken.

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{RawWaker, RawWakerVTable, Wake, Waker},
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

/// The one vtable of [`counting_waker`] wakers, so a clone shares its source's
/// data pointer and vtable address and `Waker::will_wake` matches them exactly.
/// `Waker::from(Arc<impl Wake>)` builds a vtable that codegen or Miri may
/// duplicate, which std allows `will_wake` to report as a mismatch.
static COUNTING_VTABLE: RawWakerVTable = RawWakerVTable::new(
    clone_counting,
    wake_counting,
    wake_counting_by_ref,
    drop_counting,
);

unsafe fn clone_counting(data: *const ()) -> RawWaker {
    // SAFETY: `data` came from `Arc::into_raw` on an `Arc<CountingWake>` that
    // this waker still owns a strong count of.
    unsafe { Arc::increment_strong_count(data.cast::<CountingWake>()) };
    RawWaker::new(data, &COUNTING_VTABLE)
}

unsafe fn wake_counting(data: *const ()) {
    // SAFETY: as in `clone_counting`; `wake` consumes the waker's strong count.
    let target = unsafe { Arc::from_raw(data.cast::<CountingWake>()) };
    target.0.fetch_add(1, Ordering::AcqRel);
}

unsafe fn wake_counting_by_ref(data: *const ()) {
    // SAFETY: as in `clone_counting`; the count stays owned by the waker.
    let target = unsafe { &*data.cast::<CountingWake>() };
    target.0.fetch_add(1, Ordering::AcqRel);
}

unsafe fn drop_counting(data: *const ()) {
    // SAFETY: as in `clone_counting`; `drop` releases the waker's strong count.
    drop(unsafe { Arc::from_raw(data.cast::<CountingWake>()) });
}

/// A fresh wake target and a waker that wakes it. Clones of the waker
/// `will_wake` each other.
pub(crate) fn counting_waker() -> (Arc<CountingWake>, Waker) {
    let target = Arc::new(CountingWake::default());
    let data = Arc::into_raw(Arc::clone(&target)).cast::<()>();
    // SAFETY: `data` is an owned strong count of the target, and the vtable
    // functions uphold the `RawWakerVTable` contract over it.
    let waker = unsafe { Waker::from_raw(RawWaker::new(data, &COUNTING_VTABLE)) };
    (target, waker)
}
