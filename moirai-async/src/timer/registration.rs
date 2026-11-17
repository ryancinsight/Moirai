use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
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

pub(super) enum WakerSlot {
    Registered,
    Consumed,
}

impl TimerRegistration {
    pub(super) fn new(waker: Waker) -> Arc<Self> {
        Arc::new(Self {
            waker: Mutex::new(Some(waker)),
            cancelled: AtomicBool::new(false),
            in_heap: AtomicBool::new(false),
        })
    }

    fn lock_waker(&self) -> MutexGuard<'_, Option<Waker>> {
        // No user code runs while this guard is held, so a panic cannot leave
        // a partially updated registration. Recover the completed state.
        self.waker.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(super) fn replace_waker(&self, waker: &Waker) -> WakerSlot {
        self.replace_waker_after_observation(waker, || {})
    }

    fn replace_waker_after_observation(
        &self,
        waker: &Waker,
        after_observation: impl FnOnce(),
    ) -> WakerSlot {
        {
            let stored = self.lock_waker();
            match stored.as_ref() {
                None => return WakerSlot::Consumed,
                Some(current) if current.will_wake(waker) => return WakerSlot::Registered,
                Some(_) => {}
            }
        }

        after_observation();

        // RawWaker clone and drop callbacks are arbitrary user code. Clone
        // before taking the mutex, then move the replaced waker out so neither
        // callback can re-enter while the registration mutex is held.
        let mut replacement = Some(waker.clone());
        let (previous, slot) = {
            let mut stored = self.lock_waker();
            match stored.as_ref() {
                None => (None, WakerSlot::Consumed),
                Some(current) if current.will_wake(waker) => (None, WakerSlot::Registered),
                Some(_) => (
                    stored.replace(
                        replacement
                            .take()
                            .expect("invariant: replacement waker remains available"),
                    ),
                    WakerSlot::Registered,
                ),
            }
        };
        drop(previous);
        drop(replacement);
        slot
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

    pub(super) fn take_waker(&self) -> Option<Waker> {
        let mut guard = self.lock_waker();
        let waker = guard.take();
        drop(guard);
        waker
    }

    pub(super) fn wake(&self) {
        // `take_waker` releases the guard before this arbitrary call.
        let waker = self.take_waker();
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TimerRegistration, WakerSlot};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier, Weak, mpsc::sync_channel};
    use std::task::{Wake, Waker};

    struct ReenteringDrop {
        registration: Weak<TimerRegistration>,
        completed: Arc<AtomicBool>,
    }

    impl Wake for ReenteringDrop {
        fn wake(self: Arc<Self>) {
            panic!("replacement test waker must not run");
        }
    }

    impl Drop for ReenteringDrop {
        fn drop(&mut self) {
            if let Some(registration) = self.registration.upgrade() {
                drop(registration.take_waker());
            }
            self.completed.store(true, Ordering::Release);
        }
    }

    #[test]
    fn replaced_waker_drops_after_registration_unlocks() {
        let registration = TimerRegistration::new(Waker::noop().clone());
        let completed = Arc::new(AtomicBool::new(false));
        let reentering = Waker::from(Arc::new(ReenteringDrop {
            registration: Arc::downgrade(&registration),
            completed: Arc::clone(&completed),
        }));
        let slot = registration.replace_waker(&reentering);
        assert!(matches!(slot, WakerSlot::Registered));
        drop(reentering);

        let slot = registration.replace_waker(Waker::noop());
        assert!(matches!(slot, WakerSlot::Registered));

        assert!(completed.load(Ordering::Acquire));
        assert!(registration.take_waker().is_none());
    }

    #[test]
    fn consumed_during_replacement_is_not_repopulated() {
        let registration = Arc::new(TimerRegistration::new(Waker::noop().clone()));
        let replacement = Waker::from(Arc::new(ReenteringDrop {
            registration: Weak::new(),
            completed: Arc::new(AtomicBool::new(false)),
        }));
        let (observed_tx, observed_rx) = sync_channel(0);
        let release = Arc::new(Barrier::new(2));
        let worker_registration = Arc::clone(&registration);
        let worker_release = Arc::clone(&release);
        let worker = std::thread::spawn(move || {
            worker_registration.replace_waker_after_observation(&replacement, || {
                observed_tx
                    .send(())
                    .expect("invariant: observation receiver remains active");
                worker_release.wait();
            })
        });

        observed_rx
            .recv()
            .expect("replacement reaches the post-observation window");
        drop(registration.take_waker());
        release.wait();

        let slot = worker.join().expect("replacement worker must not panic");
        assert!(matches!(slot, WakerSlot::Consumed));
        assert!(registration.take_waker().is_none());
    }
}
