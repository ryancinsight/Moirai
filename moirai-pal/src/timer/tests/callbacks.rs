use std::{
    mem::ManuallyDrop,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use super::super::TimerState;

#[derive(Debug, Eq, PartialEq)]
pub(super) struct CallbackCounts {
    pub(super) clones: usize,
    pub(super) wakes: usize,
    pub(super) drops: usize,
}

enum CallbackKind {
    Clone,
    Wake,
    Drop,
}

pub(super) struct CallbackProbe {
    state: Arc<TimerState>,
    clones: AtomicUsize,
    wakes: AtomicUsize,
    drops: AtomicUsize,
    panic_on_clone: AtomicBool,
    panic_on_wake: AtomicBool,
    panic_on_drop: AtomicBool,
}

impl CallbackProbe {
    pub(super) fn new(state: Arc<TimerState>) -> Self {
        Self {
            state,
            clones: AtomicUsize::new(0),
            wakes: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
            panic_on_clone: AtomicBool::new(false),
            panic_on_wake: AtomicBool::new(false),
            panic_on_drop: AtomicBool::new(false),
        }
    }

    fn record(&self, kind: CallbackKind) {
        if std::thread::panicking() {
            return;
        }
        let registry = self
            .state
            .waker
            .try_lock()
            .expect("waker callback must run outside the registry lock");
        drop(registry);
        let (count, panics, name) = match kind {
            CallbackKind::Clone => (&self.clones, &self.panic_on_clone, "clone"),
            CallbackKind::Wake => (&self.wakes, &self.panic_on_wake, "wake"),
            CallbackKind::Drop => (&self.drops, &self.panic_on_drop, "drop"),
        };
        count.fetch_add(1, Ordering::Relaxed);
        assert!(
            !panics.load(Ordering::Relaxed),
            "injected {name} callback panic"
        );
    }

    pub(super) fn counts(&self) -> CallbackCounts {
        CallbackCounts {
            clones: self.clones.load(Ordering::Relaxed),
            wakes: self.wakes.load(Ordering::Relaxed),
            drops: self.drops.load(Ordering::Relaxed),
        }
    }
}

static CALLBACK_WAKER: std::task::RawWakerVTable = std::task::RawWakerVTable::new(
    clone_callback,
    wake_callback,
    wake_by_ref_callback,
    drop_callback,
);

unsafe fn clone_callback(data: *const ()) -> std::task::RawWaker {
    // SAFETY: `data` owns one `Arc<CallbackProbe>` under `CALLBACK_WAKER`.
    let probe = ManuallyDrop::new(unsafe { Arc::from_raw(data.cast::<CallbackProbe>()) });
    probe.record(CallbackKind::Clone);
    let clone = Arc::clone(&probe);
    std::task::RawWaker::new(Arc::into_raw(clone).cast(), &CALLBACK_WAKER)
}

unsafe fn wake_callback(data: *const ()) {
    // SAFETY: wake consumes the `Arc<CallbackProbe>` owned by this raw waker.
    let probe = unsafe { Arc::from_raw(data.cast::<CallbackProbe>()) };
    probe.record(CallbackKind::Wake);
}

unsafe fn wake_by_ref_callback(data: *const ()) {
    // SAFETY: wake-by-reference borrows the raw waker's owned Arc.
    let probe = ManuallyDrop::new(unsafe { Arc::from_raw(data.cast::<CallbackProbe>()) });
    probe.record(CallbackKind::Wake);
}

unsafe fn drop_callback(data: *const ()) {
    // SAFETY: drop consumes the `Arc<CallbackProbe>` owned by this raw waker.
    let probe = unsafe { Arc::from_raw(data.cast::<CallbackProbe>()) };
    probe.record(CallbackKind::Drop);
}

pub(super) fn callback_waker(probe: &Arc<CallbackProbe>) -> std::task::Waker {
    let raw = std::task::RawWaker::new(Arc::into_raw(Arc::clone(probe)).cast(), &CALLBACK_WAKER);
    // SAFETY: `CALLBACK_WAKER` preserves the Arc ownership represented by `raw`.
    unsafe { std::task::Waker::from_raw(raw) }
}

pub(super) fn registered_waker_count(state: &TimerState) -> usize {
    usize::from(
        state
            .waker
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_some(),
    )
}

pub(super) fn registered_waker_matches(state: &TimerState, expected: &std::task::Waker) -> bool {
    state
        .waker
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .as_ref()
        .is_some_and(|stored| stored.will_wake(expected))
}

#[cfg(not(loom))]
fn expect_callback_panic(callback: impl FnOnce()) {
    let Err(_) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(callback)) else {
        panic!("injected callback must panic");
    };
}

#[test]
#[cfg(not(loom))]
fn pal_timer_waker_callbacks_run_outside_the_registry_lock() {
    let state = Arc::new(TimerState::new());
    let retained_probe = Arc::new(CallbackProbe::new(Arc::clone(&state)));
    let retained = callback_waker(&retained_probe);

    state.register_waker(&retained);
    state.register_waker(&retained);
    assert_eq!(
        retained_probe.counts(),
        CallbackCounts {
            clones: 2,
            wakes: 0,
            drops: 1,
        }
    );
    assert_eq!(registered_waker_count(&state), 1);
    assert!(registered_waker_matches(&state, &retained));

    let rejected_probe = Arc::new(CallbackProbe::new(Arc::clone(&state)));
    let rejected = callback_waker(&rejected_probe);
    rejected_probe.panic_on_clone.store(true, Ordering::Relaxed);
    expect_callback_panic(|| state.register_waker(&rejected));
    rejected_probe
        .panic_on_clone
        .store(false, Ordering::Relaxed);
    assert!(!state.waker.is_poisoned());
    assert_eq!(registered_waker_count(&state), 1);
    assert!(registered_waker_matches(&state, &retained));
    assert!(!registered_waker_matches(&state, &rejected));
    assert_eq!(
        rejected_probe.counts(),
        CallbackCounts {
            clones: 1,
            wakes: 0,
            drops: 0,
        }
    );

    let replacement_probe = Arc::new(CallbackProbe::new(Arc::clone(&state)));
    let replacement = callback_waker(&replacement_probe);
    retained_probe.panic_on_drop.store(true, Ordering::Relaxed);
    expect_callback_panic(|| state.register_waker(&replacement));
    retained_probe.panic_on_drop.store(false, Ordering::Relaxed);
    assert!(!state.waker.is_poisoned());
    assert_eq!(registered_waker_count(&state), 1);
    assert!(registered_waker_matches(&state, &replacement));
    assert!(!registered_waker_matches(&state, &retained));
    assert_eq!(
        retained_probe.counts(),
        CallbackCounts {
            clones: 2,
            wakes: 0,
            drops: 2,
        }
    );
    assert_eq!(
        replacement_probe.counts(),
        CallbackCounts {
            clones: 1,
            wakes: 0,
            drops: 0,
        }
    );

    state.complete();
    assert!(state.is_completed());
    assert_eq!(registered_waker_count(&state), 0);
    assert_eq!(
        replacement_probe.counts(),
        CallbackCounts {
            clones: 1,
            wakes: 1,
            drops: 0,
        }
    );

    drop(retained);
    drop(rejected);
    drop(replacement);
    assert_eq!(
        retained_probe.counts(),
        CallbackCounts {
            clones: 2,
            wakes: 0,
            drops: 3,
        }
    );
    assert_eq!(
        rejected_probe.counts(),
        CallbackCounts {
            clones: 1,
            wakes: 0,
            drops: 1,
        }
    );
    assert_eq!(
        replacement_probe.counts(),
        CallbackCounts {
            clones: 1,
            wakes: 1,
            drops: 1,
        }
    );

    let wake_state = Arc::new(TimerState::new());
    let wake_probe = Arc::new(CallbackProbe::new(Arc::clone(&wake_state)));
    let wake = callback_waker(&wake_probe);
    wake_state.register_waker(&wake);
    wake_probe.panic_on_wake.store(true, Ordering::Relaxed);
    expect_callback_panic(|| wake_state.complete());
    wake_probe.panic_on_wake.store(false, Ordering::Relaxed);
    assert!(!wake_state.waker.is_poisoned());
    assert!(wake_state.is_completed());
    assert_eq!(registered_waker_count(&wake_state), 0);
    assert_eq!(
        wake_probe.counts(),
        CallbackCounts {
            clones: 1,
            wakes: 1,
            drops: 0,
        }
    );

    drop(wake);
    let completed_counts = wake_probe.counts();
    assert_eq!(
        completed_counts,
        CallbackCounts {
            clones: 1,
            wakes: 1,
            drops: 1,
        }
    );
    wake_state.complete();
    assert!(wake_state.is_completed());
    assert_eq!(registered_waker_count(&wake_state), 0);
    assert_eq!(wake_probe.counts(), completed_counts);
}
