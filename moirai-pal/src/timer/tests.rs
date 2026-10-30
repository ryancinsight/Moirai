use std::{
    future::Future,
    mem::ManuallyDrop,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use futures::task::{ArcWake, waker};

use super::{TimerState, sleep};

#[derive(Debug, PartialEq, Eq)]
struct CallbackCounts {
    clones: usize,
    wakes: usize,
    drops: usize,
}

enum CallbackKind {
    Clone,
    Wake,
    Drop,
}

struct CallbackProbe {
    state: Arc<TimerState>,
    clones: AtomicUsize,
    wakes: AtomicUsize,
    drops: AtomicUsize,
    panic_on_clone: AtomicBool,
    panic_on_wake: AtomicBool,
    panic_on_drop: AtomicBool,
}

impl CallbackProbe {
    fn new(state: Arc<TimerState>) -> Self {
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

    fn counts(&self) -> CallbackCounts {
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

fn callback_waker(probe: &Arc<CallbackProbe>) -> std::task::Waker {
    let raw = std::task::RawWaker::new(Arc::into_raw(Arc::clone(probe)).cast(), &CALLBACK_WAKER);
    // SAFETY: `CALLBACK_WAKER` preserves the Arc ownership represented by `raw`.
    unsafe { std::task::Waker::from_raw(raw) }
}

fn registered_waker_count(state: &TimerState) -> usize {
    usize::from(
        state
            .waker
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_some(),
    )
}

fn registered_waker_matches(state: &TimerState, expected: &std::task::Waker) -> bool {
    state
        .waker
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .as_ref()
        .is_some_and(|stored| stored.will_wake(expected))
}

fn expect_callback_panic(callback: impl FnOnce()) {
    let Err(_) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(callback)) else {
        panic!("injected callback must panic");
    };
}

struct WakeFlag {
    woke: AtomicBool,
    wake_tx: std::sync::mpsc::SyncSender<()>,
}

impl WakeFlag {
    fn new(wake_tx: std::sync::mpsc::SyncSender<()>) -> Self {
        Self {
            woke: AtomicBool::new(false),
            wake_tx,
        }
    }
}

impl ArcWake for WakeFlag {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.woke.store(true, Ordering::Release);
        if let Err(error) = arc_self.wake_tx.try_send(()) {
            assert!(
                matches!(error, std::sync::mpsc::TrySendError::Full(())),
                "timer wake observer must remain connected"
            );
        }
    }
}

#[test]
fn pal_timer_is_pending_before_deadline_and_wakes() {
    let (wake_tx, wake_rx) = std::sync::mpsc::sync_channel(1);
    let wake_flag = Arc::new(WakeFlag::new(wake_tx));
    let waker = waker(Arc::clone(&wake_flag));
    let mut context = Context::from_waker(&waker);
    let mut timer = Box::pin(sleep(Duration::from_millis(20)));

    assert!(matches!(timer.as_mut().poll(&mut context), Poll::Pending));
    assert!(!wake_flag.woke.load(Ordering::Acquire));

    wake_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("timer must publish its wake event");

    assert!(wake_flag.woke.load(Ordering::Acquire));
    assert!(matches!(
        timer.as_mut().poll(&mut context),
        Poll::Ready(Ok(()))
    ));
}

#[test]
fn pal_timer_zero_duration_completes_immediately() {
    let (wake_tx, _wake_rx) = std::sync::mpsc::sync_channel(1);
    let wake_flag = Arc::new(WakeFlag::new(wake_tx));
    let waker = waker(wake_flag);
    let mut context = Context::from_waker(&waker);
    let mut timer = Box::pin(sleep(Duration::ZERO));

    assert!(matches!(
        timer.as_mut().poll(&mut context),
        Poll::Ready(Ok(()))
    ));
}

#[test]
fn pal_timer_extreme_duration_does_not_panic() {
    // Regression: `Instant::now() + Duration::MAX` panics on overflow. The
    // deadline computation must clamp/`checked_add` instead, yielding a far-future
    // deadline rather than aborting.
    let timer = super::Timer::new(Duration::MAX);
    assert!(timer.deadline() > std::time::Instant::now());

    let timer = super::Timer::new(Duration::from_secs(u64::MAX));
    assert!(timer.deadline() > std::time::Instant::now());
}

#[test]
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
    assert!(state.completed.load(Ordering::Acquire));
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
    assert!(wake_state.completed.load(Ordering::Acquire));
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
    assert!(wake_state.completed.load(Ordering::Acquire));
    assert_eq!(registered_waker_count(&wake_state), 0);
    assert_eq!(wake_probe.counts(), completed_counts);
}

#[test]
fn pal_timer_spawn_failure_preserves_error_and_releases_the_waker() {
    let state = Arc::new(TimerState::new());
    let probe = Arc::new(CallbackProbe::new(Arc::clone(&state)));
    let task_waker = callback_waker(&probe);

    state.register_waker(&task_waker);
    drop(task_waker);
    assert_eq!(Arc::strong_count(&probe), 2);

    let error = state
        .finish_spawn(Err(std::io::Error::from_raw_os_error(i32::MAX)))
        .expect_err("injected worker spawn failure must remain an error");

    assert_eq!(error.raw_os_error(), Some(i32::MAX));
    assert!(state.completed.load(Ordering::Acquire));
    assert_eq!(registered_waker_count(&state), 0);
    assert_eq!(Arc::strong_count(&probe), 1);
    assert_eq!(probe.counts().drops, 2);
}
