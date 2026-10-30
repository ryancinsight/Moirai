use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use futures::task::{ArcWake, waker};

use super::{TimerState, sleep};

static CALLBACK_PANICS: AtomicBool = AtomicBool::new(false);

static CALLBACK_WAKER: std::task::RawWakerVTable =
    std::task::RawWakerVTable::new(clone_callback, run_callback, run_callback, run_callback);

fn callback(data: *const ()) {
    if std::thread::panicking() {
        return;
    }
    // SAFETY: `callback_waker` ties this pointer to a state that the test keeps
    // alive until every derived waker has been dropped.
    let state = unsafe { &*data.cast::<TimerState>() };
    let registry = state
        .waker
        .try_lock()
        .expect("waker callback must run outside the registry lock");
    drop(registry);
    assert!(
        !CALLBACK_PANICS.load(Ordering::Relaxed),
        "injected callback panic"
    );
}

fn clone_callback(data: *const ()) -> std::task::RawWaker {
    callback(data);
    std::task::RawWaker::new(data, &CALLBACK_WAKER)
}

fn run_callback(data: *const ()) {
    callback(data);
}

fn callback_waker(state: &TimerState) -> std::task::Waker {
    // SAFETY: the test drops this waker and every clone before dropping `state`;
    // the vtable only accesses the state through this pointer.
    unsafe {
        std::task::Waker::from_raw(std::task::RawWaker::new(
            std::ptr::from_ref(state).cast(),
            &CALLBACK_WAKER,
        ))
    }
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
    let state = TimerState::new();
    let faulty = callback_waker(&state);

    state.register_waker(&faulty);
    CALLBACK_PANICS.store(true, Ordering::Relaxed);
    expect_callback_panic(|| state.register_waker(&faulty));
    CALLBACK_PANICS.store(false, Ordering::Relaxed);
    assert!(!state.waker.is_poisoned());

    state.register_waker(std::task::Waker::noop());
    state.register_waker(&faulty);
    CALLBACK_PANICS.store(true, Ordering::Relaxed);
    expect_callback_panic(|| state.register_waker(std::task::Waker::noop()));
    CALLBACK_PANICS.store(false, Ordering::Relaxed);
    assert!(!state.waker.is_poisoned());

    let wake_state = TimerState::new();
    let wake_faulty = callback_waker(&wake_state);
    wake_state.register_waker(&wake_faulty);
    CALLBACK_PANICS.store(true, Ordering::Relaxed);
    expect_callback_panic(|| wake_state.complete());
    CALLBACK_PANICS.store(false, Ordering::Relaxed);
    assert!(!wake_state.waker.is_poisoned());
}
