use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

use futures::task::{ArcWake, waker};

use super::{SleeperEvent, TimerState, sleep};

const SLEEPER_DEADLINE: Duration = Duration::from_secs(60);
const EVENT_TIMEOUT: Duration = Duration::from_secs(5);

fn event_deadline() -> Instant {
    Instant::now()
        .checked_add(EVENT_TIMEOUT)
        .expect("invariant: five seconds fits in the platform instant range")
}

fn receive_event_before(
    receiver: &std::sync::mpsc::Receiver<SleeperEvent>,
    deadline: Instant,
) -> SleeperEvent {
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .expect("sleeper event must arrive before the test deadline");
    receiver
        .recv_timeout(remaining)
        .expect("sleeper must publish the expected event")
}

fn receive_until(
    receiver: &std::sync::mpsc::Receiver<SleeperEvent>,
    expected: SleeperEvent,
    deadline: Instant,
) {
    loop {
        let event = receive_event_before(receiver, deadline);
        if event == expected {
            return;
        }
        assert_eq!(
            event,
            SleeperEvent::Waiting,
            "only a genuine spurious wake may precede the expected event"
        );
    }
}

fn observe_sleeper(state: &TimerState) -> std::sync::mpsc::Receiver<SleeperEvent> {
    let (sender, receiver) = std::sync::mpsc::sync_channel(8);
    state
        .observe_sleeper_with(sender)
        .expect("invariant: each timer test installs one sleeper observer");
    receiver
}

fn wait_until_parked<'state>(
    state: &'state TimerState,
    receiver: &std::sync::mpsc::Receiver<SleeperEvent>,
    deadline: Instant,
) -> std::sync::MutexGuard<'state, ()> {
    receive_until(receiver, SleeperEvent::Waiting, deadline);
    state
        .sleeper_wait_lock
        .lock()
        .unwrap_or_else(|error| error.into_inner())
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

fn tracked_waker() -> (
    std::sync::Weak<WakeFlag>,
    std::task::Waker,
    std::sync::mpsc::Receiver<()>,
) {
    let (wake_tx, wake_rx) = std::sync::mpsc::sync_channel(1);
    let flag = Arc::new(WakeFlag::new(wake_tx));
    let observer = Arc::downgrade(&flag);
    (observer, waker(flag), wake_rx)
}

static CLONE_PANICS: AtomicBool = AtomicBool::new(false);
static DROP_PANICS: AtomicBool = AtomicBool::new(false);
static WAKE_PANICS: AtomicBool = AtomicBool::new(false);

static PANICKING_WAKER: std::task::RawWakerVTable = std::task::RawWakerVTable::new(
    clone_or_panic,
    wake_or_panic,
    wake_by_ref_or_panic,
    drop_or_panic,
);

fn clone_or_panic(data: *const ()) -> std::task::RawWaker {
    assert!(
        !CLONE_PANICS.load(Ordering::Relaxed),
        "injected clone panic"
    );
    std::task::RawWaker::new(data, &PANICKING_WAKER)
}

fn wake_or_panic(_: *const ()) {
    assert!(!WAKE_PANICS.load(Ordering::Relaxed), "injected wake panic");
}

fn wake_by_ref_or_panic(data: *const ()) {
    wake_or_panic(data);
}

fn drop_or_panic(_: *const ()) {
    assert!(!DROP_PANICS.load(Ordering::Relaxed), "injected drop panic");
}

fn panicking_waker() -> std::task::Waker {
    // SAFETY: every vtable function ignores the data pointer, and clone returns
    // the same stateless vtable, so the null pointer is never dereferenced.
    unsafe {
        std::task::Waker::from_raw(std::task::RawWaker::new(std::ptr::null(), &PANICKING_WAKER))
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
fn pal_timer_cancellation_before_wait_exits_without_parking() {
    let timer = sleep(SLEEPER_DEADLINE);
    let deadline = timer.deadline;
    let state = Arc::clone(&timer.state);
    let events = observe_sleeper(&state);
    let (observer, task_waker, _wake_events) = tracked_waker();

    state.register_waker(&task_waker);
    drop(task_waker);
    assert_eq!(observer.strong_count(), 1);
    drop(timer);
    assert_eq!(observer.strong_count(), 0);

    state
        .spawn_sleeper(deadline)
        .expect("real sleeper must spawn after pre-cancellation");
    assert_eq!(
        receive_event_before(&events, event_deadline()),
        SleeperEvent::Exited
    );
}

#[test]
fn pal_timer_drop_during_wait_exits_and_releases_waker() {
    let (observer, task_waker, _wake_events) = tracked_waker();
    let mut timer = Box::pin(sleep(SLEEPER_DEADLINE));
    let state = Arc::clone(&timer.state);
    let events = observe_sleeper(&state);
    let deadline = event_deadline();

    {
        let mut context = Context::from_waker(&task_waker);
        assert!(matches!(timer.as_mut().poll(&mut context), Poll::Pending));
    }
    let wait_guard = wait_until_parked(&state, &events, deadline);
    drop(wait_guard);
    drop(task_waker);
    assert_eq!(observer.strong_count(), 1);

    drop(timer);
    receive_until(&events, SleeperEvent::Exited, deadline);
    assert_eq!(observer.strong_count(), 0);
}

#[test]
fn pal_timer_spurious_notification_reparks_the_sleeper() {
    let mut timer = Box::pin(sleep(SLEEPER_DEADLINE));
    let state = Arc::clone(&timer.state);
    let events = observe_sleeper(&state);
    let mut context = Context::from_waker(std::task::Waker::noop());
    let deadline = event_deadline();

    assert!(matches!(timer.as_mut().poll(&mut context), Poll::Pending));
    let wait_guard = wait_until_parked(&state, &events, deadline);
    state.sleeper_wait.notify_one();
    drop(wait_guard);

    let reparked_guard = wait_until_parked(&state, &events, deadline);
    assert!(!state.completed.load(Ordering::Acquire));
    drop(reparked_guard);

    drop(timer);
    receive_until(&events, SleeperEvent::Exited, deadline);
}

#[test]
fn pal_timer_borrowed_ready_exits_and_releases_waker() {
    let (observer, task_waker, _wake_events) = tracked_waker();
    let mut timer = Box::pin(sleep(SLEEPER_DEADLINE));
    let state = Arc::clone(&timer.state);
    let events = observe_sleeper(&state);
    let deadline = event_deadline();

    {
        let mut context = Context::from_waker(&task_waker);
        assert!(matches!(timer.as_mut().poll(&mut context), Poll::Pending));
    }
    let wait_guard = wait_until_parked(&state, &events, deadline);
    drop(wait_guard);
    drop(task_waker);
    assert_eq!(observer.strong_count(), 1);

    timer.as_mut().get_mut().deadline = Instant::now();
    let mut context = Context::from_waker(std::task::Waker::noop());
    assert!(matches!(
        timer.as_mut().poll(&mut context),
        Poll::Ready(Ok(()))
    ));
    receive_until(&events, SleeperEvent::Exited, deadline);
    assert_eq!(observer.strong_count(), 0);
}

#[test]
fn pal_timer_waker_callbacks_run_outside_the_registry_lock() {
    let state = TimerState::new();
    let faulty = panicking_waker();

    CLONE_PANICS.store(true, Ordering::Relaxed);
    let clone_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.register_waker(&faulty);
    }));
    CLONE_PANICS.store(false, Ordering::Relaxed);
    let Err(_) = clone_panic else {
        panic!("clone callback must panic");
    };
    assert!(!state.waker.is_poisoned());

    state.register_waker(&faulty);
    DROP_PANICS.store(true, Ordering::Relaxed);
    let drop_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.register_waker(std::task::Waker::noop());
    }));
    DROP_PANICS.store(false, Ordering::Relaxed);
    let Err(_) = drop_panic else {
        panic!("drop callback must panic");
    };
    assert!(!state.waker.is_poisoned());

    state.register_waker(&faulty);
    WAKE_PANICS.store(true, Ordering::Relaxed);
    let wake_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.complete();
    }));
    WAKE_PANICS.store(false, Ordering::Relaxed);
    let Err(_) = wake_panic else {
        panic!("wake callback must panic");
    };
    assert!(!state.waker.is_poisoned());
}
