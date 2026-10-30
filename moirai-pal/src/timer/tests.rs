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

#[test]
fn pal_timer_is_pending_before_deadline_and_wakes() {
    let (wake_tx, wake_rx) = std::sync::mpsc::sync_channel(1);
    let wake_flag = Arc::new(WakeFlag::new(wake_tx));
    let waker = waker(Arc::clone(&wake_flag));
    let mut context = Context::from_waker(&waker);
    let mut timer = Box::pin(
        sleep(Duration::from_millis(20))
            .expect("invariant: twenty milliseconds fits in the instant range"),
    );

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
    let mut timer = Box::pin(
        sleep(Duration::ZERO).expect("invariant: zero duration fits in the instant range"),
    );

    assert!(matches!(
        timer.as_mut().poll(&mut context),
        Poll::Ready(Ok(()))
    ));
}

#[test]
fn pal_timer_extreme_duration_does_not_panic() {
    for duration in [Duration::MAX, Duration::from_secs(u64::MAX)] {
        let error = match super::Timer::new(duration) {
            Ok(_) => panic!("unrepresentable timer duration must be rejected"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }

    let start = Instant::now();
    let duration = Duration::from_secs(366 * 24 * 60 * 60);
    let expected = start
        .checked_add(duration)
        .expect("invariant: one year fits in the platform instant range");
    assert_eq!(
        super::deadline_after(start, duration)
            .expect("invariant: the same representable deadline must be accepted"),
        expected
    );
}

#[test]
fn pal_timer_preserves_exact_century_plus_duration() {
    const ONE_HUNDRED_YEARS: Duration = Duration::from_secs(100 * 365 * 24 * 60 * 60);
    let duration = ONE_HUNDRED_YEARS
        .checked_add(Duration::from_secs(1))
        .expect("invariant: adding one second to one hundred years cannot overflow Duration");
    let start = Instant::now();
    let expected = start
        .checked_add(duration)
        .expect("invariant: supported targets represent a timer beyond one hundred years");
    assert_eq!(
        super::deadline_after(start, duration)
            .expect("invariant: a representable deadline must retain the requested duration"),
        expected
    );
}

#[test]
fn pal_timer_cancellation_before_wait_releases_the_waker() {
    let timer =
        sleep(SLEEPER_DEADLINE).expect("invariant: the test deadline fits in the instant range");
    let deadline = timer.deadline;
    let state = Arc::clone(&timer.state);
    let events = observe_sleeper(&state);
    let (observer, task_waker, _wake_events) = tracked_waker();

    state.register_waker(&task_waker);
    drop(task_waker);
    assert_eq!(observer.strong_count(), 1);
    drop(timer);

    state
        .spawn_sleeper(deadline)
        .expect("real sleeper must spawn after pre-cancellation");
    receive_until(&events, SleeperEvent::Exited, event_deadline());
    assert_eq!(observer.strong_count(), 0);
}

#[test]
fn pal_timer_drop_during_wait_is_nonblocking() {
    let (observer, task_waker, _wake_events) = tracked_waker();
    let mut timer = Box::pin(
        sleep(SLEEPER_DEADLINE).expect("invariant: the test deadline fits in the instant range"),
    );
    let state = Arc::clone(&timer.state);
    let events = observe_sleeper(&state);
    let deadline = event_deadline();

    {
        let mut context = Context::from_waker(&task_waker);
        assert!(matches!(timer.as_mut().poll(&mut context), Poll::Pending));
    }
    receive_until(&events, SleeperEvent::Waiting, deadline);
    drop(task_waker);
    assert_eq!(observer.strong_count(), 1);

    let registry = state
        .waker
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let (dropped_tx, dropped_rx) = std::sync::mpsc::sync_channel(0);
    let dropper = std::thread::spawn(move || {
        drop(timer);
        dropped_tx
            .send(())
            .expect("drop observer must remain connected");
    });
    dropped_rx
        .recv_timeout(EVENT_TIMEOUT)
        .expect("timer drop must not wait for the waker registry");
    drop(registry);
    dropper.join().expect("timer drop thread must exit");
    receive_until(&events, SleeperEvent::Exited, deadline);
    assert_eq!(observer.strong_count(), 0);
}

#[test]
fn pal_timer_spurious_unpark_reparks_the_sleeper() {
    let mut timer = Box::pin(
        sleep(SLEEPER_DEADLINE).expect("invariant: the test deadline fits in the instant range"),
    );
    let state = Arc::clone(&timer.state);
    let events = observe_sleeper(&state);
    let mut context = Context::from_waker(std::task::Waker::noop());
    let deadline = event_deadline();

    assert!(matches!(timer.as_mut().poll(&mut context), Poll::Pending));
    receive_until(&events, SleeperEvent::Waiting, deadline);
    state
        .sleeper_thread
        .get()
        .expect("invariant: waiting follows sleeper publication")
        .unpark();
    receive_until(&events, SleeperEvent::Waiting, deadline);
    assert!(!state.is_completed());

    drop(timer);
    receive_until(&events, SleeperEvent::Exited, deadline);
}

#[test]
fn pal_timer_repeated_ready_polls_release_the_waker() {
    let (observer, task_waker, _wake_events) = tracked_waker();
    let mut timer = Box::pin(
        sleep(SLEEPER_DEADLINE).expect("invariant: the test deadline fits in the instant range"),
    );
    let state = Arc::clone(&timer.state);
    let events = observe_sleeper(&state);
    let deadline = event_deadline();

    {
        let mut context = Context::from_waker(&task_waker);
        assert!(matches!(timer.as_mut().poll(&mut context), Poll::Pending));
    }
    receive_until(&events, SleeperEvent::Waiting, deadline);
    drop(task_waker);
    assert_eq!(observer.strong_count(), 1);

    timer.as_mut().get_mut().deadline = Instant::now();
    let mut context = Context::from_waker(std::task::Waker::noop());
    for _ in 0..2 {
        assert!(matches!(
            timer.as_mut().poll(&mut context),
            Poll::Ready(Ok(()))
        ));
    }
    receive_until(&events, SleeperEvent::Exited, deadline);
    assert_eq!(observer.strong_count(), 0);
}

#[test]
fn pal_timer_spawn_failure_preserves_error_and_releases_the_waker() {
    let state = Arc::new(TimerState::new());
    let (observer, task_waker, _wake_events) = tracked_waker();
    state.register_waker(&task_waker);
    drop(task_waker);
    assert_eq!(observer.strong_count(), 1);

    let error = state
        .finish_spawn(Err(std::io::Error::from_raw_os_error(i32::MAX)))
        .expect_err("injected worker spawn failure must remain an error");
    assert_eq!(error.raw_os_error(), Some(i32::MAX));
    assert!(state.is_completed());
    assert_eq!(observer.strong_count(), 0);
}

#[cfg(loom)]
#[test]
fn loom_timer_cancellation_publication_has_no_lost_wakeup() {
    let mut model = loom::model::Builder::new();
    model.preemption_bound = Some(4);
    model.check(|| {
        let state = Arc::new(TimerState::new());
        state.register_waker(std::task::Waker::noop());
        let worker_state = Arc::clone(&state);
        let deadline = Instant::now()
            .checked_add(SLEEPER_DEADLINE)
            .expect("invariant: one minute fits in the platform instant range");
        let worker = loom::thread::spawn(move || worker_state.run_sleeper(deadline));

        state.cancel();
        worker
            .join()
            .expect("timer worker must observe cancellation");
        assert_eq!(
            super::lifecycle(state.status.load(loom::sync::atomic::Ordering::Acquire)),
            super::TIMER_CANCELLED
        );
        assert!(
            state
                .waker
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_none()
        );
    });
}

#[cfg(loom)]
#[test]
fn loom_timer_completion_and_registration_release_the_waker() {
    let mut model = loom::model::Builder::new();
    model.preemption_bound = Some(4);
    model.check(|| {
        let state = Arc::new(TimerState::new());
        let registering = Arc::clone(&state);
        let completing = Arc::clone(&state);
        let register = loom::thread::spawn(move || {
            registering.register_waker(std::task::Waker::noop());
            if registering.is_completed() {
                registering.clear_waker();
            }
        });
        let complete = loom::thread::spawn(move || completing.complete());

        register.join().expect("registration must finish");
        complete.join().expect("completion must finish");
        assert!(state.is_completed());
        assert!(
            state
                .waker
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_none()
        );
    });
}
