use std::time::Instant;

use super::super::{SleeperEvent, TimerState, sleep};
use super::*;

const SLEEPER_DEADLINE: Duration = Duration::from_secs(60);
const EVENT_TIMEOUT: Duration = Duration::from_secs(5);

fn receive_until(receiver: &std::sync::mpsc::Receiver<SleeperEvent>, expected: SleeperEvent) {
    loop {
        let event = receiver
            .recv_timeout(EVENT_TIMEOUT)
            .expect("sleeper must publish the expected event");
        if event == expected {
            return;
        }
        assert_eq!(
            event,
            SleeperEvent::Waiting,
            "only a spurious wake may precede the expected event"
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

#[test]
fn pal_timer_cancellation_before_wait_releases_the_waker() {
    let timer = sleep(SLEEPER_DEADLINE);
    let deadline = timer.deadline;
    let state = Arc::clone(&timer.state);
    let events = observe_sleeper(&state);
    let probe = Arc::new(CallbackProbe::new(Arc::clone(&state)));
    let task_waker = callback_waker(&probe);

    state.register_waker(&task_waker);
    assert!(registered_waker_matches(&state, &task_waker));
    drop(task_waker);
    drop(timer);
    assert_eq!(Arc::strong_count(&probe), 2);

    state
        .spawn_sleeper(deadline)
        .expect("real sleeper must spawn after pre-cancellation");
    receive_until(&events, SleeperEvent::Exited);
    assert_eq!(Arc::strong_count(&probe), 1);
    assert_eq!(probe.counts().drops, 2);
}

#[test]
fn pal_timer_drop_during_wait_is_nonblocking() {
    let mut timer = Box::pin(sleep(SLEEPER_DEADLINE));
    let state = Arc::clone(&timer.state);
    let events = observe_sleeper(&state);
    let probe = Arc::new(CallbackProbe::new(Arc::clone(&state)));
    let task_waker = callback_waker(&probe);
    let mut context = Context::from_waker(&task_waker);

    assert!(matches!(timer.as_mut().poll(&mut context), Poll::Pending));
    receive_until(&events, SleeperEvent::Waiting);
    drop(task_waker);
    assert_eq!(Arc::strong_count(&probe), 2);

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
    receive_until(&events, SleeperEvent::Exited);
    assert_eq!(Arc::strong_count(&probe), 1);
}

#[test]
fn pal_timer_spurious_unpark_reparks_the_sleeper() {
    let mut timer = Box::pin(sleep(SLEEPER_DEADLINE));
    let state = Arc::clone(&timer.state);
    let events = observe_sleeper(&state);
    let mut context = Context::from_waker(std::task::Waker::noop());

    assert!(matches!(timer.as_mut().poll(&mut context), Poll::Pending));
    receive_until(&events, SleeperEvent::Waiting);
    state
        .sleeper_thread
        .get()
        .expect("invariant: waiting follows sleeper publication")
        .unpark();
    receive_until(&events, SleeperEvent::Waiting);
    assert!(!state.is_completed());

    drop(timer);
    receive_until(&events, SleeperEvent::Exited);
}

#[test]
fn pal_timer_repeated_ready_polls_release_the_waker() {
    let mut timer = Box::pin(sleep(SLEEPER_DEADLINE));
    let state = Arc::clone(&timer.state);
    let events = observe_sleeper(&state);
    let probe = Arc::new(CallbackProbe::new(Arc::clone(&state)));
    let task_waker = callback_waker(&probe);

    {
        let mut context = Context::from_waker(&task_waker);
        assert!(matches!(timer.as_mut().poll(&mut context), Poll::Pending));
    }
    receive_until(&events, SleeperEvent::Waiting);
    drop(task_waker);
    assert_eq!(Arc::strong_count(&probe), 2);

    timer.as_mut().get_mut().deadline = Instant::now();
    let mut context = Context::from_waker(std::task::Waker::noop());
    for _ in 0..2 {
        assert!(matches!(
            timer.as_mut().poll(&mut context),
            Poll::Ready(Ok(()))
        ));
    }
    receive_until(&events, SleeperEvent::Exited);
    assert_eq!(Arc::strong_count(&probe), 1);
    assert_eq!(registered_waker_count(&state), 0);
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
            super::super::lifecycle(state.status.load(loom::sync::atomic::Ordering::Acquire)),
            super::super::TIMER_CANCELLED
        );
        assert_eq!(registered_waker_count(&state), 0);
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
        assert_eq!(registered_waker_count(&state), 0);
    });
}
