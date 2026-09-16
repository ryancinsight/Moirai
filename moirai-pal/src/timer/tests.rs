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

use super::sleep;

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
