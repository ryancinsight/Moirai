use std::{
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
    thread,
    time::Duration,
};

use futures::task::{waker, ArcWake};

use super::sleep;

struct WakeFlag {
    woke: AtomicBool,
}

impl WakeFlag {
    fn new() -> Self {
        Self {
            woke: AtomicBool::new(false),
        }
    }
}

impl ArcWake for WakeFlag {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.woke.store(true, Ordering::Release);
    }
}

#[test]
fn pal_timer_is_pending_before_deadline_and_wakes() {
    let wake_flag = Arc::new(WakeFlag::new());
    let waker = waker(Arc::clone(&wake_flag));
    let mut context = Context::from_waker(&waker);
    let mut timer = Box::pin(sleep(Duration::from_millis(20)));

    assert!(matches!(timer.as_mut().poll(&mut context), Poll::Pending));
    assert!(!wake_flag.woke.load(Ordering::Acquire));

    thread::sleep(Duration::from_millis(40));

    assert!(wake_flag.woke.load(Ordering::Acquire));
    assert!(matches!(
        timer.as_mut().poll(&mut context),
        Poll::Ready(Ok(()))
    ));
}

#[test]
fn pal_timer_zero_duration_completes_immediately() {
    let wake_flag = Arc::new(WakeFlag::new());
    let waker = waker(wake_flag);
    let mut context = Context::from_waker(&waker);
    let mut timer = Box::pin(sleep(Duration::ZERO));

    assert!(matches!(
        timer.as_mut().poll(&mut context),
        Poll::Ready(Ok(()))
    ));
}
