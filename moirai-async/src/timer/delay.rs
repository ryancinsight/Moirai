use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use crate::timer::clamped_deadline;
use crate::timer::driver::timer_driver;
use crate::timer::registration::{TimerRegistration, WakerSlot};

/// A future that completes after a specified duration
pub struct Delay {
    deadline: Instant,
    registration: Option<Arc<TimerRegistration>>,
}

impl Delay {
    /// Create a new delay that will complete after the specified duration
    pub fn new(duration: Duration) -> Self {
        Self {
            deadline: clamped_deadline(Instant::now(), duration),
            registration: None,
        }
    }

    /// Create a delay that completes at a specific instant
    pub fn until(deadline: Instant) -> Self {
        Self {
            deadline,
            registration: None,
        }
    }

    /// Get the deadline for this delay
    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    /// Reset the delay to a new duration from now
    pub fn reset(&mut self, duration: Duration) {
        self.deadline = clamped_deadline(Instant::now(), duration);
        if let Some(registration) = self.registration.take()
            && let Some(waker) = timer_driver().cancel(&registration)
        {
            waker.wake();
        }
    }
}

impl Future for Delay {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if Instant::now() >= self.deadline {
            if let Some(registration) = self.registration.take() {
                drop(timer_driver().cancel(&registration));
            }
            Poll::Ready(())
        } else {
            match &self.registration {
                Some(registration) => {
                    if matches!(registration.replace_waker(cx.waker()), WakerSlot::Consumed) {
                        self.registration = None;
                        return Poll::Ready(());
                    }
                }
                None => {
                    let registration = TimerRegistration::new(cx.waker().clone());
                    timer_driver().schedule(self.deadline, Arc::clone(&registration));
                    self.registration = Some(registration);
                }
            }
            Poll::Pending
        }
    }
}

impl Drop for Delay {
    fn drop(&mut self) {
        if let Some(registration) = self.registration.take() {
            drop(timer_driver().cancel(&registration));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Delay, TimerRegistration};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Wake, Waker};
    use std::time::Duration;

    struct TestWake {
        owner: Arc<()>,
        wakes: Arc<AtomicUsize>,
    }

    impl Wake for TestWake {
        fn wake(self: Arc<Self>) {
            assert!(Arc::strong_count(&self.owner) > 0);
            self.wakes.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn test_waker(owner: &Arc<()>, wakes: &Arc<AtomicUsize>) -> Waker {
        Waker::from(Arc::new(TestWake {
            owner: Arc::clone(owner),
            wakes: Arc::clone(wakes),
        }))
    }

    #[test]
    fn drop_detaches_registered_delay_waker() {
        let owner = Arc::new(());
        let wakes = Arc::new(AtomicUsize::new(0));
        let waker = test_waker(&owner, &wakes);
        let mut context = Context::from_waker(&waker);
        let mut delay = Delay::new(Duration::from_secs(60));
        assert!(Pin::new(&mut delay).poll(&mut context).is_pending());
        drop(waker);
        assert_eq!(Arc::strong_count(&owner), 2);

        drop(delay);

        assert_eq!(Arc::strong_count(&owner), 1);
        assert_eq!(wakes.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn reset_wakes_detached_registration() {
        let owner = Arc::new(());
        let wakes = Arc::new(AtomicUsize::new(0));
        let waker = test_waker(&owner, &wakes);
        let mut context = Context::from_waker(&waker);
        let mut delay = Delay::new(Duration::from_secs(60));
        assert!(Pin::new(&mut delay).poll(&mut context).is_pending());

        delay.reset(Duration::ZERO);

        assert_eq!(wakes.load(Ordering::SeqCst), 1);
        assert!(Pin::new(&mut delay).poll(&mut context).is_ready());
    }

    #[test]
    fn consumed_registration_completes_pending_poll() {
        let registration = TimerRegistration::new(Waker::noop().clone());
        drop(registration.take_waker());
        let mut delay = Delay {
            deadline: std::time::Instant::now() + Duration::from_secs(60),
            registration: Some(registration),
        };
        let mut context = Context::from_waker(Waker::noop());

        assert!(Pin::new(&mut delay).poll(&mut context).is_ready());
        assert!(delay.registration.is_none());
    }
}
