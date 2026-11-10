use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use crate::timer::clamped_deadline;
use crate::timer::driver::timer_driver;
use crate::timer::registration::TimerRegistration;

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
        if let Some(registration) = self.registration.take() {
            timer_driver().cancel(&registration);
            registration.wake();
        }
    }
}

impl Future for Delay {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if Instant::now() >= self.deadline {
            if let Some(registration) = self.registration.take() {
                timer_driver().cancel(&registration);
            }
            Poll::Ready(())
        } else {
            match &self.registration {
                Some(registration) => registration.replace_waker(cx.waker()),
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
            timer_driver().cancel(&registration);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Delay;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Waker};
    use std::time::Duration;

    #[test]
    fn dropping_delays_cancels_their_registrations() {
        let mut context = Context::from_waker(Waker::noop());
        let mut delays: Vec<_> = (0..103)
            .map(|_| Delay::new(Duration::from_secs(3600)))
            .collect();
        for delay in &mut delays {
            assert!(Pin::new(delay).poll(&mut context).is_pending());
        }
        let registrations: Vec<_> = delays
            .iter()
            .map(|delay| {
                Arc::clone(
                    delay
                        .registration
                        .as_ref()
                        .expect("invariant: a pending delay owns its registration"),
                )
            })
            .collect();
        assert_eq!(registrations.len(), 103);
        assert!(
            registrations
                .iter()
                .all(|registration| !registration.is_cancelled())
        );

        delays.truncate(1);

        assert_eq!(
            registrations
                .iter()
                .filter(|registration| registration.is_cancelled())
                .count(),
            102
        );
        assert_eq!(
            registrations
                .iter()
                .filter(|registration| !registration.is_cancelled())
                .count(),
            1
        );
        assert!(Pin::new(&mut delays[0]).poll(&mut context).is_pending());
        assert!(Arc::ptr_eq(
            delays[0]
                .registration
                .as_ref()
                .expect("invariant: the surviving delay remains registered"),
            &registrations[0]
        ));
    }
}
