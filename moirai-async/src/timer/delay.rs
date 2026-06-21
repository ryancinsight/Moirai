use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use std::sync::Arc;

use crate::timer::registration::TimerRegistration;
use crate::timer::driver::timer_driver;

/// A future that completes after a specified duration
pub struct Delay {
    deadline: Instant,
    registration: Option<Arc<TimerRegistration>>,
}

impl Delay {
    /// Create a new delay that will complete after the specified duration
    pub fn new(duration: Duration) -> Self {
        Self {
            deadline: Instant::now() + duration,
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
        self.deadline = Instant::now() + duration;
        if let Some(registration) = self.registration.take() {
            registration.cancel();
            registration.wake();
        }
    }
}

impl Future for Delay {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if Instant::now() >= self.deadline {
            if let Some(registration) = self.registration.take() {
                registration.cancel();
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
            registration.cancel();
        }
    }
}
