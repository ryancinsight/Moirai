use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use crate::timer::clamped_deadline;
use crate::timer::delay::Delay;

/// Interval timer for repeated operations
pub struct Interval {
    pub(super) next_tick: Instant,
    pub(super) period: Duration,
    pub(super) delay: Option<Delay>,
}

impl Interval {
    pub(super) fn new(period: Duration) -> Self {
        let next_tick = clamped_deadline(Instant::now(), period);
        Self {
            next_tick,
            period,
            delay: None,
        }
    }

    pub(super) fn new_at(start: Instant, period: Duration) -> Self {
        Self {
            next_tick: start,
            period,
            delay: None,
        }
    }

    /// Get the next tick time
    pub fn next_tick(&self) -> Instant {
        self.next_tick
    }

    /// Reset the interval to start from now
    pub fn reset(&mut self) {
        self.next_tick = clamped_deadline(Instant::now(), self.period);
        self.delay = None;
    }

    /// Change the interval period
    pub fn set_period(&mut self, period: Duration) {
        self.period = period;
        self.next_tick = clamped_deadline(Instant::now(), period);
        self.delay = None;
    }

    /// Wait for the next tick
    pub async fn next(&mut self) -> Instant {
        if self.delay.is_none() {
            self.delay = Some(Delay::until(self.next_tick));
        }

        if let Some(delay) = &mut self.delay {
            delay.await;
            let tick_time = self.next_tick;
            self.next_tick = clamped_deadline(self.next_tick, self.period);
            self.delay = None;
            tick_time
        } else {
            Instant::now()
        }
    }
}

impl Future for Interval {
    type Output = Instant;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.delay.is_none() {
            self.delay = Some(Delay::until(self.next_tick));
        }

        if let Some(delay) = &mut self.delay {
            match Pin::new(delay).poll(cx) {
                Poll::Ready(()) => {
                    let tick_time = self.next_tick;
                    let period = self.period;
                    self.next_tick = clamped_deadline(self.next_tick, period);
                    self.delay = None;
                    Poll::Ready(tick_time)
                }
                Poll::Pending => Poll::Pending,
            }
        } else {
            Poll::Pending
        }
    }
}
