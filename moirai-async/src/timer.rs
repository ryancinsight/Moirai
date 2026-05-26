//! Async timer primitives for Moirai concurrency library.
//!
//! This module provides async timer functionality including delays, intervals,
//! timeouts, and timer wheels. Following SLAP principle with focused
//! responsibility on time-based async operations.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

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

struct TimerRegistration {
    waker: Mutex<Option<Waker>>,
    cancelled: AtomicBool,
}

impl TimerRegistration {
    fn new(waker: Waker) -> Arc<Self> {
        Arc::new(Self {
            waker: Mutex::new(Some(waker)),
            cancelled: AtomicBool::new(false),
        })
    }

    fn replace_waker(&self, waker: &Waker) {
        let mut stored = self.waker.lock().unwrap();
        match stored.as_ref() {
            Some(current) if current.will_wake(waker) => {}
            _ => *stored = Some(waker.clone()),
        }
    }

    fn cancel(&self) {
        self.cancelled.store(true, AtomicOrdering::Release);
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(AtomicOrdering::Acquire)
    }

    fn wake(&self) {
        if let Some(waker) = self.waker.lock().unwrap().take() {
            waker.wake();
        }
    }
}

struct ScheduledTimer {
    deadline: Instant,
    sequence: u64,
    registration: Arc<TimerRegistration>,
}

impl PartialEq for ScheduledTimer {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline && self.sequence == other.sequence
    }
}

impl Eq for ScheduledTimer {}

impl PartialOrd for ScheduledTimer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledTimer {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .deadline
            .cmp(&self.deadline)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

struct TimerDriver {
    state: Mutex<TimerDriverState>,
    available: Condvar,
}

struct TimerDriverState {
    timers: BinaryHeap<ScheduledTimer>,
    next_sequence: u64,
}

impl TimerDriver {
    fn start() -> Arc<Self> {
        let driver = Arc::new(Self {
            state: Mutex::new(TimerDriverState {
                timers: BinaryHeap::new(),
                next_sequence: 0,
            }),
            available: Condvar::new(),
        });

        let worker = Arc::clone(&driver);
        std::thread::Builder::new()
            .name("moirai-timer-driver".to_string())
            .spawn(move || worker.run())
            .expect("failed to start Moirai timer driver");

        driver
    }

    fn schedule(&self, deadline: Instant, registration: Arc<TimerRegistration>) {
        let mut state = self.state.lock().unwrap();
        let sequence = state.next_sequence;
        state.next_sequence = state.next_sequence.wrapping_add(1);
        state.timers.push(ScheduledTimer {
            deadline,
            sequence,
            registration,
        });
        self.available.notify_one();
    }

    fn run(&self) {
        let mut state = self.state.lock().unwrap();
        loop {
            while state
                .timers
                .peek()
                .is_some_and(|timer| timer.registration.is_cancelled())
            {
                state.timers.pop();
            }

            let Some(next_deadline) = state.timers.peek().map(|timer| timer.deadline) else {
                state = self.available.wait(state).unwrap();
                continue;
            };

            let now = Instant::now();
            if next_deadline <= now {
                let timer = state.timers.pop().expect("timer existed after peek");
                drop(state);
                if !timer.registration.is_cancelled() {
                    timer.registration.wake();
                }
                state = self.state.lock().unwrap();
                continue;
            }

            let timeout = next_deadline - now;
            let (guard, _) = self.available.wait_timeout(state, timeout).unwrap();
            state = guard;
        }
    }
}

fn timer_driver() -> &'static Arc<TimerDriver> {
    static DRIVER: OnceLock<Arc<TimerDriver>> = OnceLock::new();
    DRIVER.get_or_init(TimerDriver::start)
}

/// Create a delay future that completes after the specified duration
pub fn sleep(duration: Duration) -> Delay {
    Delay::new(duration)
}

/// Timeout wrapper for futures with comprehensive cancellation
pub struct Timeout<F> {
    future: F,
    delay: Delay,
}

impl<F> Timeout<F>
where
    F: Future,
{
    fn new(future: F, duration: Duration) -> Self {
        Self {
            future,
            delay: Delay::new(duration),
        }
    }
}

impl<F> Future for Timeout<F>
where
    F: Future,
{
    type Output = Result<F::Output, TimeoutError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Safety: once `Timeout<F>` is pinned, its fields are not moved in
        // `poll`. Projecting the generic future in place preserves support for
        // `!Unpin` futures without allocating a `Pin<Box<F>>`.
        let this = unsafe { self.get_unchecked_mut() };

        // First check if the future is ready
        let future = unsafe { Pin::new_unchecked(&mut this.future) };
        if let Poll::Ready(output) = future.poll(cx) {
            return Poll::Ready(Ok(output));
        }

        // Then check if the timeout has elapsed
        if let Poll::Ready(()) = Pin::new(&mut this.delay).poll(cx) {
            return Poll::Ready(Err(TimeoutError));
        }

        Poll::Pending
    }
}

/// Error returned when a timeout elapses
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeoutError;

impl std::fmt::Display for TimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("operation timed out")
    }
}

impl std::error::Error for TimeoutError {}

/// Timeout wrapper for futures with comprehensive cancellation
pub fn timeout<F>(duration: Duration, future: F) -> Timeout<F>
where
    F: Future,
{
    Timeout::new(future, duration)
}

/// Interval timer for repeated operations
pub struct Interval {
    next_tick: Instant,
    period: Duration,
    delay: Option<Delay>,
}

impl Interval {
    fn new(period: Duration) -> Self {
        let next_tick = Instant::now() + period;
        Self {
            next_tick,
            period,
            delay: None,
        }
    }

    fn new_at(start: Instant, period: Duration) -> Self {
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
        self.next_tick = Instant::now() + self.period;
        self.delay = None;
    }

    /// Change the interval period
    pub fn set_period(&mut self, period: Duration) {
        self.period = period;
        self.next_tick = Instant::now() + period;
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
            self.next_tick += self.period;
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
        // Create delay if needed
        if self.delay.is_none() {
            self.delay = Some(Delay::until(self.next_tick));
        }

        // Poll the delay
        if let Some(delay) = &mut self.delay {
            match Pin::new(delay).poll(cx) {
                Poll::Ready(()) => {
                    let tick_time = self.next_tick;
                    let period = self.period;
                    self.next_tick += period;
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

/// Create a new interval timer
pub fn interval(period: Duration) -> Interval {
    Interval::new(period)
}

/// Create an interval timer that starts at a specific time
pub fn interval_at(start: Instant, period: Duration) -> Interval {
    Interval::new_at(start, period)
}

mod wheel;
pub use wheel::{TimerCommand, TimerWheel};

/// Rate limiter using token bucket algorithm
pub struct RateLimiter {
    permits: u32,
    current_permits: u32,
    interval: Interval,
}

/// A permit from the rate limiter
pub struct RatePermit;

impl RateLimiter {
    /// Create a new rate limiter with specified permits per second
    pub fn new(permits_per_second: u32) -> Self {
        let interval_duration = if permits_per_second > 0 {
            Duration::from_nanos(1_000_000_000 / permits_per_second as u64)
        } else {
            Duration::from_secs(1)
        };

        Self {
            permits: permits_per_second,
            current_permits: permits_per_second,
            interval: Interval::new(interval_duration),
        }
    }

    /// Acquire a permit to perform an operation
    pub async fn acquire(&mut self) -> RatePermit {
        if self.current_permits > 0 {
            self.current_permits -= 1;
            return RatePermit;
        }

        // Wait for next interval
        self.interval.next().await;
        self.current_permits = self.permits - 1;
        RatePermit
    }

    /// Try to acquire a permit without waiting
    pub fn try_acquire(&mut self) -> Option<RatePermit> {
        if self.current_permits > 0 {
            self.current_permits -= 1;
            Some(RatePermit)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_delay_basic() {
        let delay = Delay::new(Duration::from_millis(10));

        // Test that delay can be created with proper deadline
        assert!(delay.deadline() > Instant::now());

        // Full async timing tests will be added with native runtime
    }

    #[test]
    fn test_sleep_function() {
        let timer = sleep(Duration::from_millis(10));

        // Test that sleep function creates a proper timer
        assert!(timer.deadline() > Instant::now());

        // Full async sleep tests will be added with native runtime
    }
}
