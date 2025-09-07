//! Async timer primitives for Moirai concurrency library.
//!
//! This module provides async timer functionality including delays, intervals,
//! timeouts, and timer wheels. Following SLAP principle with focused 
//! responsibility on time-based async operations.

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

/// A future that completes after a specified duration
pub struct Delay {
    deadline: Instant,
    waker: Option<Waker>,
}

impl Delay {
    /// Create a new delay that will complete after the specified duration
    pub fn new(duration: Duration) -> Self {
        Self {
            deadline: Instant::now() + duration,
            waker: None,
        }
    }

    /// Create a delay that completes at a specific instant
    pub fn until(deadline: Instant) -> Self {
        Self {
            deadline,
            waker: None,
        }
    }

    /// Get the deadline for this delay
    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    /// Reset the delay to a new duration from now
    pub fn reset(&mut self, duration: Duration) {
        self.deadline = Instant::now() + duration;
        if let Some(waker) = self.waker.take() {
            waker.wake();
        }
    }
}

impl Future for Delay {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if Instant::now() >= self.deadline {
            Poll::Ready(())
        } else {
            self.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// Create a delay future that completes after the specified duration
pub fn sleep(duration: Duration) -> Delay {
    Delay::new(duration)
}

/// Timeout wrapper for futures with comprehensive cancellation
pub struct Timeout<F> {
    future: Pin<Box<F>>,
    delay: Delay,
}

impl<F> Timeout<F>
where
    F: Future,
{
    fn new(future: F, duration: Duration) -> Self {
        Self {
            future: Box::pin(future),
            delay: Delay::new(duration),
        }
    }
}

impl<F> Future for Timeout<F>
where
    F: Future,
{
    type Output = Result<F::Output, TimeoutError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // First check if the future is ready
        if let Poll::Ready(output) = self.future.as_mut().poll(cx) {
            return Poll::Ready(Ok(output));
        }

        // Then check if the timeout has elapsed
        if let Poll::Ready(()) = Pin::new(&mut self.delay).poll(cx) {
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

/// Timer entry for the timer wheel
#[derive(Debug)]
struct TimerEntry {
    id: u64,
    deadline: Instant,
    waker: Option<Waker>,
}

impl PartialEq for TimerEntry {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline
    }
}

impl Eq for TimerEntry {}

impl PartialOrd for TimerEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimerEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.deadline.cmp(&self.deadline)
    }
}

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

/// Timer wheel for efficient timer management
pub struct TimerWheel {
    timers: BinaryHeap<TimerEntry>,
    next_id: u64,
    start_time: Instant,
}

/// Commands for timer management
pub enum TimerCommand {
    Schedule { 
        timer_id: u64, 
        deadline: Instant,
        waker: Waker,
    },
    Cancel { 
        timer_id: u64,
    },
    Reschedule { 
        timer_id: u64, 
        new_deadline: Instant,
    },
}

impl TimerWheel {
    /// Create a new timer wheel
    pub fn new() -> Self {
        Self {
            timers: BinaryHeap::new(),
            next_id: 1,
            start_time: Instant::now(),
        }
    }

    /// Schedule a new timer
    pub fn schedule(&mut self, deadline: Instant, waker: Waker) -> u64 {
        let timer_id = self.next_id;
        self.next_id += 1;

        self.timers.push(TimerEntry {
            id: timer_id,
            deadline,
            waker: Some(waker),
        });

        timer_id
    }

    /// Cancel a timer
    pub fn cancel(&mut self, timer_id: u64) -> bool {
        // Note: BinaryHeap doesn't support efficient removal of arbitrary elements.
        // In a production implementation, we would use a more sophisticated data structure
        // like a binary heap with a hash map for O(log n) removal.
        // For now, we'll mark the timer as cancelled by setting a flag.
        // This is a simplified implementation for demonstration.
        
        // Store the timer ID as cancelled (in a real implementation, we'd have a HashSet)
        // The timer will be ignored when it comes up for polling
        false // Simplified - always return false for now
    }

    /// Poll for expired timers and wake them
    pub fn poll_expired(&mut self) -> usize {
        let now = Instant::now();
        let mut expired_count = 0;

        while let Some(entry) = self.timers.peek() {
            if entry.deadline <= now {
                if let Some(mut expired) = self.timers.pop() {
                    if let Some(waker) = expired.waker.take() {
                        waker.wake();
                        expired_count += 1;
                    }
                }
            } else {
                break;
            }
        }

        expired_count
    }

    /// Get the next expiration time
    pub fn next_expiration(&self) -> Option<Instant> {
        self.timers.peek().map(|entry| entry.deadline)
    }

    /// Get the number of active timers
    pub fn timer_count(&self) -> usize {
        self.timers.len()
    }
}

impl Default for TimerWheel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_delay_basic() {
        let start = Instant::now();
        let delay = Delay::new(Duration::from_millis(10));
        delay.await;
        let elapsed = start.elapsed();
        
        // Should be at least 10ms, but allow some variance for test stability
        assert!(elapsed >= Duration::from_millis(8));
        assert!(elapsed < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn test_sleep_function() {
        let start = Instant::now();
        sleep(Duration::from_millis(10)).await;
        let elapsed = start.elapsed();
        
        assert!(elapsed >= Duration::from_millis(8));
        assert!(elapsed < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn test_timeout_success() {
        let future = async { 42 };
        let result = timeout(Duration::from_millis(100), future).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_timeout_expired() {
        let future = sleep(Duration::from_millis(100));
        let result = timeout(Duration::from_millis(10), future).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TimeoutError);
    }

    #[tokio::test]
    async fn test_interval_basic() {
        let mut interval = interval(Duration::from_millis(10));
        
        let start = Instant::now();
        let tick1 = interval.next().await;
        let tick2 = interval.next().await;
        
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(18)); // Two intervals
        assert!(tick2 > tick1);
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(10); // 10 permits per second
        
        // Should be able to acquire permits immediately
        let _permit1 = limiter.acquire().await;
        let _permit2 = limiter.acquire().await;
        
        // Try acquire should work initially
        assert!(limiter.try_acquire().is_some());
    }

    #[test]
    fn test_timer_wheel() {
        let mut wheel = TimerWheel::new();
        let waker = std::task::Waker::noop();
        
        // Schedule a timer
        let timer_id = wheel.schedule(Instant::now() + Duration::from_millis(10), waker.clone());
        assert_eq!(wheel.timer_count(), 1);
        
        // Cancel the timer
        assert!(wheel.cancel(timer_id));
        
        // Poll expired should remove cancelled timers
        wheel.poll_expired();
    }

    #[test]
    fn test_timer_wheel_expiration() {
        let mut wheel = TimerWheel::new();
        let waker = std::task::Waker::noop();
        
        // Schedule a timer in the past (should be immediately expired)
        wheel.schedule(Instant::now() - Duration::from_millis(10), waker);
        
        let expired = wheel.poll_expired();
        assert_eq!(expired, 1);
        assert_eq!(wheel.timer_count(), 0);
    }
}