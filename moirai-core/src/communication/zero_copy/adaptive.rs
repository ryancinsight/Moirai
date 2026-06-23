//! Adaptive batching and throughput monitor for zero-copy communication.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::channel::{ZeroCopyChannel, ZeroCopyReceiver, ZeroCopySender};
use super::error::{ZeroCopyError, ZeroCopyResult};

/// Adaptive threshold for batching decisions.
#[derive(Debug)]
pub struct AdaptiveThreshold {
    current: AtomicUsize,
    min_threshold: usize,
    max_threshold: usize,
    adaptation_rate: f64,
    throughput_history: std::sync::Mutex<VecDeque<f64>>,
    last_adaptation: std::sync::Mutex<Instant>,
}

impl AdaptiveThreshold {
    /// Create a new adaptive batch size controller
    pub fn new(initial: usize, min: usize, max: usize, adaptation_rate: f64) -> Self {
        assert!(min <= initial && initial <= max);
        assert!((0.0..=1.0).contains(&adaptation_rate) && adaptation_rate > 0.0);
        Self {
            current: AtomicUsize::new(initial),
            min_threshold: min,
            max_threshold: max,
            adaptation_rate,
            throughput_history: std::sync::Mutex::new(VecDeque::with_capacity(10)),
            last_adaptation: std::sync::Mutex::new(Instant::now()),
        }
    }

    /// Get the current batch size
    pub fn current(&self) -> usize {
        self.current.load(Ordering::Relaxed)
    }

    /// Update batch size based on performance metrics
    pub fn update(&self, throughput: f64, latency: Duration) {
        let mut history = self.throughput_history.lock().unwrap();
        let mut last = self.last_adaptation.lock().unwrap();
        if last.elapsed() < Duration::from_millis(100) {
            return;
        }
        history.push_back(throughput);
        if history.len() > 10 {
            history.pop_front();
        }
        let avg = if history.is_empty() {
            throughput
        } else {
            history.iter().sum::<f64>() / history.len() as f64
        };
        let cur = self.current() as f64;
        let mut new_threshold = if throughput > avg * 1.1 {
            if latency < Duration::from_micros(100) {
                (cur * (1.0 + self.adaptation_rate)).ceil()
            } else {
                cur
            }
        } else if throughput < avg * 0.9 {
            (cur * (1.0 - self.adaptation_rate)).floor()
        } else {
            cur
        };
        if new_threshold < self.min_threshold as f64 {
            new_threshold = self.min_threshold as f64;
        }
        if new_threshold > self.max_threshold as f64 {
            new_threshold = self.max_threshold as f64;
        }
        self.current
            .store(new_threshold as usize, Ordering::Relaxed);
        *last = Instant::now();
    }
}

impl Default for AdaptiveThreshold {
    fn default() -> Self {
        Self::new(32, 1, 1024, 0.1)
    }
}

/// Throughput monitor for adaptive batching.
#[derive(Debug)]
pub struct ThroughputMonitor {
    message_count: AtomicUsize,
    start_time: std::sync::Mutex<Instant>,
    last_measurement: std::sync::Mutex<Instant>,
    recent_throughput: std::sync::Mutex<VecDeque<f64>>,
}

impl ThroughputMonitor {
    /// Create a new throughput monitor
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            message_count: AtomicUsize::new(0),
            start_time: std::sync::Mutex::new(now),
            last_measurement: std::sync::Mutex::new(now),
            recent_throughput: std::sync::Mutex::new(VecDeque::with_capacity(10)),
        }
    }

    /// Record a message being processed
    pub fn record_message(&self) {
        self.message_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current throughput in messages per second
    pub fn current_throughput(&self) -> f64 {
        let count = self.message_count.load(Ordering::Relaxed);
        let start = self.start_time.lock().unwrap();
        let elapsed = start.elapsed();
        if elapsed.as_secs_f64() > 0.0 {
            count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get recent throughput over the last measurement window
    pub fn recent_throughput(&self) -> f64 {
        let t = self.recent_throughput.lock().unwrap();
        if t.is_empty() {
            0.0
        } else {
            t.iter().sum::<f64>() / t.len() as f64
        }
    }

    /// Update throughput measurements
    pub fn update(&self) {
        let mut last = self.last_measurement.lock().unwrap();
        let mut rt = self.recent_throughput.lock().unwrap();
        let now = Instant::now();
        if now.duration_since(*last) >= Duration::from_millis(100) {
            rt.push_back(self.current_throughput());
            if rt.len() > 10 {
                rt.pop_front();
            }
            *last = now;
        }
    }

    /// Get time since last measurement
    pub fn idle_time(&self) -> Duration {
        self.last_measurement.lock().unwrap().elapsed()
    }
}

impl Default for ThroughputMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive batching channel built on top of ZeroCopyChannel.
pub struct AdaptiveBatchChannel<T> {
    _zero_copy: ZeroCopyChannel<T>,
    _batch_buffer: std::sync::Mutex<VecDeque<T>>,
    _adaptive_threshold: AdaptiveThreshold,
    _throughput_monitor: ThroughputMonitor,
    _max_batch_delay: Duration,
    _last_flush: std::sync::Mutex<Instant>,
}

impl<T> AdaptiveBatchChannel<T> {
    /// Create a new adaptive batch channel pair
    pub fn new(
        capacity: usize,
        max_batch_delay: Duration,
    ) -> ZeroCopyResult<(AdaptiveBatchSender<T>, AdaptiveBatchReceiver<T>)> {
        let (sender, receiver) = ZeroCopyChannel::new(capacity)?;
        let s = AdaptiveBatchSender {
            sender,
            batch_buffer: std::sync::Mutex::new(VecDeque::new()),
            adaptive_threshold: AdaptiveThreshold::default(),
            throughput_monitor: ThroughputMonitor::new(),
            max_batch_delay,
            last_flush: std::sync::Mutex::new(Instant::now()),
        };
        let r = AdaptiveBatchReceiver { receiver };
        Ok((s, r))
    }
}

/// Adaptive batch sender for zero-copy channels.
///
/// Automatically adjusts batch sizes based on throughput and latency metrics.
pub struct AdaptiveBatchSender<T> {
    sender: ZeroCopySender<T>,
    batch_buffer: std::sync::Mutex<VecDeque<T>>,
    adaptive_threshold: AdaptiveThreshold,
    throughput_monitor: ThroughputMonitor,
    max_batch_delay: Duration,
    last_flush: std::sync::Mutex<Instant>,
}

impl<T> AdaptiveBatchSender<T> {
    /// Send a value with adaptive batching based on throughput
    pub fn send_adaptive(&self, value: T) -> ZeroCopyResult<()> {
        self.throughput_monitor.record_message();
        {
            let mut buf = self.batch_buffer.lock().unwrap();
            buf.push_back(value);
        }
        if self.should_flush_batch() {
            self.flush_batch()?;
            self.adjust_batch_size();
        }
        Ok(())
    }

    fn should_flush_batch(&self) -> bool {
        let len = { self.batch_buffer.lock().unwrap().len() };
        len >= self.adaptive_threshold.current()
            || self.last_flush.lock().unwrap().elapsed() > self.max_batch_delay
    }

    fn flush_batch(&self) -> ZeroCopyResult<()> {
        use std::thread;

        // Local queue of items to send; we guarantee all buffered items are sent before returning Ok
        let mut pending: VecDeque<T> = VecDeque::new();

        loop {
            // If we do not have local pending items, drain from the shared buffer
            if pending.is_empty() {
                pending = {
                    let mut buf = self.batch_buffer.lock().unwrap();
                    buf.drain(..).collect()
                };
            }

            // Nothing to flush
            if pending.is_empty() {
                let mut last = self.last_flush.lock().unwrap();
                *last = Instant::now();
                return Ok(());
            }

            // Try to send as many as possible without holding any locks
            while let Some(v) = pending.pop_front() {
                match self.sender.send(v) {
                    Ok(()) => {}
                    Err((v, e)) => {
                        match e {
                            ZeroCopyError::Closed => {
                                // Requeue remaining items and return terminal error
                                let mut buf = self.batch_buffer.lock().unwrap();
                                // Put back current item and any remaining (reverse to preserve order when pushing_front)
                                buf.push_front(v);
                                for x in pending.into_iter().rev() {
                                    buf.push_front(x);
                                }
                                return Err(ZeroCopyError::Closed);
                            }
                            ZeroCopyError::Full | ZeroCopyError::WouldBlock => {
                                // Put back current item at the front and retry after yielding
                                pending.push_front(v);
                                break;
                            }
                            _other => {
                                // Unexpected error path: requeue and treat as transient
                                pending.push_front(v);
                                break;
                            }
                        }
                    }
                }
            }

            // If we still have pending items, the channel was full; yield and retry
            if !pending.is_empty() {
                thread::yield_now();
            }
            // Otherwise, loop will drain again and likely exit updating last_flush
        }
    }

    fn adjust_batch_size(&self) {
        self.throughput_monitor.update();
        let t = self.throughput_monitor.recent_throughput();
        let l = self.last_flush.lock().unwrap().elapsed();
        self.adaptive_threshold.update(t, l);
    }

    /// Force flush the current batch
    pub fn flush(&self) -> ZeroCopyResult<()> {
        self.flush_batch()
    }

    /// Get current batch statistics
    pub fn batch_stats(&self) -> BatchStats {
        BatchStats {
            current_threshold: self.adaptive_threshold.current(),
            pending_messages: self.batch_buffer.lock().unwrap().len(),
            current_throughput: self.throughput_monitor.current_throughput(),
            recent_throughput: self.throughput_monitor.recent_throughput(),
            time_since_last_flush: self.last_flush.lock().unwrap().elapsed(),
        }
    }
}

/// Adaptive batch receiver for zero-copy communication
pub struct AdaptiveBatchReceiver<T> {
    receiver: ZeroCopyReceiver<T>,
}

impl<T> AdaptiveBatchReceiver<T> {
    /// Receive a value (blocking)
    pub fn recv(&self) -> ZeroCopyResult<T> {
        self.receiver.recv()
    }

    /// Try to receive a value (non-blocking)
    pub fn try_recv(&self) -> ZeroCopyResult<T> {
        self.receiver.try_recv()
    }
}

/// Statistics for adaptive batching.
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Current adaptive threshold for batching
    pub current_threshold: usize,
    /// Number of pending messages in batch
    pub pending_messages: usize,
    /// Current throughput (messages per second)
    pub current_throughput: f64,
    /// Recent average throughput
    pub recent_throughput: f64,
    /// Time since last flush operation
    pub time_since_last_flush: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_threshold_growth_and_shrinkage() {
        let threshold = AdaptiveThreshold::new(2, 1, 10, 0.1);
        assert_eq!(threshold.current(), 2);

        std::thread::sleep(Duration::from_millis(110));
        threshold.update(150.0, Duration::from_micros(50));
        assert_eq!(threshold.current(), 2);

        std::thread::sleep(Duration::from_millis(110));
        threshold.update(200.0, Duration::from_micros(50));
        assert_eq!(threshold.current(), 3);

        std::thread::sleep(Duration::from_millis(110));
        threshold.update(300.0, Duration::from_micros(50));
        assert_eq!(threshold.current(), 4);

        std::thread::sleep(Duration::from_millis(110));
        threshold.update(10.0, Duration::from_micros(50));
        assert_eq!(threshold.current(), 3);
    }
}
