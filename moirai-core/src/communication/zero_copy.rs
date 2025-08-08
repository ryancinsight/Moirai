//! Zero-copy communication channels (core SSOT)
//!
//! This module contains zero-copy primitives migrated from the transport crate
//! to enforce a Single Source of Truth (SSOT) within `moirai-core`.

use std::sync::atomic::{AtomicUsize, AtomicBool, AtomicPtr, Ordering};
use std::sync::{Arc, RwLock};
use std::ptr;
use std::mem;
use std::time::{Duration, Instant};
use std::collections::{VecDeque, HashMap};

/// Error types for zero-copy operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZeroCopyError {
    /// Channel is full
    Full,
    /// Channel is empty
    Empty,
    /// Channel is closed
    Closed,
    /// Operation would block
    WouldBlock,
    /// Memory mapping failed
    MemoryMapFailed,
    /// Invalid buffer size
    InvalidBufferSize,
    /// Alignment error
    AlignmentError,
    /// No route found for domain
    NoRoute,
}

impl std::fmt::Display for ZeroCopyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "Zero-copy channel is full"),
            Self::Empty => write!(f, "Zero-copy channel is empty"),
            Self::Closed => write!(f, "Zero-copy channel is closed"),
            Self::WouldBlock => write!(f, "Zero-copy operation would block"),
            Self::MemoryMapFailed => write!(f, "Memory mapping failed"),
            Self::InvalidBufferSize => write!(f, "Invalid buffer size"),
            Self::AlignmentError => write!(f, "Memory alignment error"),
            Self::NoRoute => write!(f, "No route found for domain"),
        }
    }
}

impl std::error::Error for ZeroCopyError {}

/// Result type for zero-copy operations.
pub type ZeroCopyResult<T> = Result<T, ZeroCopyError>;

/// Memory-mapped ring buffer for zero-copy operations.
///
/// Safety: uses raw allocation for performance while maintaining
/// acquire/release ordering and bounds checks.
pub struct MemoryMappedRing<T> {
    buffer: AtomicPtr<T>,
    capacity: usize,
    producer_cursor: AtomicUsize,
    consumer_cursor: AtomicUsize,
    buffer_size: usize,
    _element_size: usize,
    closed: AtomicBool,
}

impl<T> MemoryMappedRing<T> {
    pub fn new(capacity: usize) -> ZeroCopyResult<Self> {
        if !capacity.is_power_of_two() || capacity == 0 {
            return Err(ZeroCopyError::InvalidBufferSize);
        }

        let element_size = mem::size_of::<T>();
        let alignment = mem::align_of::<T>();
        let buffer_size = capacity.checked_mul(element_size).ok_or(ZeroCopyError::InvalidBufferSize)?;

        let layout = std::alloc::Layout::from_size_align(buffer_size, alignment)
            .map_err(|_| ZeroCopyError::AlignmentError)?;

        let buffer = unsafe {
            let ptr = std::alloc::alloc(layout) as *mut T;
            if ptr.is_null() { return Err(ZeroCopyError::MemoryMapFailed); }
            ptr
        };

        Ok(Self {
            buffer: AtomicPtr::new(buffer),
            capacity,
            producer_cursor: AtomicUsize::new(0),
            consumer_cursor: AtomicUsize::new(0),
            buffer_size,
            _element_size: element_size,
            closed: AtomicBool::new(false),
        })
    }

    pub fn send_zero_copy(&self, value: T) -> Result<(), (T, ZeroCopyError)> {
        if self.closed.load(Ordering::Acquire) {
            return Err((value, ZeroCopyError::Closed));
        }
        let p = self.producer_cursor.load(Ordering::Relaxed);
        let c = self.consumer_cursor.load(Ordering::Acquire);
        if p.wrapping_sub(c) >= self.capacity { return Err((value, ZeroCopyError::Full)); }

        let ptr = self.buffer.load(Ordering::Relaxed);
        let idx = p & (self.capacity - 1);
        unsafe { ptr::write(ptr.add(idx), value); }
        self.producer_cursor.store(p.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    pub fn recv_zero_copy(&self) -> ZeroCopyResult<T> {
        let c = self.consumer_cursor.load(Ordering::Relaxed);
        let p = self.producer_cursor.load(Ordering::Acquire);
        if c == p {
            return if self.closed.load(Ordering::Acquire) {
                Err(ZeroCopyError::Closed)
            } else {
                Err(ZeroCopyError::Empty)
            };
        }
        let ptr = self.buffer.load(Ordering::Relaxed);
        let idx = c & (self.capacity - 1);
        let value = unsafe { ptr::read(ptr.add(idx)) };
        self.consumer_cursor.store(c.wrapping_add(1), Ordering::Release);
        Ok(value)
    }

    pub fn try_send(&self, value: T) -> Result<(), (T, ZeroCopyError)> { self.send_zero_copy(value) }
    pub fn try_recv(&self) -> ZeroCopyResult<T> { self.recv_zero_copy() }
    pub fn close(&self) { self.closed.store(true, Ordering::Release); }
    pub fn is_closed(&self) -> bool { self.closed.load(Ordering::Acquire) }
    pub fn len(&self) -> usize {
        let p = self.producer_cursor.load(Ordering::Relaxed);
        let c = self.consumer_cursor.load(Ordering::Relaxed);
        p.wrapping_sub(c)
    }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn is_full(&self) -> bool { self.len() >= self.capacity }
    pub fn capacity(&self) -> usize { self.capacity }
}

impl<T> Drop for MemoryMappedRing<T> {
    fn drop(&mut self) {
        let ptr = self.buffer.load(Ordering::Relaxed);
        if !ptr.is_null() {
            let layout = std::alloc::Layout::from_size_align(self.buffer_size, mem::align_of::<T>()).unwrap();
            unsafe {
                let c = self.consumer_cursor.load(Ordering::Relaxed);
                let p = self.producer_cursor.load(Ordering::Relaxed);
                for pos in c..p {
                    let idx = pos & (self.capacity - 1);
                    ptr::drop_in_place(ptr.add(idx));
                }
                std::alloc::dealloc(ptr as *mut u8, layout);
            }
        }
    }
}

unsafe impl<T: Send> Send for MemoryMappedRing<T> {}
unsafe impl<T: Send> Sync for MemoryMappedRing<T> {}

/// Zero-copy channel implemented over MemoryMappedRing.
pub struct ZeroCopyChannel<T> {
    _ring: Arc<MemoryMappedRing<T>>,
}

impl<T> ZeroCopyChannel<T> {
    pub fn new(capacity: usize) -> ZeroCopyResult<(ZeroCopySender<T>, ZeroCopyReceiver<T>)> {
        let ring = Arc::new(MemoryMappedRing::new(capacity)?);
        Ok((ZeroCopySender { ring: ring.clone() }, ZeroCopyReceiver { ring }))
    }
}

pub struct ZeroCopySender<T> { ring: Arc<MemoryMappedRing<T>> }
impl<T> ZeroCopySender<T> {
    pub fn send(&self, value: T) -> Result<(), (T, ZeroCopyError)> { self.ring.send_zero_copy(value) }
    pub fn try_send(&self, value: T) -> Result<(), (T, ZeroCopyError)> { self.ring.try_send(value) }
    pub fn close(&self) { self.ring.close(); }
    pub fn is_closed(&self) -> bool { self.ring.is_closed() }
}
impl<T> Clone for ZeroCopySender<T> { fn clone(&self) -> Self { Self { ring: self.ring.clone() } } }

pub struct ZeroCopyReceiver<T> { ring: Arc<MemoryMappedRing<T>> }
impl<T> ZeroCopyReceiver<T> {
    pub fn recv(&self) -> ZeroCopyResult<T> { self.ring.recv_zero_copy() }
    pub fn try_recv(&self) -> ZeroCopyResult<T> { self.ring.try_recv() }
    pub fn is_closed(&self) -> bool { self.ring.is_closed() }
}
impl<T> Clone for ZeroCopyReceiver<T> { fn clone(&self) -> Self { Self { ring: self.ring.clone() } } }

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
    pub fn current(&self) -> usize { self.current.load(Ordering::Relaxed) }
    pub fn update(&self, throughput: f64, latency: Duration) {
        let mut history = self.throughput_history.lock().unwrap();
        let mut last = self.last_adaptation.lock().unwrap();
        if last.elapsed() < Duration::from_millis(100) { return; }
        history.push_back(throughput);
        if history.len() > 10 { history.pop_front(); }
        let avg = if history.is_empty() { throughput } else { history.iter().sum::<f64>() / history.len() as f64 };
        let cur = self.current() as f64;
        let mut new_threshold = if throughput > avg * 1.1 {
            if latency < Duration::from_micros(100) { cur * (1.0 + self.adaptation_rate) } else { cur }
        } else if throughput < avg * 0.9 { cur * (1.0 - self.adaptation_rate) } else { cur };
        if new_threshold < self.min_threshold as f64 { new_threshold = self.min_threshold as f64; }
        if new_threshold > self.max_threshold as f64 { new_threshold = self.max_threshold as f64; }
        self.current.store(new_threshold as usize, Ordering::Relaxed);
        *last = Instant::now();
    }
}
impl Default for AdaptiveThreshold { fn default() -> Self { Self::new(32, 1, 1024, 0.1) } }

/// Throughput monitor for adaptive batching.
#[derive(Debug)]
pub struct ThroughputMonitor {
    message_count: AtomicUsize,
    start_time: std::sync::Mutex<Instant>,
    last_measurement: std::sync::Mutex<Instant>,
    recent_throughput: std::sync::Mutex<VecDeque<f64>>,
}

impl ThroughputMonitor {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            message_count: AtomicUsize::new(0),
            start_time: std::sync::Mutex::new(now),
            last_measurement: std::sync::Mutex::new(now),
            recent_throughput: std::sync::Mutex::new(VecDeque::with_capacity(10)),
        }
    }
    pub fn record_message(&self) { self.message_count.fetch_add(1, Ordering::Relaxed); }
    pub fn current_throughput(&self) -> f64 {
        let count = self.message_count.load(Ordering::Relaxed);
        let start = self.start_time.lock().unwrap();
        let elapsed = start.elapsed();
        if elapsed.as_secs_f64() > 0.0 { count as f64 / elapsed.as_secs_f64() } else { 0.0 }
    }
    pub fn recent_throughput(&self) -> f64 {
        let t = self.recent_throughput.lock().unwrap();
        if t.is_empty() { 0.0 } else { t.iter().sum::<f64>() / t.len() as f64 }
    }
    pub fn update(&self) {
        let mut last = self.last_measurement.lock().unwrap();
        let mut rt = self.recent_throughput.lock().unwrap();
        let now = Instant::now();
        if now.duration_since(*last) >= Duration::from_millis(100) {
            rt.push_back(self.current_throughput());
            if rt.len() > 10 { rt.pop_front(); }
            *last = now;
        }
    }
    pub fn idle_time(&self) -> Duration { self.last_measurement.lock().unwrap().elapsed() }
}
impl Default for ThroughputMonitor { fn default() -> Self { Self::new() } }

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
    pub fn new(capacity: usize, max_batch_delay: Duration) -> ZeroCopyResult<(AdaptiveBatchSender<T>, AdaptiveBatchReceiver<T>)> {
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

pub struct AdaptiveBatchSender<T> {
    sender: ZeroCopySender<T>,
    batch_buffer: std::sync::Mutex<VecDeque<T>>,
    adaptive_threshold: AdaptiveThreshold,
    throughput_monitor: ThroughputMonitor,
    max_batch_delay: Duration,
    last_flush: std::sync::Mutex<Instant>,
}

impl<T> AdaptiveBatchSender<T> {
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
        len >= self.adaptive_threshold.current() || self.last_flush.lock().unwrap().elapsed() > self.max_batch_delay
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
                                for x in pending.into_iter().rev() { buf.push_front(x); }
                                return Err(ZeroCopyError::Closed);
                            }
                            ZeroCopyError::Full | ZeroCopyError::WouldBlock => {
                                // Put back current item at the front and retry after yielding
                                pending.push_front(v);
                                break;
                            }
                            other => {
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
                continue;
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
    pub fn flush(&self) -> ZeroCopyResult<()> { self.flush_batch() }
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

pub struct AdaptiveBatchReceiver<T> { receiver: ZeroCopyReceiver<T> }
impl<T> AdaptiveBatchReceiver<T> {
    pub fn recv(&self) -> ZeroCopyResult<T> { self.receiver.recv() }
    pub fn try_recv(&self) -> ZeroCopyResult<T> { self.receiver.try_recv() }
}

/// Statistics for adaptive batching.
#[derive(Debug, Clone)]
pub struct BatchStats {
    pub current_threshold: usize,
    pub pending_messages: usize,
    pub current_throughput: f64,
    pub recent_throughput: f64,
    pub time_since_last_flush: Duration,
}

/// Zero-copy message router with domain-based routing.
pub struct ZeroCopyRouter<T> {
    routes: Arc<RwLock<HashMap<DomainId, Arc<ZeroCopySender<T>>>>>,
    default_route: Option<Arc<ZeroCopySender<T>>>,
    stats: RouterStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DomainId(u64);
impl DomainId { pub const SYNC: Self = DomainId(0); pub const ASYNC: Self = DomainId(1); pub const PARALLEL: Self = DomainId(2); pub const DISTRIBUTED: Self = DomainId(3); pub fn new(id: u64) -> Self { DomainId(id) } }

#[derive(Debug, Default)]
struct RouterStats {
    messages_routed: AtomicUsize,
    routing_failures: AtomicUsize,
    zero_copy_sends: AtomicUsize,
}

impl<T: Send + 'static> ZeroCopyRouter<T> {
    pub fn new() -> Self { Self { routes: Arc::new(RwLock::new(HashMap::new())), default_route: None, stats: RouterStats::default() } }
    pub fn add_route(&self, domain: DomainId, capacity: usize) -> ZeroCopyResult<ZeroCopyReceiver<T>> {
        let (s, r) = ZeroCopyChannel::new(capacity)?;
        self.routes.write().unwrap().insert(domain, Arc::new(s));
        Ok(r)
    }
    pub fn set_default_route(&mut self, capacity: usize) -> ZeroCopyResult<ZeroCopyReceiver<T>> {
        let (s, r) = ZeroCopyChannel::new(capacity)?; self.default_route = Some(Arc::new(s)); Ok(r)
    }
    pub fn route(&self, domain: DomainId, message: T) -> Result<(), (T, ZeroCopyError)> {
        self.stats.messages_routed.fetch_add(1, Ordering::Relaxed);
        if let Some(ch) = self.routes.read().unwrap().get(&domain) {
            match ch.send(message) {
                Ok(()) => { self.stats.zero_copy_sends.fetch_add(1, Ordering::Relaxed); Ok(()) }
                Err((msg, e)) => { self.stats.routing_failures.fetch_add(1, Ordering::Relaxed); Err((msg, e)) }
            }
        } else if let Some(def) = &self.default_route { def.send(message) } else { self.stats.routing_failures.fetch_add(1, Ordering::Relaxed); Err((message, ZeroCopyError::NoRoute)) }
    }
    pub fn stats(&self) -> (usize, usize, usize) { (self.stats.messages_routed.load(Ordering::Relaxed), self.stats.routing_failures.load(Ordering::Relaxed), self.stats.zero_copy_sends.load(Ordering::Relaxed)) }
}