//! High-performance communication infrastructure for concurrent components.
//! 
//! This module provides efficient communication primitives inspired by:
//! - Crossbeam's lock-free channels
//! - MPI's collective operations
//! - LMAX Disruptor's ring buffers
//! - Zero-copy message passing

use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

/// Cache line size for padding
const CACHE_LINE: usize = 64;

/// Padding to prevent false sharing
#[repr(align(64))]
struct CachePadded<T> {
    value: T,
}

/// Zero-copy message for efficient communication
pub struct Message<T> {
    /// The actual data
    data: T,
    /// Reference count for shared ownership
    refcount: Arc<AtomicUsize>,
}

impl<T> Message<T> {
    /// Create a new message
    pub fn new(data: T) -> Self {
        Self {
            data,
            refcount: Arc::new(AtomicUsize::new(1)),
        }
    }
    
    /// Get a reference to the data
    pub fn data(&self) -> &T {
        &self.data
    }
    
    /// Take ownership of the data if this is the only reference
    pub fn try_unwrap(self) -> Result<T, Self> {
        if self.refcount.load(Ordering::Acquire) == 1 {
            Ok(self.data)
        } else {
            Err(self)
        }
    }
}

impl<T: Clone> Clone for Message<T> {
    fn clone(&self) -> Self {
        self.refcount.fetch_add(1, Ordering::Relaxed);
        Self {
            data: self.data.clone(),
            refcount: self.refcount.clone(),
        }
    }
}

/// Lock-free SPSC (Single Producer Single Consumer) channel
/// Optimized for low latency and high throughput
pub struct SpscChannel<T> {
    /// Ring buffer for messages
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Capacity mask for fast modulo
    mask: usize,
    /// Producer position
    head: CachePadded<AtomicUsize>,
    /// Consumer position
    tail: CachePadded<AtomicUsize>,
    /// Channel state
    closed: AtomicBool,
}

unsafe impl<T: Send> Send for SpscChannel<T> {}
unsafe impl<T: Send> Sync for SpscChannel<T> {}

impl<T> SpscChannel<T> {
    /// Create a new SPSC channel with given capacity
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let buffer = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
            
        Self {
            buffer,
            mask: capacity - 1,
            head: CachePadded { value: AtomicUsize::new(0) },
            tail: CachePadded { value: AtomicUsize::new(0) },
            closed: AtomicBool::new(false),
        }
    }
    
    /// Send a message (producer side)
    pub fn send(&self, value: T) -> Result<(), T> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(value);
        }
        
        let head = self.head.value.load(Ordering::Relaxed);
        let tail = self.tail.value.load(Ordering::Acquire);
        
        // Check if full
        if head.wrapping_sub(tail) >= self.buffer.len() {
            return Err(value);
        }
        
        unsafe {
            let slot = &mut *self.buffer[head & self.mask].get();
            slot.write(value);
        }
        
        self.head.value.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }
    
    /// Receive a message (consumer side)
    pub fn recv(&self) -> Option<T> {
        let tail = self.tail.value.load(Ordering::Relaxed);
        let head = self.head.value.load(Ordering::Acquire);
        
        if tail == head {
            return None;
        }
        
        let value = unsafe {
            let slot = &*self.buffer[tail & self.mask].get();
            slot.assume_init_read()
        };
        
        self.tail.value.store(tail.wrapping_add(1), Ordering::Release);
        Some(value)
    }
    
    /// Close the channel
    pub fn close(&self) {
        self.closed.store(true, Ordering::Relaxed);
    }
}

/// Multi-producer multi-consumer broadcast channel
/// Allows efficient one-to-many communication
pub struct BroadcastChannel<T: Clone> {
    /// Current value
    value: Arc<RwLock<Option<T>>>,
    /// Version number for change detection
    version: Arc<AtomicUsize>,
    /// Subscriber count
    subscribers: Arc<AtomicUsize>,
}

/// Read-write lock optimized for many readers
struct RwLock<T> {
    data: UnsafeCell<T>,
    readers: AtomicUsize,
    writer: AtomicBool,
}

unsafe impl<T: Send> Send for RwLock<T> {}
unsafe impl<T: Send> Sync for RwLock<T> {}

impl<T> RwLock<T> {
    fn new(data: T) -> Self {
        Self {
            data: UnsafeCell::new(data),
            readers: AtomicUsize::new(0),
            writer: AtomicBool::new(false),
        }
    }
    
    fn read(&self) -> RwLockReadGuard<'_, T> {
        loop {
            self.readers.fetch_add(1, Ordering::Acquire);
            
            if !self.writer.load(Ordering::Acquire) {
                return RwLockReadGuard { lock: self };
            }
            
            self.readers.fetch_sub(1, Ordering::Release);
            while self.writer.load(Ordering::Relaxed) {
                std::hint::spin_loop();
            }
        }
    }
    
    fn write(&self) -> RwLockWriteGuard<'_, T> {
        while self.writer.compare_exchange_weak(
            false,
            true,
            Ordering::Acquire,
            Ordering::Relaxed
        ).is_err() {
            std::hint::spin_loop();
        }
        
        while self.readers.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        
        RwLockWriteGuard { lock: self }
    }
}

struct RwLockReadGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<'a, T> Drop for RwLockReadGuard<'a, T> {
    fn drop(&mut self) {
        self.lock.readers.fetch_sub(1, Ordering::Release);
    }
}

impl<'a, T> std::ops::Deref for RwLockReadGuard<'a, T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.data.get() }
    }
}

struct RwLockWriteGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<'a, T> Drop for RwLockWriteGuard<'a, T> {
    fn drop(&mut self) {
        self.lock.writer.store(false, Ordering::Release);
    }
}

impl<'a, T> std::ops::Deref for RwLockWriteGuard<'a, T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T> std::ops::DerefMut for RwLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T: Clone> BroadcastChannel<T> {
    /// Create a new broadcast channel
    pub fn new() -> Self {
        Self {
            value: Arc::new(RwLock::new(None)),
            version: Arc::new(AtomicUsize::new(0)),
            subscribers: Arc::new(AtomicUsize::new(0)),
        }
    }
    
    /// Broadcast a new value to all subscribers
    pub fn broadcast(&self, value: T) {
        let mut guard = self.value.write();
        *guard = Some(value);
        self.version.fetch_add(1, Ordering::Release);
    }
    
    /// Subscribe to broadcasts
    pub fn subscribe(&self) -> BroadcastReceiver<T> {
        self.subscribers.fetch_add(1, Ordering::Relaxed);
        BroadcastReceiver {
            channel: self.clone(),
            last_version: 0,
        }
    }
}

impl<T: Clone> Clone for BroadcastChannel<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            version: self.version.clone(),
            subscribers: self.subscribers.clone(),
        }
    }
}

/// Receiver for broadcast channel
pub struct BroadcastReceiver<T: Clone> {
    channel: BroadcastChannel<T>,
    last_version: usize,
}

impl<T: Clone> BroadcastReceiver<T> {
    /// Try to receive the latest broadcast value
    pub fn try_recv(&mut self) -> Option<T> {
        let current_version = self.channel.version.load(Ordering::Acquire);
        
        if current_version > self.last_version {
            self.last_version = current_version;
            let guard = self.channel.value.read();
            guard.clone()
        } else {
            None
        }
    }
}

impl<T: Clone> Drop for BroadcastReceiver<T> {
    fn drop(&mut self) {
        self.channel.subscribers.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Efficient collective operations for group communication
pub struct CollectiveOps;

impl CollectiveOps {
    /// All-reduce operation: combine values from all participants
    pub fn all_reduce<T, F>(values: Vec<T>, op: F) -> Vec<T>
    where
        T: Clone + Send,
        F: Fn(T, T) -> T + Sync,
    {
        if values.is_empty() {
            return vec![];
        }
        
        let result_len = values.len();
        
        // Tree reduction for efficiency
        let mut current = values;
        while current.len() > 1 {
            let mut next = Vec::with_capacity((current.len() + 1) / 2);
            
            for chunk in current.chunks(2) {
                if chunk.len() == 2 {
                    next.push(op(chunk[0].clone(), chunk[1].clone()));
                } else {
                    next.push(chunk[0].clone());
                }
            }
            
            current = next;
        }
        
        // Broadcast result to all
        vec![current[0].clone(); result_len]
    }
    
    /// Scatter operation: distribute data chunks to participants
    pub fn scatter<T: Clone>(data: Vec<T>, num_participants: usize) -> Vec<Vec<T>> {
        let chunk_size = (data.len() + num_participants - 1) / num_participants;
        data.chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
    
    /// Gather operation: collect data from all participants
    pub fn gather<T>(chunks: Vec<Vec<T>>) -> Vec<T> {
        chunks.into_iter().flatten().collect()
    }
    
    /// All-to-all communication pattern
    pub fn all_to_all<T: Clone>(data: Vec<Vec<T>>) -> Vec<Vec<T>> {
        let n = data.len();
        let mut result = vec![Vec::new(); n];
        
        for (_i, row) in data.iter().enumerate() {
            for (j, item) in row.iter().enumerate() {
                if j < n {
                    result[j].push(item.clone());
                }
            }
        }
        
        result
    }
}

/// Zero-copy ring buffer for high-throughput communication
pub struct RingBuffer<T> {
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    capacity: usize,
    mask: usize,
    /// Producer sequence number
    producer_seq: CachePadded<AtomicUsize>,
    /// Consumer sequence number
    consumer_seq: CachePadded<AtomicUsize>,
    /// Cached consumer position (producer's view)
    cached_consumer: CachePadded<UnsafeCell<usize>>,
    /// Cached producer position (consumer's view)
    cached_producer: CachePadded<UnsafeCell<usize>>,
}

unsafe impl<T: Send> Send for RingBuffer<T> {}
unsafe impl<T: Send> Sync for RingBuffer<T> {}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let buffer = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
            
        Self {
            buffer,
            capacity,
            mask: capacity - 1,
            producer_seq: CachePadded { value: AtomicUsize::new(0) },
            consumer_seq: CachePadded { value: AtomicUsize::new(0) },
            cached_consumer: CachePadded { value: UnsafeCell::new(0) },
            cached_producer: CachePadded { value: UnsafeCell::new(0) },
        }
    }
    
    /// Try to publish a value
    pub fn try_publish(&self, value: T) -> Result<(), T> {
        let current = self.producer_seq.value.load(Ordering::Relaxed);
        let cached_consumer = unsafe { *self.cached_consumer.value.get() };
        
        // Check if there's space
        if current - cached_consumer >= self.capacity {
            // Update cached consumer position
            let actual_consumer = self.consumer_seq.value.load(Ordering::Acquire);
            unsafe { *self.cached_consumer.value.get() = actual_consumer; }
            
            if current - actual_consumer >= self.capacity {
                return Err(value);
            }
        }
        
        // Write value
        unsafe {
            let slot = &mut *self.buffer[current & self.mask].get();
            slot.write(value);
        }
        
        // Publish
        self.producer_seq.value.store(current + 1, Ordering::Release);
        Ok(())
    }
    
    /// Try to consume a value
    pub fn try_consume(&self) -> Option<T> {
        let current = self.consumer_seq.value.load(Ordering::Relaxed);
        let cached_producer = unsafe { *self.cached_producer.value.get() };
        
        // Check if there's data
        if current >= cached_producer {
            // Update cached producer position
            let actual_producer = self.producer_seq.value.load(Ordering::Acquire);
            unsafe { *self.cached_producer.value.get() = actual_producer; }
            
            if current >= actual_producer {
                return None;
            }
        }
        
        // Read value
        let value = unsafe {
            let slot = &*self.buffer[current & self.mask].get();
            slot.assume_init_read()
        };
        
        // Consume
        self.consumer_seq.value.store(current + 1, Ordering::Release);
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spsc_channel() {
        let channel = SpscChannel::new(4);
        
        // Send some values
        assert!(channel.send(1).is_ok());
        assert!(channel.send(2).is_ok());
        assert!(channel.send(3).is_ok());
        
        // Receive values
        assert_eq!(channel.recv(), Some(1));
        assert_eq!(channel.recv(), Some(2));
        assert_eq!(channel.recv(), Some(3));
        assert_eq!(channel.recv(), None);
    }
    
    #[test]
    fn test_broadcast_channel() {
        let channel = BroadcastChannel::new();
        let mut receiver1 = channel.subscribe();
        let mut receiver2 = channel.subscribe();
        
        // Broadcast a value
        channel.broadcast(42);
        
        // Both receivers should get it
        assert_eq!(receiver1.try_recv(), Some(42));
        assert_eq!(receiver2.try_recv(), Some(42));
        
        // No new values
        assert_eq!(receiver1.try_recv(), None);
        assert_eq!(receiver2.try_recv(), None);
    }
    
    #[test]
    fn test_ring_buffer() {
        let buffer = RingBuffer::new(4);
        
        // Publish values
        assert!(buffer.try_publish(1).is_ok());
        assert!(buffer.try_publish(2).is_ok());
        
        // Consume values
        assert_eq!(buffer.try_consume(), Some(1));
        assert_eq!(buffer.try_consume(), Some(2));
        assert_eq!(buffer.try_consume(), None);
    }
    
    #[test]
    fn test_collective_ops() {
        // Test all_reduce
        let values = vec![1, 2, 3, 4];
        let result = CollectiveOps::all_reduce(values, |a, b| a + b);
        assert_eq!(result, vec![10, 10, 10, 10]);
        
        // Test scatter
        let data = vec![1, 2, 3, 4, 5, 6];
        let scattered = CollectiveOps::scatter(data, 3);
        assert_eq!(scattered.len(), 3);
        assert_eq!(scattered[0], vec![1, 2]);
        assert_eq!(scattered[1], vec![3, 4]);
        assert_eq!(scattered[2], vec![5, 6]);
        
        // Test gather
        let chunks = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let gathered = CollectiveOps::gather(chunks);
        assert_eq!(gathered, vec![1, 2, 3, 4, 5, 6]);
    }
}