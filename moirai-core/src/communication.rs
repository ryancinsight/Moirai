//! High-performance communication patterns for concurrent systems.
//! 
//! This module provides advanced communication mechanisms that build on top
//! of the unified channel implementations:
//! - Broadcast channels for one-to-many communication
//! - Collective operations for group communication
//! - Ring buffers for zero-copy streaming
//! - Message routing and pub/sub patterns
//!
//! # Design Principles
//! - Build on top of core channel primitives (DRY principle)
//! - Provide higher-level abstractions for complex patterns
//! - Maintain zero-copy semantics where possible
//! - Follow SOLID principles with focused interfaces

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::collections::HashMap;
use std::hash::Hash;

use crate::channel::{MpmcSender, MpmcReceiver, mpmc, ChannelError};

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

/// Broadcast channel for one-to-many communication
/// Multiple receivers get copies of each message
pub struct BroadcastChannel<T: Clone> {
    /// Current value (protected by RwLock for concurrent access)
    value: Arc<RwLock<Option<T>>>,
    /// Version number for detecting updates
    version: Arc<AtomicUsize>,
    /// Number of active subscribers
    subscribers: Arc<AtomicUsize>,
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
    
    /// Broadcast a value to all subscribers
    pub fn broadcast(&self, value: T) {
        {
            let mut guard = self.value.write().unwrap();
            *guard = Some(value);
        }
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
    
    /// Get the current number of subscribers
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.load(Ordering::Relaxed)
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
            let guard = self.channel.value.read().unwrap();
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
    
    /// Zero-copy scatter operation using slices
    pub fn scatter_zero_copy<T>(data: &[T], num_chunks: usize) -> Vec<&[T]> {
        let chunk_size = data.len() / num_chunks;
        let mut chunks = Vec::with_capacity(num_chunks);
        
        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = if i == num_chunks - 1 {
                data.len()
            } else {
                (i + 1) * chunk_size
            };
            chunks.push(&data[start..end]);
        }
        
        chunks
    }
    
    /// Zero-copy gather operation using iterators
    pub fn gather_zero_copy<'a, T, I>(chunks: I) -> impl Iterator<Item = &'a T>
    where
        I: IntoIterator<Item = &'a [T]>,
        T: 'a,
    {
        chunks.into_iter().flat_map(|chunk| chunk.iter())
    }
    
    /// Zero-copy all-reduce operation
    pub fn all_reduce_zero_copy<T, F>(data: &[T], op: F) -> T
    where
        T: Clone,
        F: Fn(&T, &T) -> T,
    {
        data.iter()
            .skip(1)
            .fold(data[0].clone(), |acc, item| op(&acc, item))
    }
}

/// Zero-copy ring buffer for high-throughput streaming
/// 
/// # Safety
/// 
/// This structure uses `MaybeUninit` for zero-copy performance:
/// - Values are written with `write()` before incrementing producer_seq
/// - The `assume_init_read()` in `try_consume()` is safe because we check
///   that producer_seq > current, ensuring data was written
pub struct RingBuffer<T> {
    /// Buffer storage
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Capacity mask for fast modulo
    mask: usize,
    /// Producer sequence number
    producer_seq: CachePadded<AtomicUsize>,
    /// Consumer sequence number
    consumer_seq: CachePadded<AtomicUsize>,
}

unsafe impl<T: Send> Send for RingBuffer<T> {}
unsafe impl<T: Send> Sync for RingBuffer<T> {}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let buffer = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
            
        Self {
            buffer,
            mask: capacity - 1,
            producer_seq: CachePadded { value: AtomicUsize::new(0) },
            consumer_seq: CachePadded { value: AtomicUsize::new(0) },
        }
    }
    
    /// Try to produce a value
    pub fn try_produce(&self, value: T) -> Result<(), T> {
        let current = self.producer_seq.value.load(Ordering::Relaxed);
        let consumer = self.consumer_seq.value.load(Ordering::Acquire);
        
        // Check if full
        if current.wrapping_sub(consumer) >= self.buffer.len() {
            return Err(value);
        }
        
        unsafe {
            let slot = &mut *self.buffer[current & self.mask].get();
            slot.write(value);
        }
        
        self.producer_seq.value.store(current.wrapping_add(1), Ordering::Release);
        Ok(())
    }
    
    /// Try to consume a value
    pub fn try_consume(&self) -> Option<T> {
        let current = self.consumer_seq.value.load(Ordering::Relaxed);
        let producer = self.producer_seq.value.load(Ordering::Acquire);
        
        if current == producer {
            return None;
        }
        
        let value = unsafe {
            let slot = &*self.buffer[current & self.mask].get();
            // SAFETY: producer > current check ensures this slot has data
            slot.assume_init_read()
        };
        
        self.consumer_seq.value.store(current.wrapping_add(1), Ordering::Release);
        Some(value)
    }
}

/// Topic-based publish/subscribe system built on channels
pub struct PubSub<K: Hash + Eq + Clone, V: Clone + Send + 'static> {
    /// Subscribers mapped by topic
    subscribers: Arc<RwLock<HashMap<K, Vec<MpmcSender<V>>>>>,
}

impl<K: Hash + Eq + Clone, V: Clone + Send + 'static> PubSub<K, V> {
    /// Create a new pub/sub system
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Subscribe to a topic
    pub fn subscribe(&self, topic: K) -> MpmcReceiver<V> {
        let (tx, rx) = mpmc(100);
        
        let mut subs = self.subscribers.write().unwrap();
        subs.entry(topic).or_insert_with(Vec::new).push(tx);
        
        rx
    }
    
    /// Publish a message to a topic
    pub fn publish(&self, topic: &K, message: V) -> Result<usize, ChannelError> {
        let subs = self.subscribers.read().unwrap();
        
        if let Some(subscribers) = subs.get(topic) {
            let mut sent = 0;
            for sub in subscribers {
                if sub.try_send(message.clone()).is_ok() {
                    sent += 1;
                }
            }
            Ok(sent)
        } else {
            Ok(0)
        }
    }
    
    /// Get the number of subscribers for a topic
    pub fn subscriber_count(&self, topic: &K) -> usize {
        let subs = self.subscribers.read().unwrap();
        subs.get(topic).map_or(0, |v| v.len())
    }
}

impl<K: Hash + Eq + Clone, V: Clone + Send + 'static> Clone for PubSub<K, V> {
    fn clone(&self) -> Self {
        Self {
            subscribers: self.subscribers.clone(),
        }
    }
}

/// Router for message-based communication patterns
pub struct MessageRouter<K: Hash + Eq + Clone, V: Send + 'static> {
    /// Routes mapped by key
    routes: Arc<RwLock<HashMap<K, MpmcSender<V>>>>,
}

impl<K: Hash + Eq + Clone, V: Send + 'static> MessageRouter<K, V> {
    /// Create a new message router
    pub fn new() -> Self {
        Self {
            routes: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Register a route
    pub fn register(&self, key: K, sender: MpmcSender<V>) {
        let mut routes = self.routes.write().unwrap();
        routes.insert(key, sender);
    }
    
    /// Route a message to the appropriate channel
    pub fn route(&self, key: &K, message: V) -> Result<(), ChannelError> {
        let routes = self.routes.read().unwrap();
        
        if let Some(sender) = routes.get(key) {
            sender.try_send(message)
        } else {
            Err(ChannelError::Closed)
        }
    }
    
    /// Remove a route
    pub fn unregister(&self, key: &K) -> bool {
        let mut routes = self.routes.write().unwrap();
        routes.remove(key).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_broadcast_channel() {
        let channel = BroadcastChannel::new();
        let mut rx1 = channel.subscribe();
        let mut rx2 = channel.subscribe();
        
        channel.broadcast(42);
        
        assert_eq!(rx1.try_recv(), Some(42));
        assert_eq!(rx2.try_recv(), Some(42));
        
        // No new broadcasts
        assert_eq!(rx1.try_recv(), None);
    }
    
    #[test]
    fn test_collective_ops() {
        let values = vec![1, 2, 3, 4];
        let result = CollectiveOps::all_reduce(values, |a, b| a + b);
        assert_eq!(result, vec![10, 10, 10, 10]);
        
        let data = vec![1, 2, 3, 4, 5, 6];
        let scattered = CollectiveOps::scatter(data, 3);
        assert_eq!(scattered.len(), 3);
        assert_eq!(scattered[0], vec![1, 2]);
        
        let gathered = CollectiveOps::gather(scattered);
        assert_eq!(gathered, vec![1, 2, 3, 4, 5, 6]);
    }
    
    #[test]
    fn test_ring_buffer() {
        let rb = RingBuffer::new(4);
        
        assert!(rb.try_produce(1).is_ok());
        assert!(rb.try_produce(2).is_ok());
        
        assert_eq!(rb.try_consume(), Some(1));
        assert_eq!(rb.try_consume(), Some(2));
        assert_eq!(rb.try_consume(), None);
    }
    
    #[test]
    fn test_pubsub() {
        let pubsub = PubSub::new();
        let rx = pubsub.subscribe("topic1");
        
        assert_eq!(pubsub.publish(&"topic1", 42).unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 42);
        
        assert_eq!(pubsub.publish(&"topic2", 99).unwrap(), 0);
    }
}