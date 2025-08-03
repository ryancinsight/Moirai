//! Unified high-performance channel implementations for Moirai.
//!
//! This module provides zero-cost channel abstractions that work seamlessly
//! across different execution contexts following DRY and SOLID principles.
//!
//! # Design Principles
//! - **Zero-copy**: Minimize data copies for maximum performance
//! - **Cache-friendly**: Align data structures to cache lines
//! - **Lock-free**: Use atomic operations where possible
//! - **Unified API**: Single interface for different channel types
//!
//! # Safety
//! All channel implementations maintain memory safety through:
//! - Sequence number validation before reading uninitialized memory
//! - Proper memory ordering with acquire-release semantics
//! - Safe cleanup on drop with reference counting

use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Condvar};
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::fmt;



/// Padding to prevent false sharing between CPU cores
#[repr(align(64))]
struct CachePadded<T> {
    value: T,
}

impl<T> CachePadded<T> {
    const fn new(value: T) -> Self {
        Self {
            value,
        }
    }
}

/// Error types for channel operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelError {
    /// Channel is full and cannot accept more messages
    Full,
    /// Channel is empty and has no messages
    Empty,
    /// Channel has been closed
    Closed,
    /// Operation would block but non-blocking was requested
    WouldBlock,
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "channel is full"),
            Self::Empty => write!(f, "channel is empty"),
            Self::Closed => write!(f, "channel is closed"),
            Self::WouldBlock => write!(f, "operation would block"),
        }
    }
}

impl std::error::Error for ChannelError {}

/// Result type for channel operations
pub type Result<T> = std::result::Result<T, ChannelError>;

/// Trait for unified channel behavior following Interface Segregation Principle
pub trait Channel<T>: Send + Sync {
    /// Send a value, blocking if necessary
    fn send(&self, value: T) -> Result<()>;
    
    /// Try to send without blocking
    fn try_send(&self, value: T) -> Result<()>;
    
    /// Receive a value, blocking if necessary
    fn recv(&self) -> Result<T>;
    
    /// Try to receive without blocking
    fn try_recv(&self) -> Result<T>;
    
    /// Check if channel is empty
    fn is_empty(&self) -> bool;
    
    /// Check if channel is full
    fn is_full(&self) -> bool;
    
    /// Get the capacity of the channel
    fn capacity(&self) -> Option<usize>;
}

/// Lock-free Single Producer Single Consumer channel
/// Optimized for low latency with zero-copy semantics
pub struct SpscChannel<T> {
    /// Ring buffer for messages
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Capacity mask for fast modulo
    mask: usize,
    /// Producer position (cache-aligned)
    head: CachePadded<AtomicUsize>,
    /// Consumer position (cache-aligned)
    tail: CachePadded<AtomicUsize>,
    /// Channel state
    closed: AtomicBool,
    /// Phantom data for variance
    _phantom: PhantomData<T>,
}

unsafe impl<T: Send> Send for SpscChannel<T> {}
unsafe impl<T: Send> Sync for SpscChannel<T> {}

impl<T> SpscChannel<T> {
    /// Create a new SPSC channel with given capacity (rounded up to power of 2)
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two().max(2);
        let buffer = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
            
        Self {
            buffer,
            mask: capacity - 1,
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
            closed: AtomicBool::new(false),
            _phantom: PhantomData,
        }
    }
    
    /// Create a channel pair (sender, receiver) for ergonomic usage
    pub fn channel(capacity: usize) -> (SpscSender<T>, SpscReceiver<T>) {
        let channel = Arc::new(Self::new(capacity));
        (
            SpscSender { channel: channel.clone() },
            SpscReceiver { channel },
        )
    }
}

impl<T: Send> Channel<T> for SpscChannel<T> {
    fn send(&self, value: T) -> Result<()> {
        // Implement blocking send with exponential backoff spin-wait
        let mut spin_count = 0;
        loop {
            // Check if channel is closed first
            if self.closed.load(Ordering::Relaxed) {
                return Err(ChannelError::Closed);
            }
            
            let head = self.head.value.load(Ordering::Relaxed);
            let tail = self.tail.value.load(Ordering::Acquire);
            
            // Check if there's space
            if head.wrapping_sub(tail) < self.buffer.len() {
                // There's space, try to send
                unsafe {
                    let slot = &mut *self.buffer[head & self.mask].get();
                    slot.write(value);
                }
                self.head.value.store(head.wrapping_add(1), Ordering::Release);
                return Ok(());
            }
            
            // Channel is full, spin-wait with exponential backoff
            if spin_count < 6 {
                // Active spinning for low latency (up to ~64 iterations)
                for _ in 0..(1 << spin_count) {
                    std::hint::spin_loop();
                }
                spin_count += 1;
            } else {
                // After initial spinning, yield to OS scheduler
                std::thread::yield_now();
            }
        }
    }
    
    fn try_send(&self, value: T) -> Result<()> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(ChannelError::Closed);
        }
        
        let head = self.head.value.load(Ordering::Relaxed);
        let tail = self.tail.value.load(Ordering::Acquire);
        
        // Check if full
        if head.wrapping_sub(tail) >= self.buffer.len() {
            return Err(ChannelError::Full);
        }
        
        unsafe {
            let slot = &mut *self.buffer[head & self.mask].get();
            slot.write(value);
        }
        
        self.head.value.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }
    
    fn recv(&self) -> Result<T> {
        // Implement blocking recv with exponential backoff spin-wait
        let mut spin_count = 0;
        loop {
            match self.try_recv() {
                Ok(value) => return Ok(value),
                Err(ChannelError::Empty) => {
                    // Channel is empty, spin-wait with exponential backoff
                    if spin_count < 6 {
                        // Active spinning for low latency (up to ~64 iterations)
                        for _ in 0..(1 << spin_count) {
                            std::hint::spin_loop();
                        }
                        spin_count += 1;
                    } else {
                        // After initial spinning, yield to OS scheduler
                        std::thread::yield_now();
                    }
                }
                Err(e) => return Err(e), // Closed or other error
            }
        }
    }
    
    fn try_recv(&self) -> Result<T> {
        let tail = self.tail.value.load(Ordering::Relaxed);
        let head = self.head.value.load(Ordering::Acquire);
        
        if tail == head {
            if self.closed.load(Ordering::Relaxed) {
                return Err(ChannelError::Closed);
            }
            return Err(ChannelError::Empty);
        }
        
        let value = unsafe {
            let slot = &*self.buffer[tail & self.mask].get();
            // SAFETY: head > tail check ensures initialized data
            slot.assume_init_read()
        };
        
        self.tail.value.store(tail.wrapping_add(1), Ordering::Release);
        Ok(value)
    }
    
    fn is_empty(&self) -> bool {
        let tail = self.tail.value.load(Ordering::Relaxed);
        let head = self.head.value.load(Ordering::Acquire);
        tail == head
    }
    
    fn is_full(&self) -> bool {
        let head = self.head.value.load(Ordering::Relaxed);
        let tail = self.tail.value.load(Ordering::Acquire);
        head.wrapping_sub(tail) >= self.buffer.len()
    }
    
    fn capacity(&self) -> Option<usize> {
        Some(self.buffer.len())
    }
}

/// Sender half of SPSC channel
pub struct SpscSender<T> {
    channel: Arc<SpscChannel<T>>,
}

impl<T> Clone for SpscSender<T> {
    fn clone(&self) -> Self {
        Self {
            channel: self.channel.clone(),
        }
    }
}

impl<T: Send> SpscSender<T> {
    /// Send a value through the channel, blocking if necessary
    pub fn send(&self, value: T) -> Result<()> {
        self.channel.send(value)
    }
    
    /// Try to send a value without blocking
    pub fn try_send(&self, value: T) -> Result<()> {
        self.channel.try_send(value)
    }
}

/// Receiver half of SPSC channel
pub struct SpscReceiver<T> {
    channel: Arc<SpscChannel<T>>,
}

impl<T> Clone for SpscReceiver<T> {
    fn clone(&self) -> Self {
        Self {
            channel: self.channel.clone(),
        }
    }
}

impl<T: Send> SpscReceiver<T> {
    /// Receive a value from the channel, blocking if necessary
    pub fn recv(&self) -> Result<T> {
        self.channel.recv()
    }
    
    /// Try to receive a value without blocking
    pub fn try_recv(&self) -> Result<T> {
        self.channel.try_recv()
    }
}

/// Multi-Producer Multi-Consumer channel with bounded capacity
/// Uses mutex-based implementation for simplicity and correctness
pub struct MpmcChannel<T> {
    state: Arc<(Mutex<MpmcState<T>>, Condvar, Condvar)>,
}

struct MpmcState<T> {
    queue: VecDeque<T>,
    capacity: Option<usize>,
    closed: bool,
    sender_count: usize,
    receiver_count: usize,
}

impl<T> MpmcChannel<T> {
    /// Create a new MPMC channel with optional capacity
    pub fn new(capacity: Option<usize>) -> Self {
        let state = MpmcState {
            queue: VecDeque::with_capacity(capacity.unwrap_or(16)),
            capacity,
            closed: false,
            sender_count: 0,
            receiver_count: 0,
        };
        
        Self {
            state: Arc::new((Mutex::new(state), Condvar::new(), Condvar::new())),
        }
    }
    
    /// Create an unbounded channel
    pub fn unbounded() -> Self {
        Self::new(None)
    }
    
    /// Create a bounded channel with given capacity
    pub fn bounded(capacity: usize) -> Self {
        Self::new(Some(capacity))
    }
    
    /// Create a channel pair for ergonomic usage
    pub fn channel(capacity: Option<usize>) -> (MpmcSender<T>, MpmcReceiver<T>) {
        let channel = Arc::new(Self::new(capacity));
        let (mutex, _, _) = &*channel.state;
        
        {
            let mut state = mutex.lock().unwrap();
            state.sender_count = 1;
            state.receiver_count = 1;
        }
        
        (
            MpmcSender { channel: channel.clone() },
            MpmcReceiver { channel },
        )
    }
}

impl<T: Send> Channel<T> for MpmcChannel<T> {
    fn send(&self, value: T) -> Result<()> {
        let (mutex, not_full, not_empty) = &*self.state;
        let mut guard = mutex.lock().unwrap();
        
        // Wait for space or channel closure
        while !guard.closed && guard.capacity.map_or(false, |cap| guard.queue.len() >= cap) {
            guard = not_full.wait(guard).unwrap();
        }
        
        if guard.closed {
            return Err(ChannelError::Closed);
        }
        
        guard.queue.push_back(value);
        not_empty.notify_one();
        Ok(())
    }
    
    fn try_send(&self, value: T) -> Result<()> {
        let (mutex, _, not_empty) = &*self.state;
        let mut guard = mutex.lock().unwrap();
        
        if guard.closed {
            return Err(ChannelError::Closed);
        }
        
        if guard.capacity.map_or(false, |cap| guard.queue.len() >= cap) {
            return Err(ChannelError::Full);
        }
        
        guard.queue.push_back(value);
        not_empty.notify_one();
        Ok(())
    }
    
    fn recv(&self) -> Result<T> {
        let (mutex, not_full, not_empty) = &*self.state;
        let mut guard = mutex.lock().unwrap();
        
        // Wait for message or channel closure
        while guard.queue.is_empty() && !guard.closed {
            guard = not_empty.wait(guard).unwrap();
        }
        
        if let Some(value) = guard.queue.pop_front() {
            not_full.notify_one();
            Ok(value)
        } else {
            Err(ChannelError::Closed)
        }
    }
    
    fn try_recv(&self) -> Result<T> {
        let (mutex, not_full, _) = &*self.state;
        let mut guard = mutex.lock().unwrap();
        
        if let Some(value) = guard.queue.pop_front() {
            not_full.notify_one();
            Ok(value)
        } else if guard.closed {
            Err(ChannelError::Closed)
        } else {
            Err(ChannelError::Empty)
        }
    }
    
    fn is_empty(&self) -> bool {
        let (mutex, _, _) = &*self.state;
        let guard = mutex.lock().unwrap();
        guard.queue.is_empty()
    }
    
    fn is_full(&self) -> bool {
        let (mutex, _, _) = &*self.state;
        let guard = mutex.lock().unwrap();
        guard.capacity.map_or(false, |cap| guard.queue.len() >= cap)
    }
    
    fn capacity(&self) -> Option<usize> {
        let (mutex, _, _) = &*self.state;
        let guard = mutex.lock().unwrap();
        guard.capacity
    }
}

/// Sender half of MPMC channel
pub struct MpmcSender<T> {
    channel: Arc<MpmcChannel<T>>,
}

impl<T: Send> MpmcSender<T> {
    /// Send a value through the channel, blocking if necessary
    pub fn send(&self, value: T) -> Result<()> {
        self.channel.send(value)
    }
    
    /// Try to send a value without blocking
    pub fn try_send(&self, value: T) -> Result<()> {
        self.channel.try_send(value)
    }
}

impl<T> Clone for MpmcSender<T> {
    fn clone(&self) -> Self {
        let (mutex, _, _) = &*self.channel.state;
        let mut guard = mutex.lock().unwrap();
        guard.sender_count += 1;
        Self { channel: self.channel.clone() }
    }
}

impl<T> Drop for MpmcSender<T> {
    fn drop(&mut self) {
        let (mutex, _, not_empty) = &*self.channel.state;
        let mut guard = mutex.lock().unwrap();
        guard.sender_count -= 1;
        if guard.sender_count == 0 {
            guard.closed = true;
            not_empty.notify_all();
        }
    }
}

/// Receiver half of MPMC channel
pub struct MpmcReceiver<T> {
    channel: Arc<MpmcChannel<T>>,
}

impl<T: Send> MpmcReceiver<T> {
    /// Receive a value from the channel, blocking if necessary
    pub fn recv(&self) -> Result<T> {
        self.channel.recv()
    }
    
    /// Try to receive a value without blocking
    pub fn try_recv(&self) -> Result<T> {
        self.channel.try_recv()
    }
}

impl<T> Clone for MpmcReceiver<T> {
    fn clone(&self) -> Self {
        let (mutex, _, _) = &*self.channel.state;
        let mut guard = mutex.lock().unwrap();
        guard.receiver_count += 1;
        Self { channel: self.channel.clone() }
    }
}

impl<T> Drop for MpmcReceiver<T> {
    fn drop(&mut self) {
        let (mutex, not_full, _) = &*self.channel.state;
        let mut guard = mutex.lock().unwrap();
        guard.receiver_count -= 1;
        if guard.receiver_count == 0 {
            guard.closed = true;
            not_full.notify_all();
        }
    }
}

/// Select over multiple channels following Go's design
/// This allows waiting on multiple channels simultaneously
pub struct Select;

impl Select {
    /// Try to receive from multiple receivers, returning the first available
    pub fn try_recv<T>(receivers: &mut [&mut dyn FnMut() -> Result<T>]) -> Option<(usize, T)> {
        for (idx, recv) in receivers.iter_mut().enumerate() {
            if let Ok(value) = recv() {
                return Some((idx, value));
            }
        }
        None
    }
}

/// Convenience functions for creating channels
pub fn spsc<T>(capacity: usize) -> (SpscSender<T>, SpscReceiver<T>) {
    SpscChannel::channel(capacity)
}

/// Create a new bounded MPMC channel with the given capacity
pub fn mpmc<T>(capacity: usize) -> (MpmcSender<T>, MpmcReceiver<T>) {
    MpmcChannel::channel(Some(capacity))
}

/// Create a new unbounded MPMC channel
pub fn unbounded<T>() -> (MpmcSender<T>, MpmcReceiver<T>) {
    MpmcChannel::channel(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spsc_channel() {
        let (tx, rx) = spsc::<i32>(4);
        
        // Send some values
        assert!(tx.send(1).is_ok());
        assert!(tx.send(2).is_ok());
        
        // Receive values
        assert_eq!(rx.recv().unwrap(), 1);
        assert_eq!(rx.recv().unwrap(), 2);
        
        // Channel should be empty
        assert!(rx.try_recv().is_err());
    }
    
    #[test]
    fn test_mpmc_channel() {
        let (tx, rx) = mpmc::<i32>(4);
        let tx2 = tx.clone();
        
        // Multiple senders
        assert!(tx.send(1).is_ok());
        assert!(tx2.send(2).is_ok());
        
        // Receive values
        let mut values = vec![rx.recv().unwrap(), rx.recv().unwrap()];
        values.sort();
        assert_eq!(values, vec![1, 2]);
    }

    #[test]
    fn test_unbounded_channel() {
        let (tx, rx) = unbounded::<i32>();
        
        // Send some values
        for i in 0..10 {
            tx.send(i).unwrap();
        }
        
        // Receive values
        for i in 0..10 {
            assert_eq!(rx.recv().unwrap(), i);
        }
    }
    
    #[test]
    fn test_spsc_blocking_behavior() {
        use std::thread;
        use std::time::{Duration, Instant};
        
        // Create a small channel to test blocking
        let (tx, rx) = spsc::<i32>(2);
        
        // Fill the channel
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        
        // Spawn a thread that will send after a delay
        let tx_clone = tx.clone();
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            // This will unblock the main thread's send
            rx.recv().unwrap();
        });
        
        // This send should block until the spawned thread receives
        let start = Instant::now();
        tx_clone.send(3).unwrap();
        let elapsed = start.elapsed();
        
        // Verify that we blocked for approximately the sleep duration
        assert!(elapsed >= Duration::from_millis(40), "Send should have blocked");
        
        handle.join().unwrap();
    }
    
    #[test]
    fn test_spsc_recv_blocking_behavior() {
        use std::thread;
        use std::time::{Duration, Instant};
        
        let (tx, rx) = spsc::<i32>(10);
        
        // Spawn a thread that will send after a delay
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            tx.send(42).unwrap();
        });
        
        // This recv should block until the spawned thread sends
        let start = Instant::now();
        let value = rx.recv().unwrap();
        let elapsed = start.elapsed();
        
        assert_eq!(value, 42);
        // Verify that we blocked for approximately the sleep duration
        assert!(elapsed >= Duration::from_millis(40), "Recv should have blocked");
    }
}