//! Utility functions and data structures for Moirai concurrency library.

#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use std::collections::HashMap;

// Core imports needed for no_std compatibility
use core::sync::atomic::{AtomicUsize, AtomicU64, Ordering};

// Std-only imports
#[cfg(feature = "std")]
use std::{
    boxed::Box,
    vec::{self, Vec},
};

/// Cache line size for alignment optimizations.
pub const CACHE_LINE_SIZE: usize = 64;

/// Align a value to the cache line boundary.
#[must_use]
pub const fn align_to_cache_line(size: usize) -> usize {
    (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
}

/// A cache-aligned wrapper for data structures.
#[repr(align(64))]
pub struct CacheAligned<T>(pub T);

impl<T> CacheAligned<T> {
    /// Create a new cache-aligned value.
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Get a reference to the inner value.
    pub const fn get(&self) -> &T {
        &self.0
    }

    /// Get a mutable reference to the inner value.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> core::ops::Deref for CacheAligned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> core::ops::DerefMut for CacheAligned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A power-of-two sized ring buffer optimized for single-producer, single-consumer scenarios.
#[repr(align(64))]
pub struct RingBuffer<T> {
    data: Box<[Option<T>]>,
    capacity: usize,
    mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity (must be a power of 2).
    pub fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two(), "Capacity must be a power of 2");
        assert!(capacity > 0, "Capacity must be greater than 0");

        let data = (0..capacity).map(|_| None).collect::<Vec<_>>().into_boxed_slice();

        Self {
            data,
            capacity,
            mask: capacity - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Get the capacity of the ring buffer.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if the ring buffer is empty.
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head == tail
    }

    /// Check if the ring buffer is full.
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (tail + 1) & self.mask == head
    }

    /// Get the current size of the ring buffer.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (tail.wrapping_sub(head)) & self.mask
    }

    /// Try to push an item to the ring buffer.
    /// Returns `Err(item)` if the buffer is full.
    pub fn try_push(&self, item: T) -> Result<(), T> {
        let tail = self.tail.load(Ordering::Acquire);
        let next_tail = (tail + 1) & self.mask;
        let head = self.head.load(Ordering::Acquire);

        if next_tail == head {
            return Err(item); // Buffer is full
        }

        // Safety: We've checked that the slot is available
        unsafe {
            let slot = &mut *(self.data.as_ptr().add(tail) as *mut Option<T>);
            *slot = Some(item);
        }

        self.tail.store(next_tail, Ordering::Release);
        Ok(())
    }

    /// Try to pop an item from the ring buffer.
    /// Returns `None` if the buffer is empty.
    pub fn try_pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);

        if head == tail {
            return None; // Buffer is empty
        }

        // Safety: We've checked that there's an item available
        let item = unsafe {
            let slot = &mut *(self.data.as_ptr().add(head) as *mut Option<T>);
            slot.take()
        };

        let next_head = (head + 1) & self.mask;
        self.head.store(next_head, Ordering::Release);

        item
    }

    /// Clear all items from the ring buffer.
    pub fn clear(&self) {
        while let Some(_) = self.try_pop() {
            // Items are dropped automatically
        }
    }
}

// Safety: RingBuffer is safe to send between threads
unsafe impl<T: Send> Send for RingBuffer<T> {}
unsafe impl<T: Send> Sync for RingBuffer<T> {}

/// A lock-free, multi-producer, multi-consumer queue using a linked list structure.
pub struct LockFreeQueue<T> {
    head: core::sync::atomic::AtomicPtr<Node<T>>,
    tail: core::sync::atomic::AtomicPtr<Node<T>>,
}

struct Node<T> {
    data: Option<T>,
    next: core::sync::atomic::AtomicPtr<Node<T>>,
}

impl<T> Node<T> {
    fn new(data: Option<T>) -> Box<Self> {
        Box::new(Self {
            data,
            next: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
        })
    }
}

impl<T> LockFreeQueue<T> {
    /// Create a new lock-free queue.
    pub fn new() -> Self {
        let dummy = Node::new(None);
        let dummy_ptr = Box::into_raw(dummy);

        Self {
            head: core::sync::atomic::AtomicPtr::new(dummy_ptr),
            tail: core::sync::atomic::AtomicPtr::new(dummy_ptr),
        }
    }

    /// Enqueue an item to the back of the queue.
    pub fn enqueue(&self, item: T) {
        let new_node = Box::into_raw(Node::new(Some(item)));

        loop {
            let tail = self.tail.load(core::sync::atomic::Ordering::Acquire);
            let next = unsafe { (*tail).next.load(core::sync::atomic::Ordering::Acquire) };

            if tail == self.tail.load(core::sync::atomic::Ordering::Acquire) {
                if next.is_null() {
                    if unsafe {
                        (*tail).next.compare_exchange_weak(
                            next,
                            new_node,
                            core::sync::atomic::Ordering::Release,
                            core::sync::atomic::Ordering::Relaxed,
                        )
                    }
                    .is_ok()
                    {
                        break;
                    }
                } else {
                    let _ = self.tail.compare_exchange_weak(
                        tail,
                        next,
                        core::sync::atomic::Ordering::Release,
                        core::sync::atomic::Ordering::Relaxed,
                    );
                }
            }
        }

        let _ = self.tail.compare_exchange_weak(
            self.tail.load(core::sync::atomic::Ordering::Acquire),
            new_node,
            core::sync::atomic::Ordering::Release,
            core::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Try to dequeue an item from the front of the queue.
    /// Returns `None` if the queue is empty.
    pub fn try_dequeue(&self) -> Option<T> {
        loop {
            let head = self.head.load(core::sync::atomic::Ordering::Acquire);
            let tail = self.tail.load(core::sync::atomic::Ordering::Acquire);
            let next = unsafe { (*head).next.load(core::sync::atomic::Ordering::Acquire) };

            if head == self.head.load(core::sync::atomic::Ordering::Acquire) {
                if head == tail {
                    if next.is_null() {
                        return None; // Queue is empty
                    }
                    let _ = self.tail.compare_exchange_weak(
                        tail,
                        next,
                        core::sync::atomic::Ordering::Release,
                        core::sync::atomic::Ordering::Relaxed,
                    );
                } else {
                    if next.is_null() {
                        continue;
                    }

                    let data = unsafe { (*next).data.take() };

                    if self
                        .head
                        .compare_exchange_weak(
                            head,
                            next,
                            core::sync::atomic::Ordering::Release,
                            core::sync::atomic::Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        unsafe {
                            drop(Box::from_raw(head));
                        }
                        return data;
                    }
                }
            }
        }
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(core::sync::atomic::Ordering::Acquire);
        let tail = self.tail.load(core::sync::atomic::Ordering::Acquire);
        let next = unsafe { (*head).next.load(core::sync::atomic::Ordering::Acquire) };

        head == tail && next.is_null()
    }
}

impl<T> Default for LockFreeQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for LockFreeQueue<T> {
    fn drop(&mut self) {
        while self.try_dequeue().is_some() {}

        // Clean up the dummy node
        let head = self.head.load(core::sync::atomic::Ordering::Acquire);
        if !head.is_null() {
            unsafe {
                drop(Box::from_raw(head));
            }
        }
    }
}

// Safety: LockFreeQueue is safe to send between threads
unsafe impl<T: Send> Send for LockFreeQueue<T> {}
unsafe impl<T: Send> Sync for LockFreeQueue<T> {}

/// Atomic counter with overflow protection.
#[derive(Debug)]
pub struct AtomicCounter {
    value: core::sync::atomic::AtomicU64,
}

impl AtomicCounter {
    /// Create a new atomic counter with initial value 0.
    pub const fn new() -> Self {
        Self {
            value: core::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Create a new atomic counter with the given initial value.
    pub const fn with_value(initial: u64) -> Self {
        Self {
            value: core::sync::atomic::AtomicU64::new(initial),
        }
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }

    /// Set the value.
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Release);
    }

    /// Increment the counter and return the new value.
    pub fn increment(&self) -> u64 {
        self.value.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Decrement the counter and return the new value.
    /// Returns 0 if the counter would underflow.
    pub fn decrement(&self) -> u64 {
        loop {
            let current = self.value.load(Ordering::Acquire);
            if current == 0 {
                return 0;
            }

            match self.value.compare_exchange_weak(
                current,
                current - 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return current - 1,
                Err(_) => continue,
            }
        }
    }

    /// Add a value to the counter and return the new value.
    pub fn add(&self, value: u64) -> u64 {
        self.value.fetch_add(value, Ordering::AcqRel) + value
    }

    /// Subtract a value from the counter and return the new value.
    /// Returns 0 if the counter would underflow.
    pub fn sub(&self, value: u64) -> u64 {
        loop {
            let current = self.value.load(Ordering::Acquire);
            if current <= value {
                match self.value.compare_exchange_weak(
                    current,
                    0,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return 0,
                    Err(_) => continue,
                }
            }

            match self.value.compare_exchange_weak(
                current,
                current - value,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return current - value,
                Err(_) => continue,
            }
        }
    }

    /// Reset the counter to 0 and return the previous value.
    pub fn reset(&self) -> u64 {
        self.value.swap(0, Ordering::AcqRel)
    }
}

impl Default for AtomicCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for AtomicCounter {
    fn clone(&self) -> Self {
        Self::with_value(self.get())
    }
}

/// Backoff strategy for lock-free algorithms.
#[derive(Debug)]
pub struct Backoff {
    step: AtomicUsize,
}

impl Backoff {
    /// Create a new backoff strategy.
    pub const fn new() -> Self {
        Self {
            step: core::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Perform a backoff operation.
    pub fn backoff(&self) {
        let step = self.step.fetch_add(1, Ordering::Relaxed);

        if step <= 6 {
            // Spin for a few iterations
            for _ in 0..(1 << step.min(6)) {
                core::hint::spin_loop();
            }
        } else {
            // Yield to the scheduler
            #[cfg(feature = "std")]
            std::thread::yield_now();
        }
    }

    /// Reset the backoff strategy.
    pub fn reset(&self) {
        self.step.store(0, Ordering::Relaxed);
    }
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Backoff {
    fn clone(&self) -> Self {
        Self {
            step: AtomicUsize::new(self.step.load(Ordering::Relaxed)),
        }
    }
}

/// Fast random number generator using xorshift algorithm.
#[derive(Debug, Clone)]
pub struct FastRng {
    state: u64,
}

impl FastRng {
    /// Create a new random number generator with a seed.
    pub const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Create a new random number generator with a default seed.
    pub fn default_seed() -> Self {
        #[cfg(feature = "std")]
        {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            use std::time::SystemTime;

            let mut hasher = DefaultHasher::new();
            SystemTime::now().hash(&mut hasher);
            Self::new(hasher.finish())
        }

        #[cfg(not(feature = "std"))]
        {
            Self::new(0x123456789abcdef0)
        }
    }

    /// Generate the next random number.
    pub fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generate a random number in the range [0, max).
    pub fn next_range(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        self.next() % max
    }

    /// Generate a random boolean.
    pub fn next_bool(&mut self) -> bool {
        self.next() & 1 == 1
    }

    /// Generate a random f64 in the range [0.0, 1.0).
    pub fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

impl Default for FastRng {
    fn default() -> Self {
        Self::default_seed()
    }
}

/// Bit manipulation utilities.
pub mod bits {
    /// Count the number of set bits in a u64.
    #[must_use]
    pub const fn popcount_u64(x: u64) -> u32 {
        x.count_ones()
    }

    /// Find the position of the least significant set bit.
    /// Returns 64 if no bits are set.
    #[must_use]
    pub const fn trailing_zeros_u64(x: u64) -> u32 {
        x.trailing_zeros()
    }

    /// Find the position of the most significant set bit.
    /// Returns 64 if no bits are set.
    #[must_use]
    pub const fn leading_zeros_u64(x: u64) -> u32 {
        x.leading_zeros()
    }

    /// Check if a number is a power of 2.
    #[must_use]
    pub const fn is_power_of_two(x: u64) -> bool {
        x != 0 && (x & (x - 1)) == 0
    }

    /// Round up to the next power of 2.
    #[must_use]
    pub const fn next_power_of_two(x: u64) -> u64 {
        if x <= 1 {
            1
        } else {
            1 << (64 - (x - 1).leading_zeros())
        }
    }
}

/// Memory utilities.
pub mod memory {
    use super::CacheAligned;
    #[cfg(feature = "std")]
    use std::{vec, vec::Vec};

    /// Prefetch memory for reading.
    #[inline(always)]
    pub fn prefetch_read<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = ptr; // Suppress unused variable warning
        }
    }

    /// Prefetch memory for writing.
    #[inline(always)]
    pub fn prefetch_write<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                core::arch::x86_64::_mm_prefetch(
                    ptr as *const i8,
                    core::arch::x86_64::_MM_HINT_T0,
                );
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = ptr; // Suppress unused variable warning
        }
    }

    /// Create a cache-aligned array.
    pub fn aligned_vec<T: Clone>(value: T, count: usize) -> CacheAligned<Vec<T>> {
        CacheAligned::new(vec![value; count])
    }
}

/// Time utilities.
#[cfg(feature = "std")]
pub mod time {
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    /// High-resolution timer for performance measurements.
    #[derive(Debug, Clone)]
    pub struct HighResTimer {
        start: Instant,
    }

    impl HighResTimer {
        /// Create a new timer and start measuring.
        pub fn new() -> Self {
            Self {
                start: Instant::now(),
            }
        }

        /// Get the elapsed time since the timer was created.
        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }

        /// Get the elapsed time in nanoseconds.
        pub fn elapsed_nanos(&self) -> u64 {
            self.elapsed().as_nanos() as u64
        }

        /// Get the elapsed time in microseconds.
        pub fn elapsed_micros(&self) -> u64 {
            self.elapsed().as_micros() as u64
        }

        /// Get the elapsed time in milliseconds.
        pub fn elapsed_millis(&self) -> u64 {
            self.elapsed().as_millis() as u64
        }

        /// Reset the timer.
        pub fn reset(&mut self) {
            self.start = Instant::now();
        }
    }

    impl Default for HighResTimer {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Get the current Unix timestamp in nanoseconds.
    pub fn unix_timestamp_nanos() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// Get the current Unix timestamp in microseconds.
    pub fn unix_timestamp_micros() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }

    /// Get the current Unix timestamp in milliseconds.
    pub fn unix_timestamp_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// Statistics utilities.
pub mod stats {
    #[cfg(feature = "std")]
    use std::{vec, vec::Vec};
    /// Moving average calculator.
    #[derive(Debug, Clone)]
    pub struct MovingAverage {
        values: Vec<f64>,
        sum: f64,
        index: usize,
        count: usize,
        capacity: usize,
    }

    impl MovingAverage {
        /// Create a new moving average calculator with the given window size.
        pub fn new(window_size: usize) -> Self {
            assert!(window_size > 0, "Window size must be greater than 0");

            Self {
                values: vec![0.0; window_size],
                sum: 0.0,
                index: 0,
                count: 0,
                capacity: window_size,
            }
        }

        /// Add a new value to the moving average.
        pub fn add(&mut self, value: f64) {
            if self.count < self.capacity {
                self.sum += value;
                self.count += 1;
            } else {
                self.sum = self.sum - self.values[self.index] + value;
            }

            self.values[self.index] = value;
            self.index = (self.index + 1) % self.capacity;
        }

        /// Get the current moving average.
        pub fn average(&self) -> f64 {
            if self.count == 0 {
                0.0
            } else {
                self.sum / self.count as f64
            }
        }

        /// Get the number of values added.
        pub fn count(&self) -> usize {
            self.count
        }

        /// Check if the window is full.
        pub fn is_full(&self) -> bool {
            self.count == self.capacity
        }

        /// Reset the moving average.
        pub fn reset(&mut self) {
            self.sum = 0.0;
            self.index = 0;
            self.count = 0;
            for value in &mut self.values {
                *value = 0.0;
            }
        }
    }

    /// Simple statistics calculator.
    #[derive(Debug, Clone)]
    pub struct Statistics {
        count: u64,
        sum: f64,
        sum_squares: f64,
        min: f64,
        max: f64,
    }

    impl Statistics {
        /// Create a new statistics calculator.
        pub const fn new() -> Self {
            Self {
                count: 0,
                sum: 0.0,
                sum_squares: 0.0,
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
            }
        }

        /// Add a value to the statistics.
        pub fn add(&mut self, value: f64) {
            self.count += 1;
            self.sum += value;
            self.sum_squares += value * value;

            if value < self.min {
                self.min = value;
            }
            if value > self.max {
                self.max = value;
            }
        }

        /// Get the number of values added.
        pub const fn count(&self) -> u64 {
            self.count
        }

        /// Get the mean value.
        pub fn mean(&self) -> f64 {
            if self.count == 0 {
                0.0
            } else {
                self.sum / self.count as f64
            }
        }

        /// Get the variance.
        pub fn variance(&self) -> f64 {
            if self.count <= 1 {
                0.0
            } else {
                let mean = self.mean();
                (self.sum_squares - self.sum * mean) / (self.count - 1) as f64
            }
        }

        /// Get the standard deviation.
        pub fn std_dev(&self) -> f64 {
            self.variance().sqrt()
        }

        /// Get the minimum value.
        pub fn min(&self) -> f64 {
            if self.count == 0 {
                0.0
            } else {
                self.min
            }
        }

        /// Get the maximum value.
        pub fn max(&self) -> f64 {
            if self.count == 0 {
                0.0
            } else {
                self.max
            }
        }

        /// Reset the statistics.
        pub fn reset(&mut self) {
            self.count = 0;
            self.sum = 0.0;
            self.sum_squares = 0.0;
            self.min = f64::INFINITY;
            self.max = f64::NEG_INFINITY;
        }
    }

    impl Default for Statistics {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Probabilistic data structures.
pub mod probabilistic {
    #[allow(unused_imports)]
    use super::FastRng;
    #[cfg(feature = "std")]
    use std::{vec, vec::Vec};

    /// Bloom filter for fast membership testing.
    pub struct BloomFilter {
        bits: Vec<u64>,
        hash_count: usize,
        bit_count: usize,
    }

    impl BloomFilter {
        /// Create a new Bloom filter with the given capacity and false positive rate.
        pub fn new(capacity: usize, false_positive_rate: f64) -> Self {
            assert!(capacity > 0, "Capacity must be greater than 0");
            assert!(
                false_positive_rate > 0.0 && false_positive_rate < 1.0,
                "False positive rate must be between 0 and 1"
            );

            let bit_count = Self::optimal_bit_count(capacity, false_positive_rate);
            let hash_count = Self::optimal_hash_count(capacity, bit_count);

            let word_count = (bit_count + 63) / 64;

            Self {
                bits: vec![0; word_count],
                hash_count,
                bit_count,
            }
        }

        /// Add an item to the Bloom filter.
        pub fn insert(&mut self, item: &[u8]) {
            let hashes = self.hash_item(item);
            for hash in hashes {
                let bit_index = hash % self.bit_count;
                let word_index = bit_index / 64;
                let bit_offset = bit_index % 64;
                self.bits[word_index] |= 1u64 << bit_offset;
            }
        }

        /// Check if an item might be in the Bloom filter.
        /// Returns `true` if the item might be present, `false` if it's definitely not present.
        pub fn contains(&self, item: &[u8]) -> bool {
            let hashes = self.hash_item(item);
            for hash in hashes {
                let bit_index = hash % self.bit_count;
                let word_index = bit_index / 64;
                let bit_offset = bit_index % 64;
                if (self.bits[word_index] & (1u64 << bit_offset)) == 0 {
                    return false;
                }
            }
            true
        }

        /// Clear all items from the Bloom filter.
        pub fn clear(&mut self) {
            for word in &mut self.bits {
                *word = 0;
            }
        }

        /// Get the estimated number of items in the filter.
        pub fn estimated_count(&self) -> usize {
            let set_bits = self.bits.iter().map(|word| word.count_ones() as usize).sum::<usize>();

            if set_bits == 0 {
                return 0;
            }

            let ratio = set_bits as f64 / self.bit_count as f64;
            let estimate = -(self.bit_count as f64 * (1.0_f64 - ratio).ln()) / self.hash_count as f64;

            estimate.round() as usize
        }

        fn optimal_bit_count(capacity: usize, false_positive_rate: f64) -> usize {
            let bits = -(capacity as f64 * false_positive_rate.ln()) / (2.0_f64.ln().powi(2));
            bits.ceil() as usize
        }

        fn optimal_hash_count(capacity: usize, bit_count: usize) -> usize {
            let hashes = (bit_count as f64 / capacity as f64) * 2.0_f64.ln();
            hashes.round().max(1.0) as usize
        }

        fn hash_item(&self, item: &[u8]) -> Vec<usize> {
            // Simple hash function - in production, use a proper hash function
            let mut hashes = Vec::with_capacity(self.hash_count);
            let mut hash1 = 0u64;
            let mut hash2 = 0u64;

            for &byte in item {
                hash1 = hash1.wrapping_mul(31).wrapping_add(byte as u64);
                hash2 = hash2.wrapping_mul(37).wrapping_add(byte as u64);
            }

            for i in 0..self.hash_count {
                let hash = hash1.wrapping_add((i as u64).wrapping_mul(hash2));
                hashes.push(hash as usize);
            }

            hashes
        }
    }

    /// HyperLogLog for cardinality estimation.
    pub struct HyperLogLog {
        buckets: Vec<u8>,
        bucket_count: usize,
        alpha: f64,
    }

    impl HyperLogLog {
        /// Create a new HyperLogLog with the given precision (4-16).
        pub fn new(precision: usize) -> Self {
            assert!(
                precision >= 4 && precision <= 16,
                "Precision must be between 4 and 16"
            );

            let bucket_count = 1 << precision;
            let alpha = match bucket_count {
                16 => 0.673,
                32 => 0.697,
                64 => 0.709,
                _ => 0.7213 / (1.0 + 1.079 / bucket_count as f64),
            };

            Self {
                buckets: vec![0; bucket_count],
                bucket_count,
                alpha,
            }
        }

        /// Add an item to the HyperLogLog.
        pub fn add(&mut self, item: &[u8]) {
            let hash = self.hash_item(item);
            let bucket_index = (hash & (self.bucket_count as u64 - 1)) as usize;
            let remaining_bits = hash >> (64 - self.bucket_count.trailing_zeros());
            let leading_zeros = remaining_bits.leading_zeros() as u8 + 1;

            if leading_zeros > self.buckets[bucket_index] {
                self.buckets[bucket_index] = leading_zeros;
            }
        }

        /// Estimate the cardinality.
        pub fn estimate(&self) -> f64 {
            let raw_estimate = self.alpha
                * (self.bucket_count as f64).powi(2)
                / self.buckets.iter().map(|&b| 2.0_f64.powi(-(b as i32))).sum::<f64>();

            // Apply bias correction for small estimates
            if raw_estimate <= 2.5 * self.bucket_count as f64 {
                let zeros = self.buckets.iter().filter(|&&b| b == 0).count();
                if zeros != 0 {
                    return self.bucket_count as f64 * (self.bucket_count as f64 / zeros as f64).ln();
                }
            }

            raw_estimate
        }

        /// Clear all items from the HyperLogLog.
        pub fn clear(&mut self) {
            for bucket in &mut self.buckets {
                *bucket = 0;
            }
        }

        fn hash_item(&self, item: &[u8]) -> u64 {
            // Simple hash function - in production, use a proper hash function
            let mut hash = 0u64;
            for &byte in item {
                hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
            }
            hash
        }
    }
}

/// SIMD operations module.
pub mod simd;