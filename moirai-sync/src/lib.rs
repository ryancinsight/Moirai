//! High-performance synchronization primitives for Moirai concurrency library.
//!
//! This module provides specialized synchronization primitives that add value
//! beyond the standard library, following YAGNI and DRY principles.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering, AtomicI32};
use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::hint;
use std::collections::HashMap;
use std::hash::{Hash, BuildHasher, Hasher};
use std::collections::hash_map::RandomState;

// Re-export standard library primitives directly (DRY principle)
pub use std::sync::{
    Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard,
    Condvar, Barrier, OnceLock as Once,
};

#[cfg(target_os = "linux")]
mod futex {
    // Linux futex operations
    const FUTEX_WAIT: i32 = 0;
    const FUTEX_WAKE: i32 = 1;
    
    /// Wait on a futex if the value matches expected
    pub fn futex_wait(addr: *const i32, expected: i32) -> i32 {
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                addr,
                FUTEX_WAIT,
                expected,
                std::ptr::null::<libc::timespec>(),
                std::ptr::null::<i32>(),
                0,
            ) as i32
        }
    }
    
    /// Wake up waiters on a futex
    pub fn futex_wake(addr: *const i32, num_waiters: i32) -> i32 {
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                addr,
                FUTEX_WAKE,
                num_waiters,
                std::ptr::null::<libc::timespec>(),
                std::ptr::null::<i32>(),
                0,
            ) as i32
        }
    }
}

/// A wait group for synchronizing multiple threads (Go-inspired).
/// This provides value beyond standard library primitives.
pub struct WaitGroup {
    counter: AtomicU64,
    generation: AtomicU64,
}

impl WaitGroup {
    /// Create a new wait group.
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
            generation: AtomicU64::new(0),
        }
    }

    /// Add to the wait group counter.
    pub fn add(&self, delta: u64) {
        self.counter.fetch_add(delta, Ordering::Release);
    }

    /// Decrement the wait group counter.
    pub fn done(&self) {
        let old = self.counter.fetch_sub(1, Ordering::Release);
        if old == 1 {
            // Last one out, increment generation to wake waiters
            self.generation.fetch_add(1, Ordering::Release);
            std::thread::yield_now(); // Give waiters a chance to wake
        }
    }

    /// Wait for the counter to reach zero.
    pub fn wait(&self) {
        let gen = self.generation.load(Ordering::Acquire);
        while self.counter.load(Ordering::Acquire) > 0 {
            hint::spin_loop();
            if self.generation.load(Ordering::Acquire) != gen {
                break;
            }
        }
    }
}

/// An atomic counter with convenience methods.
pub struct AtomicCounter {
    inner: AtomicU64,
}

impl AtomicCounter {
    /// Create a new atomic counter.
    pub const fn new(value: u64) -> Self {
        Self {
            inner: AtomicU64::new(value),
        }
    }

    /// Increment the counter and return the new value.
    pub fn inc(&self) -> u64 {
        self.inner.fetch_add(1, Ordering::Relaxed).wrapping_add(1)
    }

    /// Decrement the counter and return the new value.
    pub fn dec(&self) -> u64 {
        self.inner.fetch_sub(1, Ordering::Relaxed).wrapping_sub(1)
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.inner.load(Ordering::Relaxed)
    }

    /// Set the value.
    pub fn set(&self, value: u64) {
        self.inner.store(value, Ordering::Relaxed);
    }
}

/// A fast mutex with futex-based blocking on Linux.
/// This provides real value over std::sync::Mutex through adaptive spinning.
pub struct FastMutex<T> {
    #[cfg(target_os = "linux")]
    state: AtomicI32,  // 0 = unlocked, 1 = locked, 2 = locked with waiters
    #[cfg(not(target_os = "linux"))]
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for FastMutex<T> {}
unsafe impl<T: Send> Sync for FastMutex<T> {}

impl<T> FastMutex<T> {
    /// Create a new fast mutex.
    pub const fn new(data: T) -> Self {
        Self {
            #[cfg(target_os = "linux")]
            state: AtomicI32::new(0),
            #[cfg(not(target_os = "linux"))]
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

    /// Lock the mutex with adaptive spinning.
    pub fn lock(&self) -> FastMutexGuard<'_, T> {
        // Try to acquire the lock with spinning first
        for _ in 0..100 {
            if self.try_lock_fast() {
                return FastMutexGuard {
                    mutex: self,
                    _phantom: std::marker::PhantomData,
                };
            }
            hint::spin_loop();
        }
        
        // Fall back to blocking
        self.lock_slow();
        FastMutexGuard {
            mutex: self,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    fn try_lock_fast(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.state.compare_exchange_weak(
                0, 1,
                Ordering::Acquire,
                Ordering::Relaxed
            ).is_ok()
        }
        #[cfg(not(target_os = "linux"))]
        {
            !self.locked.swap(true, Ordering::Acquire)
        }
    }

    #[cold]
    fn lock_slow(&self) {
        #[cfg(target_os = "linux")]
        {
            loop {
                let state = self.state.load(Ordering::Relaxed);
                
                if state == 0 && self.state.compare_exchange_weak(
                    0, 1,
                    Ordering::Acquire,
                    Ordering::Relaxed
                ).is_ok() {
                    return;
                }
                
                if state == 1 {
                    self.state.compare_exchange_weak(
                        1, 2,
                        Ordering::Relaxed,
                        Ordering::Relaxed
                    ).ok();
                }
                
                futex::futex_wait(self.state.as_ptr(), 2);
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            while self.locked.load(Ordering::Relaxed) {
                std::thread::yield_now();
            }
            while self.locked.swap(true, Ordering::Acquire) {
                while self.locked.load(Ordering::Relaxed) {
                    std::thread::yield_now();
                }
            }
        }
    }

    fn unlock(&self) {
        #[cfg(target_os = "linux")]
        {
            if self.state.swap(0, Ordering::Release) == 2 {
                futex::futex_wake(self.state.as_ptr(), 1);
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            self.locked.store(false, Ordering::Release);
        }
    }
}

/// Guard for FastMutex that automatically unlocks on drop.
pub struct FastMutexGuard<'a, T> {
    mutex: &'a FastMutex<T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> Drop for FastMutexGuard<'a, T> {
    fn drop(&mut self) {
        self.mutex.unlock();
    }
}

impl<'a, T> Deref for FastMutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.mutex.data.get() }
    }
}

impl<'a, T> DerefMut for FastMutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.mutex.data.get() }
    }
}

/// A spin lock for very short critical sections.
/// Use only when you know the critical section is extremely short.
pub struct SpinLock<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for SpinLock<T> {}
unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    /// Create a new spin lock.
    pub const fn new(data: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

    /// Lock the spin lock.
    pub fn lock(&self) -> SpinLockGuard<'_, T> {
        while self.locked.swap(true, Ordering::Acquire) {
            while self.locked.load(Ordering::Relaxed) {
                hint::spin_loop();
            }
        }
        SpinLockGuard {
            lock: self,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Try to lock without spinning.
    pub fn try_lock(&self) -> Option<SpinLockGuard<'_, T>> {
        if !self.locked.swap(true, Ordering::Acquire) {
            Some(SpinLockGuard {
                lock: self,
                _phantom: std::marker::PhantomData,
            })
        } else {
            None
        }
    }
}

/// Guard for SpinLock that automatically unlocks on drop.
pub struct SpinLockGuard<'a, T> {
    lock: &'a SpinLock<T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> Drop for SpinLockGuard<'a, T> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}

impl<'a, T> Deref for SpinLockGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T> DerefMut for SpinLockGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.lock.data.get() }
    }
}

// Re-export LockFreeStack from moirai-core to maintain DRY principle
pub use moirai_core::pool::LockFreeStack;

/// Concurrent hash map with segment-based locking for scalability.
/// This provides better scalability than a single mutex-protected HashMap.
pub struct ConcurrentHashMap<K, V, S = RandomState> {
    segments: Vec<Mutex<HashMap<K, V, S>>>,
    hasher: S,
}

impl<K: Hash + Eq, V> ConcurrentHashMap<K, V> {
    /// Create a new concurrent hash map with default hasher.
    pub fn new() -> Self {
        Self::with_segments(16)
    }

    /// Create with a specific number of segments (must be power of 2).
    pub fn with_segments(num_segments: usize) -> Self {
        let num_segments = num_segments.next_power_of_two();
        
        let segments = (0..num_segments)
            .map(|_| Mutex::new(HashMap::new()))
            .collect();

        Self {
            segments,
            hasher: RandomState::new(),
        }
    }
}

impl<K: Hash + Eq, V, S: BuildHasher> ConcurrentHashMap<K, V, S> {
    /// Get the segment index for a key.
    fn segment_index(&self, key: &K) -> usize {
        let mut hasher = self.hasher.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        // Use bitmask for even distribution across power-of-2 segments
        (hash as usize) & (self.segments.len() - 1)
    }

    /// Insert a key-value pair.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let idx = self.segment_index(&key);
        self.segments[idx].lock().unwrap().insert(key, value)
    }

    /// Get a value by key.
    pub fn get(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let idx = self.segment_index(key);
        self.segments[idx].lock().unwrap().get(key).cloned()
    }

    /// Remove a key-value pair.
    pub fn remove(&self, key: &K) -> Option<V> {
        let idx = self.segment_index(key);
        self.segments[idx].lock().unwrap().remove(key)
    }

    /// Check if a key exists.
    pub fn contains_key(&self, key: &K) -> bool {
        let idx = self.segment_index(key);
        self.segments[idx].lock().unwrap().contains_key(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;

    #[test]
    fn test_wait_group() {
        let wg = Arc::new(WaitGroup::new());
        let mut handles = vec![];

        wg.add(3);

        for i in 0..3 {
            let wg = wg.clone();
            handles.push(thread::spawn(move || {
                thread::sleep(std::time::Duration::from_millis(i * 10));
                wg.done();
            }));
        }

        wg.wait();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_fast_mutex() {
        let mutex = Arc::new(FastMutex::new(0));
        let mut handles = vec![];

        for _ in 0..10 {
            let mutex = mutex.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let mut guard = mutex.lock();
                    *guard += 1;
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(*mutex.lock(), 1000);
    }

    #[test]
    fn test_lock_free_stack() {
        let stack = Arc::new(LockFreeStack::new());
        let mut handles = vec![];

        // Push from multiple threads
        for i in 0..10 {
            let stack = stack.clone();
            handles.push(thread::spawn(move || {
                stack.push(i);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Pop all items
        let mut items = vec![];
        while let Some(item) = stack.pop() {
            items.push(item);
        }

        items.sort();
        assert_eq!(items, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_concurrent_hashmap() {
        let map = ConcurrentHashMap::new();
        
        // Insert some values
        map.insert("key1", 100);
        map.insert("key2", 200);
        
        // Test retrieval
        assert_eq!(map.get(&"key1"), Some(100));
        assert_eq!(map.get(&"key2"), Some(200));
        assert_eq!(map.get(&"key3"), None);
        
        // Test removal
        assert_eq!(map.remove(&"key1"), Some(100));
        assert_eq!(map.get(&"key1"), None);
    }
    
    #[test]
    fn test_concurrent_hashmap_segment_distribution() {
        use std::collections::HashSet;
        
        // Create a map with 16 segments
        let map = ConcurrentHashMap::<i32, i32>::with_segments(16);
        
        // Track which segments are used
        let mut segments_used = HashSet::new();
        
        // Insert many keys and track segment distribution
        for i in 0..1000 {
            map.insert(i, i);
            let segment_idx = map.segment_index(&i);
            segments_used.insert(segment_idx);
        }
        
        // With proper distribution, we should use most segments
        // With 1000 keys across 16 segments, we expect to use all segments
        assert!(segments_used.len() >= 14, "Poor segment distribution: only {} of 16 segments used", segments_used.len());
        
        // Verify all keys can be retrieved
        for i in 0..1000 {
            assert_eq!(map.get(&i), Some(i));
        }
    }
}