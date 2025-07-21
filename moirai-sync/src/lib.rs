//! Synchronization primitives for Moirai concurrency library.

use std::sync::{
    Arc, Mutex as StdMutex, RwLock as StdRwLock, Condvar as StdCondvar,
    atomic::{AtomicU64, Ordering},
    Barrier as StdBarrier,
};
use std::sync::OnceLock;

/// A mutual exclusion primitive.
pub struct Mutex<T> {
    inner: StdMutex<T>,
}

/// A reader-writer lock.
pub struct RwLock<T> {
    inner: StdRwLock<T>,
}

/// A condition variable.
pub struct Condvar {
    inner: StdCondvar,
}

/// A barrier for synchronizing multiple threads.
pub struct Barrier {
    inner: StdBarrier,
}

/// A one-time initialization primitive.
pub struct Once {
    inner: OnceLock<()>,
}

/// A wait group for synchronizing multiple threads.
pub struct WaitGroup {
    counter: AtomicU64,
    condvar: Condvar,
    mutex: Mutex<()>,
}

/// An atomic counter.
pub struct AtomicCounter {
    inner: AtomicU64,
}

impl<T> Mutex<T> {
    /// Create a new mutex.
    pub fn new(data: T) -> Self {
        Self {
            inner: StdMutex::new(data),
        }
    }

    /// Lock the mutex.
    pub fn lock(&self) -> std::sync::LockResult<std::sync::MutexGuard<'_, T>> {
        self.inner.lock()
    }

    /// Try to lock the mutex.
    pub fn try_lock(&self) -> std::sync::TryLockResult<std::sync::MutexGuard<'_, T>> {
        self.inner.try_lock()
    }
}

impl<T> RwLock<T> {
    /// Create a new reader-writer lock.
    pub fn new(data: T) -> Self {
        Self {
            inner: StdRwLock::new(data),
        }
    }

    /// Acquire a read lock.
    pub fn read(&self) -> std::sync::LockResult<std::sync::RwLockReadGuard<'_, T>> {
        self.inner.read()
    }

    /// Acquire a write lock.
    pub fn write(&self) -> std::sync::LockResult<std::sync::RwLockWriteGuard<'_, T>> {
        self.inner.write()
    }

    /// Try to acquire a read lock.
    pub fn try_read(&self) -> std::sync::TryLockResult<std::sync::RwLockReadGuard<'_, T>> {
        self.inner.try_read()
    }

    /// Try to acquire a write lock.
    pub fn try_write(&self) -> std::sync::TryLockResult<std::sync::RwLockWriteGuard<'_, T>> {
        self.inner.try_write()
    }
}

impl Condvar {
    /// Create a new condition variable.
    pub fn new() -> Self {
        Self {
            inner: StdCondvar::new(),
        }
    }

    /// Wait on the condition variable.
    pub fn wait<'a, T>(&self, guard: std::sync::MutexGuard<'a, T>) -> std::sync::LockResult<std::sync::MutexGuard<'a, T>> {
        self.inner.wait(guard)
    }

    /// Wait on the condition variable with a timeout.
    pub fn wait_timeout<'a, T>(&self, guard: std::sync::MutexGuard<'a, T>, dur: std::time::Duration) -> std::sync::LockResult<(std::sync::MutexGuard<'a, T>, std::sync::WaitTimeoutResult)> {
        self.inner.wait_timeout(guard, dur)
    }

    /// Notify one waiting thread.
    pub fn notify_one(&self) {
        self.inner.notify_one();
    }

    /// Notify all waiting threads.
    pub fn notify_all(&self) {
        self.inner.notify_all();
    }
}

impl Default for Condvar {
    fn default() -> Self {
        Self::new()
    }
}

impl Barrier {
    /// Create a new barrier.
    pub fn new(n: usize) -> Self {
        Self {
            inner: StdBarrier::new(n),
        }
    }

    /// Wait at the barrier.
    pub fn wait(&self) -> std::sync::BarrierWaitResult {
        self.inner.wait()
    }
}

impl Once {
    /// Create a new once.
    pub const fn new() -> Self {
        Self {
            inner: OnceLock::new(),
        }
    }

    /// Call the given closure once.
    pub fn call_once<F>(&self, f: F)
    where
        F: FnOnce(),
    {
        self.inner.get_or_init(|| {
            f();
        });
    }

    /// Check if the closure has been called.
    pub fn is_completed(&self) -> bool {
        self.inner.get().is_some()
    }
}

impl Default for Once {
    fn default() -> Self {
        Self::new()
    }
}

impl WaitGroup {
    /// Create a new wait group.
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
            condvar: Condvar::new(),
            mutex: Mutex::new(()),
        }
    }

    /// Add to the wait group counter.
    pub fn add(&self, delta: u64) {
        self.counter.fetch_add(delta, Ordering::AcqRel);
    }

    /// Mark one task as done.
    pub fn done(&self) {
        let prev = self.counter.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            // Last task completed, notify all waiters
            self.condvar.notify_all();
        }
    }

    /// Wait for all tasks to complete.
    pub fn wait(&self) {
        let mut guard = self.mutex.lock().unwrap();
        while self.counter.load(Ordering::Acquire) > 0 {
            guard = self.condvar.wait(guard).unwrap();
        }
    }

    /// Get the current counter value.
    pub fn count(&self) -> u64 {
        self.counter.load(Ordering::Acquire)
    }
}

impl Default for WaitGroup {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomicCounter {
    /// Create a new atomic counter.
    pub const fn new(value: u64) -> Self {
        Self {
            inner: AtomicU64::new(value),
        }
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.inner.load(Ordering::Acquire)
    }

    /// Set the value.
    pub fn set(&self, value: u64) {
        self.inner.store(value, Ordering::Release);
    }

    /// Increment the counter.
    pub fn increment(&self) -> u64 {
        self.inner.fetch_add(1, Ordering::AcqRel)
    }

    /// Decrement the counter.
    pub fn decrement(&self) -> u64 {
        self.inner.fetch_sub(1, Ordering::AcqRel)
    }

    /// Add to the counter.
    pub fn add(&self, value: u64) -> u64 {
        self.inner.fetch_add(value, Ordering::AcqRel)
    }

    /// Subtract from the counter.
    pub fn sub(&self, value: u64) -> u64 {
        self.inner.fetch_sub(value, Ordering::AcqRel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_mutex() {
        let mutex = Mutex::new(0);
        {
            let mut guard = mutex.lock().unwrap();
            *guard = 42;
        }
        assert_eq!(*mutex.lock().unwrap(), 42);
    }

    #[test]
    fn test_rwlock() {
        let lock = RwLock::new(0);
        {
            let mut guard = lock.write().unwrap();
            *guard = 42;
        }
        {
            let guard = lock.read().unwrap();
            assert_eq!(*guard, 42);
        }
    }

    #[test]
    fn test_atomic_counter() {
        let counter = AtomicCounter::new(0);
        assert_eq!(counter.get(), 0);
        
        counter.increment();
        assert_eq!(counter.get(), 1);
        
        counter.decrement();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_wait_group() {
        let wg = Arc::new(WaitGroup::new());
        
        wg.add(2);
        
        let wg1 = wg.clone();
        let handle1 = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            wg1.done();
        });
        
        let wg2 = wg.clone();
        let handle2 = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            wg2.done();
        });
        
        wg.wait();
        
        handle1.join().unwrap();
        handle2.join().unwrap();
        
        assert_eq!(wg.count(), 0);
    }

    #[test]
    fn test_once() {
        let once = Once::new();
        let mut called = false;
        
        once.call_once(|| {
            called = true;
        });
        
        assert!(called);
        assert!(once.is_completed());
        
        // Second call should not execute
        once.call_once(|| {
            panic!("Should not be called");
        });
    }
}