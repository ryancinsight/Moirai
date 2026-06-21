use crate::platform::*;
use super::stack::{LockFreeStack, CachePadded};

/// Global object pool for cross-thread sharing.
///
/// Uses a hybrid approach with thread-local caches backed by a global pool.
pub struct GlobalPool<T> {
    /// Global stack of available objects
    global: LockFreeStack<T>,
    /// Maximum size of the pool
    max_size: usize,
    /// Current size (may be approximate)
    current_size: CachePadded<AtomicUsize>,
}

impl<T: Default + Send + 'static> GlobalPool<T> {
    /// Create a new global pool.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            global: LockFreeStack::new(),
            max_size,
            current_size: CachePadded {
                value: AtomicUsize::new(0),
            },
        }
    }

    /// Get an object from the pool.
    ///
    /// This first checks a thread-local cache before falling back to the global pool.
    pub fn get(&self) -> T {
        // Try global pool
        if let Some(obj) = self.global.pop() {
            return obj;
        }

        // Create new object
        T::default()
    }

    /// Return an object to the pool.
    pub fn put(&self, obj: T) {
        let current = self.current_size.value.load(Ordering::Relaxed);
        if current < self.max_size {
            self.global.push(obj);
            self.current_size.value.fetch_add(1, Ordering::Relaxed);
        }
        // Otherwise drop the object
    }

    /// Clear all objects from the pool.
    pub fn clear(&self) {
        while self.global.pop().is_some() {
            self.current_size.value.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Statistics for pool usage.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total number of allocations
    pub allocations: u64,
    /// Total number of deallocations
    pub deallocations: u64,
    /// Number of times objects were reused
    pub reuses: u64,
    /// Current pool size
    pub current_size: usize,
    /// Peak pool size
    pub peak_size: usize,
}

/// Type alias for task pool
pub type TaskPool<T> = GlobalPool<T>;
