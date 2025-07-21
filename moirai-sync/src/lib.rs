//! Synchronization primitives for Moirai concurrency library.

/// A mutual exclusion primitive.
pub struct Mutex<T> {
    _phantom: std::marker::PhantomData<T>,
}

/// A reader-writer lock.
pub struct RwLock<T> {
    _phantom: std::marker::PhantomData<T>,
}

/// A condition variable.
pub struct Condvar {
    // Placeholder
}

/// A barrier for synchronizing multiple threads.
pub struct Barrier {
    // Placeholder
}

/// A one-time initialization primitive.
pub struct Once {
    // Placeholder
}

/// An atomic counter.
pub struct AtomicCounter {
    // Placeholder
}

/// A wait group for waiting on multiple tasks.
pub struct WaitGroup {
    // Placeholder
}

impl<T> Mutex<T> {
    /// Create a new mutex.
    pub fn new(_value: T) -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T> RwLock<T> {
    /// Create a new reader-writer lock.
    pub fn new(_value: T) -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl Condvar {
    /// Create a new condition variable.
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for Condvar {
    fn default() -> Self {
        Self::new()
    }
}

impl Barrier {
    /// Create a new barrier.
    pub fn new(_n: usize) -> Self {
        Self {}
    }
}

impl Once {
    /// Create a new once.
    pub const fn new() -> Self {
        Self {}
    }
}

impl AtomicCounter {
    /// Create a new atomic counter.
    pub fn new(_value: u64) -> Self {
        Self {}
    }
}

impl WaitGroup {
    /// Create a new wait group.
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for WaitGroup {
    fn default() -> Self {
        Self::new()
    }
}