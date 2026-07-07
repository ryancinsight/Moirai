use super::stack::LockFreeStack;

/// Global object pool for cross-thread sharing.
///
/// A lock-free stack of reusable objects shared by all threads; there is no
/// per-thread caching layer. [`Self::get`] pops a pooled object or constructs
/// a new one via `T::default()`; [`Self::put`] returns an object, dropping it
/// when the pool already holds `max_size` objects.
pub struct GlobalPool<T> {
    /// Global stack of available objects, sized to `max_size` slots.
    global: LockFreeStack<T>,
}

impl<T: Default + Send + 'static> GlobalPool<T> {
    /// Create a new global pool retaining at most `max_size` objects.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            global: LockFreeStack::with_capacity(max_size),
        }
    }

    /// Get an object from the pool, or construct a new one if the pool is empty.
    pub fn get(&self) -> T {
        self.global.pop().unwrap_or_default()
    }

    /// Return an object to the pool.
    ///
    /// When the pool already holds `max_size` objects the object is dropped;
    /// this is the pool's documented retention cap, not an error condition.
    pub fn put(&self, obj: T) {
        if let Err(obj) = self.global.push(obj) {
            drop(obj); // pool at capacity: intentional retention-cap drop
        }
    }

    /// Clear all objects from the pool.
    pub fn clear(&self) {
        while self.global.pop().is_some() {}
    }
}
