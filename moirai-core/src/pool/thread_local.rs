use crate::platform::*;

/// Thread-local task pool for zero-allocation task execution
pub struct ThreadLocalPool<T> {
    /// Stack of available objects
    pool: UnsafeCell<Vec<T>>,
    /// Maximum pool size
    max_size: usize,
    /// Marker to ensure !Send and !Sync
    _marker: PhantomData<*const T>,
}

impl<T> ThreadLocalPool<T> {
    /// Create a new thread-local pool
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            pool: UnsafeCell::new(Vec::with_capacity(max_size)),
            max_size,
            _marker: PhantomData,
        }
    }

    /// Get an object from the pool or create a new one
    pub fn get_or_create<F>(&self, create: F) -> T
    where
        F: FnOnce() -> T,
    {
        // SAFETY: `ThreadLocalPool` is !Sync (and !Send) via its raw-pointer
        // phantom marker, so `&self` implies same-thread access; the cell is
        // therefore not aliased and unique `&mut` is sound.
        unsafe {
            let pool = &mut *self.pool.get();
            pool.pop().unwrap_or_else(create)
        }
    }

    /// Return an object to the pool
    pub fn put(&self, obj: T) {
        // SAFETY: same-thread uniqueness as in `get_or_create`; the cell is
        // never aliased on a !Sync type.
        unsafe {
            let pool = &mut *self.pool.get();
            if pool.len() < self.max_size {
                pool.push(obj);
            }
        }
    }
}
