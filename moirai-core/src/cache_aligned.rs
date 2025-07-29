//! Cache-aligned data structures for optimal performance.
//!
//! This module provides cache-aligned wrappers to prevent false sharing
//! and improve cache efficiency in concurrent environments.

use std::ops::{Deref, DerefMut};
use std::mem::{size_of, align_of};

/// Standard cache line size for x86_64 and ARM64 processors
pub const CACHE_LINE_SIZE: usize = 64;

/// A cache-aligned wrapper that ensures data is aligned to cache line boundaries.
/// This prevents false sharing when multiple threads access adjacent data.
#[repr(align(64))]
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheAligned<T> {
    data: T,
}

impl<T> CacheAligned<T> {
    /// Create a new cache-aligned value
    pub const fn new(data: T) -> Self {
        Self { data }
    }
    
    /// Get a reference to the inner value
    pub fn get(&self) -> &T {
        &self.data
    }
    
    /// Get a mutable reference to the inner value
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }
    
    /// Extract the inner value
    pub fn into_inner(self) -> T {
        self.data
    }
}

impl<T> Deref for CacheAligned<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for CacheAligned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T> From<T> for CacheAligned<T> {
    fn from(data: T) -> Self {
        Self::new(data)
    }
}

/// A padded structure that ensures each element in an array is on its own cache line
#[repr(C)]
pub struct CachePadded<T> {
    data: T,
    _padding: [u8; CACHE_LINE_SIZE - (size_of::<T>() % CACHE_LINE_SIZE)],
}

impl<T> CachePadded<T> {
    /// Create a new cache-padded value
    pub fn new(data: T) -> Self {
        Self {
            data,
            _padding: [0; CACHE_LINE_SIZE - (size_of::<T>() % CACHE_LINE_SIZE)],
        }
    }
    
    /// Get a reference to the inner value
    pub fn get(&self) -> &T {
        &self.data
    }
    
    /// Get a mutable reference to the inner value
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<T> Deref for CachePadded<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for CachePadded<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// Compile-time assertion that a type is cache-aligned
#[macro_export]
macro_rules! assert_cache_aligned {
    ($type:ty) => {
        const _: () = {
            assert!(std::mem::align_of::<$type>() >= $crate::cache_aligned::CACHE_LINE_SIZE);
        };
    };
}

/// Helper trait for types that should be cache-aligned
pub trait CacheAlignedType: Sized {
    /// Wrap this type in a cache-aligned container
    fn cache_aligned(self) -> CacheAligned<Self> {
        CacheAligned::new(self)
    }
    
    /// Wrap this type in a cache-padded container
    fn cache_padded(self) -> CachePadded<Self> {
        CachePadded::new(self)
    }
}

// Implement for all types
impl<T> CacheAlignedType for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    #[test]
    fn test_cache_aligned() {
        let aligned = CacheAligned::new(42u64);
        assert_eq!(*aligned, 42);
        assert_eq!(align_of::<CacheAligned<u64>>(), CACHE_LINE_SIZE);
    }
    
    #[test]
    fn test_cache_padded() {
        let padded = CachePadded::new(100i32);
        assert_eq!(*padded, 100);
        assert!(size_of::<CachePadded<i32>>() >= CACHE_LINE_SIZE);
    }
    
    #[test]
    fn test_false_sharing_prevention() {
        // Create an array of cache-aligned atomics
        let counters: Vec<CacheAligned<AtomicUsize>> = (0..4)
            .map(|_| CacheAligned::new(AtomicUsize::new(0)))
            .collect();
        
        // Each counter should be on its own cache line
        for i in 1..counters.len() {
            let addr1 = &counters[i-1] as *const _ as usize;
            let addr2 = &counters[i] as *const _ as usize;
            assert!(addr2 - addr1 >= CACHE_LINE_SIZE);
        }
    }
}