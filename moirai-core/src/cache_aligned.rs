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
/// 
/// This implementation ensures the total size is always a multiple of CACHE_LINE_SIZE
/// to maintain proper alignment in arrays.
#[repr(C, align(64))]
pub struct CachePadded<T> {
    data: T,
}

impl<T> CachePadded<T> {
    /// Create a new cache-padded value
    pub fn new(data: T) -> Self {
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
    
    #[test]
    fn test_cache_padded_alignment() {
        // Test with various sizes to ensure correct padding
        struct Small { _x: u8 }
        struct Medium { _x: [u8; 32] }
        struct Large { _x: [u8; 64] }
        struct ExtraLarge { _x: [u8; 128] }
        
        // All CachePadded types should be aligned to cache line
        assert_eq!(align_of::<CachePadded<Small>>(), CACHE_LINE_SIZE);
        assert_eq!(align_of::<CachePadded<Medium>>(), CACHE_LINE_SIZE);
        assert_eq!(align_of::<CachePadded<Large>>(), CACHE_LINE_SIZE);
        assert_eq!(align_of::<CachePadded<ExtraLarge>>(), CACHE_LINE_SIZE);
        
        // Size should be at least the cache line size
        assert!(size_of::<CachePadded<Small>>() >= CACHE_LINE_SIZE);
        assert!(size_of::<CachePadded<Medium>>() >= CACHE_LINE_SIZE);
        assert!(size_of::<CachePadded<Large>>() >= CACHE_LINE_SIZE);
        assert!(size_of::<CachePadded<ExtraLarge>>() >= CACHE_LINE_SIZE);
        
        // Test array alignment
        let padded_array: [CachePadded<AtomicUsize>; 4] = [
            CachePadded::new(AtomicUsize::new(0)),
            CachePadded::new(AtomicUsize::new(1)),
            CachePadded::new(AtomicUsize::new(2)),
            CachePadded::new(AtomicUsize::new(3)),
        ];
        
        // Each element should be on its own cache line
        for i in 1..padded_array.len() {
            let addr1 = &padded_array[i-1] as *const _ as usize;
            let addr2 = &padded_array[i] as *const _ as usize;
            assert_eq!(addr2 - addr1, size_of::<CachePadded<AtomicUsize>>());
            assert!(addr2 - addr1 >= CACHE_LINE_SIZE);
        }
    }
}