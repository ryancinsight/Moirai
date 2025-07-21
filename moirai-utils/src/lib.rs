//! Utility functions and data structures for Moirai concurrency library.

#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

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

/// CPU topology information.
#[cfg(feature = "numa")]
pub mod numa {
    //! NUMA topology detection and management.
    
    /// NUMA node identifier.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NumaNode(pub u32);
    
    /// CPU core identifier.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct CpuCore(pub u32);
    
    /// Get the current NUMA node.
    pub fn current_numa_node() -> NumaNode {
        // Placeholder implementation
        NumaNode(0)
    }
    
    /// Get the number of NUMA nodes.
    pub fn numa_node_count() -> usize {
        // Placeholder implementation
        1
    }
}