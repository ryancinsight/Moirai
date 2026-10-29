//! Object pooling for efficient memory management.
//!
//! This module implements advanced pooling techniques inspired by:
//! - Tokio's slab allocator
//! - Lock-free stacks for thread-safe pooling
//! - Thread-local pools for single-thread hot paths
//!
//! The allocation-backed pools support `no_std`; `TaskWrapper` requires
//! `std` because its age is measured by the host monotonic clock.

mod global;
mod slab;
mod stack;
mod thread_local;
#[cfg(feature = "std")]
mod wrapper;

#[cfg(test)]
mod tests;

pub use global::GlobalPool;
pub use slab::SlabAllocator;
pub use stack::{CacheAligned, DEFAULT_STACK_CAPACITY, LockFreeStack};
pub use thread_local::ThreadLocalPool;
#[cfg(feature = "std")]
pub use wrapper::TaskWrapper;
