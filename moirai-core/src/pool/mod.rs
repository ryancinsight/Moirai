//! Object pooling for efficient memory management.
//!
//! This module implements advanced pooling techniques inspired by:
//! - Tokio's slab allocator
//! - Lock-free stacks for thread-safe pooling
//! - Thread-local caching for hot paths

mod stack;
mod slab;
mod wrapper;
mod thread_local;
mod global;

#[cfg(test)]
mod tests;

pub use stack::{LockFreeStack, CachePadded};
pub use slab::SlabAllocator;
pub use wrapper::TaskWrapper;
pub use thread_local::ThreadLocalPool;
pub use global::{GlobalPool, PoolStats, TaskPool};
