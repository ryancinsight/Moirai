//! Object pooling for efficient memory management.
//!
//! This module implements advanced pooling techniques inspired by:
//! - Tokio's slab allocator
//! - Lock-free stacks for thread-safe pooling
//! - Thread-local caching for hot paths

mod global;
mod slab;
mod stack;
mod thread_local;
mod wrapper;

#[cfg(test)]
mod tests;

pub use global::{GlobalPool, PoolStats, TaskPool};
pub use slab::SlabAllocator;
pub use stack::{CachePadded, LockFreeStack};
pub use thread_local::ThreadLocalPool;
pub use wrapper::TaskWrapper;
