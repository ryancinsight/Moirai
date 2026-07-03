//! Object pooling for efficient memory management.
//!
//! This module implements advanced pooling techniques inspired by:
//! - Tokio's slab allocator
//! - Lock-free stacks for thread-safe pooling
//! - Thread-local pools for single-thread hot paths

mod global;
mod slab;
mod stack;
mod thread_local;
mod wrapper;

#[cfg(test)]
mod tests;

pub use global::GlobalPool;
pub use slab::SlabAllocator;
pub use stack::{CachePadded, LockFreeStack, DEFAULT_STACK_CAPACITY};
pub use thread_local::ThreadLocalPool;
pub use wrapper::TaskWrapper;
