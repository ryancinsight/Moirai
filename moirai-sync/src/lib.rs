//! High-performance synchronization primitives for Moirai concurrency library.
//!
//! This module provides specialized synchronization primitives that add value
//! beyond the standard library, following YAGNI and DRY principles.

#![deny(missing_docs)]
#![allow(clippy::new_without_default)]
#![allow(clippy::manual_hash_one)]

/// Synchronization primitives: sharded maps and pools, spin locks,
/// futex-backed mutexes, and completion barriers.
pub mod sync;

// Re-export the canonical AtomicCounter from moirai-utils (SSOT)
pub use moirai_utils::AtomicCounter;

// Re-export standard library primitives directly (DRY principle)
pub use std::sync::{
    Barrier, Condvar, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard,
};

// Re-export LockFreeStack from moirai-core to maintain DRY principle
pub use moirai_core::pool::LockFreeStack;

// Re-export sync submodule items directly at crate root
pub use self::sync::{
    ConcurrentHashMap, FutexMutex, FutexMutexGuard, SegmentPoisoned, ShardedResourcePool,
    SizeBounded, SpinLock, SpinLockGuard, WaitGroup,
};
