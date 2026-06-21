//! High-performance synchronization primitives for Moirai concurrency library.
//!
//! This module provides specialized synchronization primitives that add value
//! beyond the standard library, following YAGNI and DRY principles.

#![allow(clippy::new_without_default)]
#![allow(clippy::manual_hash_one)]

pub mod sync;

// Re-export standard library primitives directly (DRY principle)
pub use std::sync::{
    Barrier, Condvar, Mutex, MutexGuard, OnceLock as Once, RwLock, RwLockReadGuard,
    RwLockWriteGuard,
};

// Re-export LockFreeStack from moirai-core to maintain DRY principle
pub use moirai_core::pool::LockFreeStack;

// Re-export sync submodule items directly at crate root
pub use self::sync::{
    AtomicCounter, ConcurrentHashMap, FutexMutex, FutexMutexGuard, SpinLock, SpinLockGuard,
    WaitGroup, SizeBounded, ShardedResourcePool,
};
