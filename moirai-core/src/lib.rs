//! # Moirai Core
//!
//! Core abstractions and traits for the Moirai concurrency library.
//!
//! This crate provides the fundamental building blocks that all other
//! Moirai crates build upon, including task abstractions, executor traits,
//! and scheduling interfaces.
//!
//! With default features disabled, the crate uses `core` and `alloc`. Host
//! capabilities such as threads, clocks, task completion handles, executor
//! lifecycle traits, and automatic topology selection require the `std`
//! feature; the allocation-backed task, scheduler, memory, and pool vocabulary
//! remains available.

#![deny(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::new_without_default)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::borrowed_box)]
#![allow(clippy::new_ret_no_self)]

extern crate alloc;
#[cfg(test)]
extern crate std;

// Platform abstraction layer
pub mod platform;

// Core modules
pub mod error;
pub mod executor;
pub mod memory;
pub mod pool;
pub mod scheduler;
pub mod task;

// Unified channel implementation
#[cfg(feature = "std")]
pub mod channel;

// CacheAligned is imported from moirai-utils

#[cfg(feature = "coroutine")]
pub mod coroutine;

#[cfg(feature = "std")]
pub mod communication;

#[cfg(all(any(unix, windows), feature = "std"))]
pub mod ipc;

// Core type definitions
pub use error::{ExecutorError, PlacementFailure, SchedulerError, TaskError};
pub use executor::{ExecutorConfig, WorkerPlacement};
#[cfg(feature = "std")]
pub use executor::{TaskManager, TaskSpawner, TaskStatus};
pub use scheduler::SchedulerId;
#[cfg(feature = "std")]
pub use task::TaskHandle;
pub use task::{Priority, Task, TaskBuilder, TaskContext, TaskExt, TaskFuture, TaskId};

// `channel::unbounded` is deliberately not re-exported here or in `prelude`:
// an unbounded queue turns a slow consumer into unbounded memory growth, so
// the short, discoverable names stay bounded and the unbounded constructor is
// reachable only through its fully qualified path.
#[cfg(feature = "std")]
pub use channel::{
    Channel, ChannelConfig, ChannelError, ChannelStatistics, Consumer, DEFAULT_CHANNEL_CAPACITY,
    MpmcChannel, MpmcReceiver, MpmcSender, Producer, Select, SpscConsumer, SpscProducer,
    SpscReceiver, SpscRing, SpscSender, UnifiedReceiver, UnifiedSender, mpmc, spsc,
    unified_channel, unified_channel_with_config,
};

// Re-export CacheAligned from moirai-utils for convenience
pub use moirai_utils::CacheAligned;

#[cfg(feature = "coroutine")]
pub use coroutine::{
    Coroutine, CoroutineExt, CoroutineFuture, CoroutineIterator, CoroutineResult, CoroutineState,
    FunctionCoroutine,
};

// Re-export platform types for convenience
pub use platform::{Arc, Box, String, Vec};

/// Type alias for boxed errors.
pub type BoxError = Box<dyn core::error::Error + Send + Sync>;

/// Type alias for results with boxed errors.
pub type Result<T> = core::result::Result<T, BoxError>;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::{
        ExecutorError, Priority, SchedulerError, SchedulerId, Task, TaskContext, TaskError,
        TaskExt, TaskFuture, TaskId,
    };

    #[cfg(feature = "std")]
    pub use crate::{
        Channel, ChannelError, Consumer, DEFAULT_CHANNEL_CAPACITY, MpmcReceiver, MpmcSender,
        Producer, SpscConsumer, SpscProducer, SpscReceiver, SpscRing, SpscSender, TaskManager,
        TaskSpawner, TaskStatus, mpmc, spsc,
    };
}
