//! # Moirai Core
//!
//! Core abstractions and traits for the Moirai concurrency library.
//!
//! This crate provides the fundamental building blocks that all other
//! Moirai crates build upon, including task abstractions, executor traits,
//! and scheduling interfaces.

#![deny(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::new_without_default)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::borrowed_box)]
#![allow(clippy::new_ret_no_self)]

#[cfg(not(feature = "std"))]
extern crate alloc;

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
pub use error::{ExecutorError, SchedulerError, TaskError};
pub use executor::{ExecutorConfig, TaskManager, TaskSpawner, TaskStatus};
pub use scheduler::SchedulerId;
pub use task::{Priority, Task, TaskBuilder, TaskContext, TaskExt, TaskFuture, TaskHandle, TaskId};

// `channel::unbounded` is deliberately not re-exported here or in `prelude`:
// an unbounded queue turns a slow consumer into unbounded memory growth, so
// the short, discoverable names stay bounded and the unbounded constructor is
// reachable only through its fully qualified path.
#[cfg(feature = "std")]
pub use channel::{
    mpmc, spsc, unified_channel, unified_channel_with_config, Channel, ChannelConfig, ChannelError,
    ChannelStatistics, Consumer, MpmcChannel, MpmcReceiver, MpmcSender, Producer, Select,
    SpscConsumer, SpscProducer, SpscReceiver, SpscRing, SpscSender, UnifiedReceiver, UnifiedSender,
    DEFAULT_CHANNEL_CAPACITY,
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
        TaskExt, TaskFuture, TaskId, TaskManager, TaskSpawner, TaskStatus,
    };

    #[cfg(feature = "std")]
    pub use crate::{
        mpmc, spsc, Channel, ChannelError, Consumer, MpmcReceiver, MpmcSender, Producer,
        SpscConsumer, SpscProducer, SpscReceiver, SpscRing, SpscSender, DEFAULT_CHANNEL_CAPACITY,
    };
}
