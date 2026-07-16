//! # Moirai Core
//!
//! Core abstractions and traits for the Moirai concurrency library.
//!
//! This crate provides the fundamental building blocks that all other
//! Moirai crates build upon, including task abstractions, executor traits,
//! and scheduling interfaces.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(nightly_tls_active, feature(thread_local))]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::ptr_as_ptr)]
#![allow(clippy::used_underscore_binding)]
#![allow(clippy::unused_async)]
// TODO(D9): 56 `missing_errors_doc` + 25 `missing_panics_doc` sites pending —
// documentation task exceeding this sweep's bound; tracked for its own effort.
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::new_without_default)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::borrowed_box)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::new_ret_no_self)]
#![allow(clippy::single_match_else)]

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
#[cfg(feature = "std")]
pub mod unified_channel;

// CacheAligned is imported from moirai-utils

#[cfg(feature = "coroutine")]
pub mod coroutine;

#[cfg(feature = "std")]
pub mod communication;

#[cfg(all(any(unix, windows), feature = "std"))]
pub mod ipc;

// pub mod hybrid; // Removed: Duplicate implementation, using moirai-executor::HybridExecutor instead

// Core type definitions
pub use error::{ExecutorError, SchedulerError, TaskError};
pub use executor::{ExecutorConfig, TaskManager, TaskSpawner, TaskStatus};
pub use scheduler::SchedulerId;
pub use task::{Priority, Task, TaskBuilder, TaskContext, TaskExt, TaskFuture, TaskHandle, TaskId};

#[cfg(feature = "std")]
pub use channel::{
    mpmc, spsc, unbounded, Channel, ChannelError, MpmcChannel, MpmcReceiver, MpmcSender, Select,
    SpscChannel, SpscReceiver, SpscSender,
};

#[cfg(feature = "std")]
pub use unified_channel::{
    unified_channel, unified_channel_with_config, ChannelConfig, ChannelStatistics,
    UnifiedChannelError, UnifiedReceiver, UnifiedSender,
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
        mpmc, spsc, unbounded, Channel, ChannelError, MpmcReceiver, MpmcSender, SpscReceiver,
        SpscSender,
    };
}
