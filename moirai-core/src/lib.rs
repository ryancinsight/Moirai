//! # Moirai Core
//!
//! Core abstractions and traits for the Moirai concurrency library.
//!
//! This crate provides the fundamental building blocks that all other
//! Moirai crates build upon, including task abstractions, executor traits,
//! and scheduling interfaces.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
// Temporarily allow these while fixing critical issues
#![allow(clippy::needless_continue)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::ptr_as_ptr)]
#![allow(clippy::used_underscore_binding)]
#![allow(clippy::unused_async)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// Platform abstraction layer
pub mod platform;

// Core modules
pub mod error;
pub mod executor;
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

#[cfg(all(unix, feature = "std"))]
pub mod ipc;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm_executor;

#[cfg(feature = "metrics")]
pub mod metrics;

#[cfg(feature = "std")]
pub mod security;

// pub mod hybrid; // Removed: Duplicate implementation, using moirai-executor::HybridExecutor instead

// Core type definitions
pub use error::{ExecutorError, SchedulerError, TaskError};
pub use executor::{ExecutorConfig, TaskManager, TaskSpawner, TaskStatus};
pub use scheduler::{Scheduler, SchedulerConfig, SchedulerId};
pub use task::{
    BoxedTask, Priority, Task, TaskBuilder, TaskContext, TaskExt, TaskFuture, TaskHandle, TaskId,
};

#[cfg(feature = "std")]
pub use channel::{
    mpmc, spsc, unbounded, Channel, ChannelError, MpmcChannel, MpmcReceiver, MpmcSender, Select,
    SpscChannel, SpscReceiver, SpscSender,
};

// Re-export CacheAligned from moirai-utils for convenience
pub use moirai_utils::CacheAligned;

#[cfg(feature = "coroutine")]
pub use coroutine::{
    Coroutine, CoroutineExt, CoroutineFuture, CoroutineIterator, CoroutineResult, CoroutineState,
    SimpleCoroutine,
};

// Re-export platform types for convenience
pub use platform::{Arc, Box, String, Vec};

/// Type alias for boxed errors.
pub type BoxError = Box<dyn core::error::Error + Send + Sync>;

/// Type alias for results with boxed errors.
pub type Result<T> = core::result::Result<T, BoxError>;

// Platform-specific re-exports
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use wasm_executor::{WasmExecutor, WasmTask};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::{
        ExecutorError, Priority, Scheduler, SchedulerError, SchedulerId, Task, TaskContext,
        TaskError, TaskExt, TaskFuture, TaskId, TaskManager, TaskSpawner, TaskStatus,
    };

    #[cfg(feature = "std")]
    pub use crate::{
        mpmc, spsc, unbounded, Channel, ChannelError, MpmcReceiver, MpmcSender, SpscReceiver,
        SpscSender,
    };

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub use crate::{WasmExecutor, WasmTask};
}
