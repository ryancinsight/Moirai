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
#![allow(clippy::new_without_default)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::borrowed_box)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::unused_enumerate_index)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::new_ret_no_self)]
#![allow(clippy::single_match_else)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::manual_div_ceil)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// Platform abstraction layer
pub mod platform;

// Global constants (SSOT principle)
pub mod constants;

// Core modules
pub mod dtype;
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
pub use scheduler::{ScheduledTask, Scheduler, SchedulerConfig, SchedulerId};
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

// Re-export unified data type traits
pub use dtype::{
    ComputeContext, DefaultFloat, DefaultInt, DefaultUint, Dtype, FloatDtype, IntegerDtype,
};

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

// Platform-specific re-exports
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use wasm_executor::{WasmExecutor, WasmTask};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::{
        ExecutorError, Priority, ScheduledTask, Scheduler, SchedulerError, SchedulerId, Task,
        TaskContext, TaskError, TaskExt, TaskFuture, TaskId, TaskManager, TaskSpawner, TaskStatus,
    };

    #[cfg(feature = "std")]
    pub use crate::{
        mpmc, spsc, unbounded, Channel, ChannelError, MpmcReceiver, MpmcSender, SpscReceiver,
        SpscSender,
    };

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub use crate::{WasmExecutor, WasmTask};
}
