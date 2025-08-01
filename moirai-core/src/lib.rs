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

#[cfg(not(feature = "std"))]
extern crate alloc;

// Re-exports for no-std compatibility
#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec, string::String};
#[cfg(not(feature = "std"))]
use core as std;

#[cfg(feature = "std")]
use std::sync::mpsc;
#[cfg(feature = "std")]
use std::time::Duration;

pub mod task;
pub mod executor;
pub mod scheduler;
pub mod error;
pub mod pool;
pub mod cache_aligned;

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

// Core type definitions
pub use task::{Task, TaskContext, TaskId, Priority, TaskFuture, TaskExt, BoxedTask};
pub use executor::{TaskSpawner, TaskManager, TaskMonitor, TaskStatus, ExecutorConfig};
pub use scheduler::{Scheduler, SchedulerId, Generic as GenericScheduler};
pub use error::{TaskError, ExecutorError, SchedulerError};

// Re-export commonly used types
pub use alloc::boxed::Box;
pub use alloc::vec::Vec;

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
        Task, TaskContext, TaskId, Priority, TaskFuture, TaskExt,
        TaskSpawner, TaskManager, TaskMonitor, TaskStatus,
        Scheduler, SchedulerId, 
        TaskError, ExecutorError, SchedulerError,
    };
    
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub use crate::{WasmExecutor, WasmTask};
}