//! # Hybrid Executor Implementation
//!
//! This module provides a high-performance hybrid executor that seamlessly combines
//! asynchronous and parallel execution models in a unified runtime system.
//!
//! ## Architecture Overview
//!
//! The `HybridExecutor` is built on three core principles:
//! - **Work-Stealing Scheduler**: Intelligent load balancing across CPU cores
//! - **Adaptive Thread Pools**: Separate pools for async I/O and CPU-bound work
//! - **Zero-Copy Task Passing**: Minimal overhead task distribution
//!
//! ## Design Principles
//!
//! - **SOLID**: Each component has a single responsibility and clear interfaces
//! - **CUPID**: Composable, predictable, and domain-centric design
//! - **GRASP**: Information expert pattern with low coupling
//! - **Zero-cost abstractions**: Compile-time optimizations
//! - **Memory safety**: Rust ownership model prevents data races

// Module declarations - following SRP and SOC principles
pub mod hybrid;
pub mod metrics;
pub mod reactor;
pub mod registry;
pub mod task;
pub mod types;
pub mod worker;

// Re-export key types for clean API
pub use hybrid::HybridExecutor;
pub use metrics::ExecutorMetrics;
pub use registry::TaskRegistry;
pub use task::{TaskMetadata, TaskPerformanceMetrics, TaskWaitFuture};
pub use types::{IoEvent, WorkerId};
pub use worker::{Worker, WorkerMetrics};

// Essential imports for any remaining implementation
use std::sync::atomic::{AtomicBool, Ordering};

/// Main executor builder for creating configured instances
pub struct ExecutorBuilder {
    worker_threads: usize,
    async_threads: usize,
    blocking_threads: Option<usize>,
}

impl ExecutorBuilder {
    /// Create a new executor builder with default settings
    pub fn new() -> Self {
        Self {
            worker_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            async_threads: 4,
            blocking_threads: None,
        }
    }

    /// Set the number of worker threads
    pub fn worker_threads(mut self, count: usize) -> Self {
        self.worker_threads = count;
        self
    }

    /// Set the number of async threads
    pub fn async_threads(mut self, count: usize) -> Self {
        self.async_threads = count;
        self
    }

    /// Set the number of blocking threads
    pub fn blocking_threads(mut self, count: usize) -> Self {
        self.blocking_threads = Some(count);
        self
    }

    /// Build the hybrid executor
    pub fn build(self) -> Result<HybridExecutor, Box<dyn std::error::Error>> {
        // Create a basic configuration
        // In practice, this would use the moirai_core::executor::ExecutorConfig
        let config = moirai_core::executor::ExecutorConfig::default();
        HybridExecutor::new(config).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
}

impl Default for ExecutorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
