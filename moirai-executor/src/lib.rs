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

#![allow(clippy::incompatible_msrv)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::manual_map)]
#![allow(clippy::type_complexity)]
#![cfg_attr(nightly_tls_active, feature(thread_local))]
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
pub mod schedule;
pub mod task;
pub mod types;

// Re-export key types for clean API
pub use hybrid::HybridExecutor;
pub use metrics::ExecutorMetrics;
pub use registry::TaskRegistry;
pub use schedule::{
    AsyncLaneId, AsyncLanesPerProcess, AsyncTask, BlockingTask, HybridRoutePolicy, HybridRouter,
    ProcessCount, ProcessId, ProcessRoute, RoutePolicy, RouteSummary, RouteTopology,
    ScheduleMetrics, SchedulerRoute, SchedulerScope, ServerCount, ServerId, ServerRoute,
    ServerRoutePolicy, SyncTask, ThreadId, ThreadRoute, ThreadRoutePolicy, ThreadScheduler,
    WorkClass, WorkerCount,
};
#[cfg(feature = "scheduler-diagnostics")]
pub use schedule::{
    ContendedWakeDecision, DiagnosticWakeDecision, EmptyWakeDecision, SaturatedWakeDecision,
};
pub use task::{TaskMetadata, TaskPerformanceMetrics, TaskWaitFuture};
pub use types::{IoEvent, WorkerId};

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
        let config = moirai_core::executor::ExecutorConfig {
            worker_threads: self.worker_threads,
            async_threads: self.async_threads,
            ..moirai_core::executor::ExecutorConfig::default()
        };
        HybridExecutor::new(config).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
}

impl Default for ExecutorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared, lazily-initialized process-wide executor.
///
/// Provides a single default runtime so higher-level crates (e.g.
/// `moirai-parallel`'s data-parallel primitives) can schedule work without each
/// constructing — and over-subscribing — their own thread pool. Built once with
/// the default [`ExecutorBuilder`] configuration on first access.
///
/// # Panics
///
/// Panics if the executor cannot be initialized, which should not happen under
/// normal conditions.
#[derive(Copy, Clone)]
struct SendPtr(usize);

unsafe fn melinoe_executor_bridge(
    num_tasks: usize,
    task_fn: unsafe fn(usize, *mut ()),
    data: *mut (),
) {
    let data_ptr = SendPtr(data as usize);
    let res = global().for_each_indexed::<SyncTask, _>(num_tasks, move |index| {
        let p = data_ptr;
        // SAFETY: task_fn is called concurrently on separate indices.
        unsafe {
            task_fn(index, p.0 as *mut ());
        }
    });
    if let Err(e) = res {
        panic!(
            "Moirai executor failure in Melinoe parallel driver: {:?}",
            e
        );
    }
}

fn global_arc() -> &'static std::sync::Arc<HybridExecutor> {
    static GLOBAL_EXECUTOR: std::sync::OnceLock<std::sync::Arc<HybridExecutor>> =
        std::sync::OnceLock::new();
    GLOBAL_EXECUTOR.get_or_init(|| {
        let exec = std::sync::Arc::new(
            ExecutorBuilder::new()
                .build()
                .expect("initialize global Moirai executor"),
        );
        // Register the global parallel executor in melinoe.
        melinoe::register_parallel_executor(melinoe_executor_bridge);
        exec
    })
}

/// Borrow the shared process-wide executor.
pub fn global() -> &'static HybridExecutor {
    global_arc()
}

/// Obtain an owned handle to the shared process-wide executor.
///
/// Higher layers (e.g. the `moirai` umbrella's global runtime) wrap this same
/// `Arc` so that parallel data-parallel work and async tasks run on **one**
/// unified hybrid scheduler rather than separate thread pools.
pub fn shared() -> std::sync::Arc<HybridExecutor> {
    std::sync::Arc::clone(global_arc())
}
