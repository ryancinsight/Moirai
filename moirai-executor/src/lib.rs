//! # Hybrid Executor Implementation
//!
//! This crate provides a high-performance hybrid executor that combines
//! synchronous, asynchronous, and blocking execution on **one** unified
//! scheduler facade. Synchronous and async-ready work use the compute
//! work-stealing pool; potentially blocking work uses a lazily initialized,
//! bounded lane owned by that scheduler.
//!
//! ## Architecture Overview
//!
//! - **Static Work-Class Routing**: sync, async, and blocking jobs are routed
//!   by zero-sized work-class markers; blocking admission is isolated from the
//!   compute worker pool.
//! - **Priority-Partitioned Queues**: per-worker Chase-Lev deques indexed by
//!   [`moirai_core::Priority::index`].
//! - **Zero-Copy Task Passing**: minimal overhead task distribution.

#![cfg_attr(nightly_tls_active, feature(thread_local))]

// Module declarations - following SRP and SOC principles
pub mod hybrid;
pub mod metrics;
pub mod registry;
pub mod schedule;
pub mod task;

// Re-export key types for clean API
pub use hybrid::HybridExecutor;
pub use metrics::ExecutorMetrics;
pub use registry::TaskRegistry;
pub use schedule::{
    AcceleratorCounts, AcceleratorId, AcceleratorKind, AcceleratorRoute, AcceleratorRoutePolicy,
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
pub use task::TaskMetadata;

/// Block the current thread until `future` resolves.
///
/// This is the Moirai-owned synchronous wait primitive for code that only needs
/// to bridge an async operation into a synchronous boundary. It uses the same
/// parking waker as [`moirai_core::executor::ExecutorControl::block_on`] without constructing or
/// touching the process-wide scheduler.
pub fn block_on<F>(future: F) -> F::Output
where
    F: core::future::Future,
{
    schedule::wake::block_on_current_thread(future)
}

/// Main executor builder for creating configured instances
pub struct ExecutorBuilder {
    worker_threads: usize,
    async_threads: usize,
}

impl ExecutorBuilder {
    /// Create a new executor builder with default settings
    pub fn new() -> Self {
        Self {
            worker_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            async_threads: 4,
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

/// Address-carrying wrapper that lets the melinoe bridge move a raw data
/// pointer into `Send` task closures; safety is owed by the bridge caller.
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

// SAFETY: on success, `for_each_indexed` owns the complete `0..num_tasks`
// domain and invokes its closure once per index. On scheduler failure, the
// bridge panics after `for_each_indexed` has joined every scheduled invocation;
// Melinoe's unwind guard handles omitted slots. The unchanged context pointer
// never outlives the blocking scheduler call.
const MELINOE_EXECUTOR: melinoe::ParallelExecutor =
    unsafe { melinoe::ParallelExecutor::new(melinoe_executor_bridge) };

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
        melinoe::register_parallel_executor(MELINOE_EXECUTOR);
        exec
    })
}

/// Borrow the shared, lazily-initialized process-wide executor.
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
