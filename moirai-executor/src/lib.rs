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
#![deny(missing_docs)]

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
    IdleHook, IdleHookRegistrationError, MAX_IDLE_HOOKS, ProcessCount, ProcessId, ProcessRoute,
    RoutePolicy, RouteSummary, RouteTopology, ScheduleMetrics, SchedulerRoute, SchedulerScope,
    ServerCount, ServerId, ServerRoute, ServerRoutePolicy, SyncTask, ThreadId, ThreadRoute,
    ThreadRoutePolicy, ThreadScheduler, WorkClass, WorkerCount, register_idle_hook, run_idle_hooks,
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
            worker_threads: themis::CpuTopology::detect()
                .map(|topology| topology.logical_processors())
                .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
                .unwrap_or(4)
                .max(1),
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

/// Address-carrying wrapper that lets the bridge move a type-erased context
/// pointer into `Send` task closures.
///
/// The pointee is opaque here — Melinoe owns its type and guarantees it stays
/// live and unaliased for the whole call — so the only thing this wrapper
/// asserts is that *moving the address* to another worker is sound. That is
/// Melinoe's own obligation, discharged in `TaskContext`'s documentation.
#[derive(Copy, Clone)]
struct SendContext(*mut ());

// SAFETY: the pointed-to `TaskContext` is documented by Melinoe as valid for
// the whole executor call and as permitting concurrent field access from
// distinct tasks. Reading the address on another thread is therefore sound;
// this wrapper is moved and copied, never dereferenced by this crate.
unsafe impl Send for SendContext {}

// SAFETY: `Sync` is required because the closure that captures this wrapper is
// shared across the pool. Sharing the *address* is sound for the same reason as
// `Send` — the pointee is never accessed through this wrapper, and Melinoe
// guarantees concurrent access to distinct fields of its context is disjoint.
unsafe impl Sync for SendContext {}

impl SendContext {
    /// Recover the erased context pointer.
    ///
    /// Reading the field through a method rather than at the field keeps the
    /// `*mut ()` out of any closure's capture analysis: the closure moves a
    /// `SendContext` and calls a method, so the raw type never appears in its
    /// environment.
    #[inline]
    fn get(self) -> *mut () {
        self.0
    }
}

/// Moirai's implementation of Melinoe's parallel-executor contract.
///
/// Drives a partition's tasks on the shared work-stealing pool, so a branded
/// partition pays no OS-thread spawn. See [`melinoe::sync::ParallelExecutor`]
/// for the contract this discharges.
struct MoiraiExecutor;

// SAFETY: `global().for_each_indexed` owns the complete `0..num_tasks` domain
// and invokes its closure exactly once per index; it blocks until every
// scheduled invocation has completed, so the method satisfies both the
// "every index exactly once" and "no invocation outliving the return"
// obligations. On scheduler failure it panics *after* joining the scheduled
// invocations, which is the unwind path the contract permits — Melinoe's
// `ExecutorDropGuard` contains the omitted slots. The context pointer is
// forwarded unchanged and never outlives the blocking call.
unsafe impl melinoe::ParallelExecutor for MoiraiExecutor {
    unsafe fn run_indexed(num_tasks: usize, task: unsafe fn(usize, *mut ()), context: *mut ()) {
        // Bind the type-erased context into a `Send + Sync` wrapper before the
        // closure exists, so the raw pointer is absent from its capture set.
        // Function pointers already satisfy the pool's `Send + Sync` bound.
        let context = SendContext(context);
        let res = global().for_each_indexed::<SyncTask, _>(num_tasks, move |index| {
            // SAFETY: forwarded from the caller. `for_each_indexed` invokes this
            // closure exactly once per index and never concurrently for the same
            // index, so each call addresses a distinct slot of Melinoe's context.
            unsafe {
                task(index, context.get());
            }
        });
        if let Err(e) = res {
            panic!(
                "Moirai executor failure in Melinoe parallel driver: {:?}",
                e
            );
        }
    }
}

fn global_arc() -> &'static std::sync::Arc<HybridExecutor> {
    static GLOBAL_EXECUTOR: std::sync::OnceLock<std::sync::Arc<HybridExecutor>> =
        std::sync::OnceLock::new();
    GLOBAL_EXECUTOR.get_or_init(|| {
        let executor = std::sync::Arc::new(
            ExecutorBuilder::new()
                .build()
                .expect("initialize global Moirai executor"),
        );
        // Register after the pool exists so a re-entrant callback can never
        // observe a partially initialized scheduler.
        melinoe::register_parallel_executor::<MoiraiExecutor>();
        executor
    })
}

/// Initialize the shared executor and install its Melinoe partition bridge.
///
/// Call this during application startup when code may invoke
/// `melinoe::sync::partition_*` directly. The function is idempotent: the
/// scheduler is built once, and the process-global Melinoe slot is refreshed on
/// each call. Higher-level Moirai partition helpers initialize the same bridge
/// automatically when they enter the pool path.
pub fn initialize() {
    let _ = global_arc();
    // `clear_parallel_executor` is a supported lifecycle hook for tests and
    // integrations. Refresh the slot after such a reset without adding an
    // atomic store to every ordinary `global()` access.
    melinoe::register_parallel_executor::<MoiraiExecutor>();
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
