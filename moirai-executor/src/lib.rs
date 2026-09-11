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

/// A task entry point re-typed so it carries no raw pointer in its signature.
///
/// Melinoe hands the bridge `unsafe fn(usize, *mut ())`. A bare function
/// pointer is `Send + Sync`, but *that* signature mentions `*mut ()`, which is
/// neither — so a closure capturing the original binding fails the pool's
/// `Send + Sync` bound even though the pointer it names is only ever supplied
/// at the call site. Erasing the parameter type here keeps the pool bound
/// satisfiable while preserving the ABI: the value is called only after being
/// cast back to the original signature.
#[derive(Copy, Clone)]
struct TaskFn(unsafe fn(usize, *mut ()));

// SAFETY: a function pointer is `Send + Sync`; the only thing that made the
// original type fail those bounds was the *mention* of `*mut ()` in its
// parameter list, not any pointer value this wrapper holds. Nothing is
// dereferenced through `TaskFn`; it is transmuted back before use.
unsafe impl Send for TaskFn {}
unsafe impl Sync for TaskFn {}

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

impl TaskFn {
    /// Invoke the wrapped entry point.
    ///
    /// # Safety
    ///
    /// The caller must uphold whatever contract the original function required:
    /// `index` must be one of the indices the executor promised, and `context`
    /// must point at a live value of the type the original task expected.
    #[inline]
    unsafe fn call(self, index: usize, context: *mut ()) {
        // SAFETY: `self.0` was created from a value of exactly this type, so the
        // cast is the identity on the ABI; the caller supplies the contract.
        let task: unsafe fn(usize, *mut ()) = self.0;
        unsafe { task(index, context) }
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
    unsafe fn run_indexed(
        &self,
        num_tasks: usize,
        task: unsafe fn(usize, *mut ()),
        context: *mut (),
    ) {
        // Bind the type-erased pieces into `Send + Sync` wrappers *before* the
        // closure exists, so no raw pointer is ever in its capture set. The
        // pool's closure bound is `Fn + Send + Sync`, and a closure that names a
        // `*mut ()` in scope fails it even when the pointer is only forwarded.
        let task = TaskFn(task);
        let context = SendContext(context);
        let res = global().for_each_indexed::<SyncTask, _>(num_tasks, move |index| {
            // SAFETY: forwarded from the caller. `for_each_indexed` invokes this
            // closure exactly once per index and never concurrently for the same
            // index, so each call addresses a distinct slot of Melinoe's context.
            unsafe {
                task.call(index, context.get());
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
        let exec = std::sync::Arc::new(
            ExecutorBuilder::new()
                .build()
                .expect("initialize global Moirai executor"),
        );
        // Register the global parallel executor in melinoe.
        melinoe::register_parallel_executor::<MoiraiExecutor>();
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
