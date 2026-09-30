//! Configuration settings for executor behavior.

use crate::platform::String;

use super::placement::WorkerPlacement;

// Memory pool size constants
const KILOBYTE: usize = 1024;
const MEGABYTE: usize = 1024 * KILOBYTE;
/// Default capacity for the small object allocation pool.
pub const SMALL_POOL_SIZE: usize = 64 * KILOBYTE;
/// Default capacity for the medium object allocation pool.
pub const MEDIUM_POOL_SIZE: usize = MEGABYTE;
/// Default capacity for the large object allocation pool.
pub const LARGE_POOL_SIZE: usize = 16 * MEGABYTE;

/// Default aggregate bound for worker admission queues (tasks, not bytes).
/// Sized for burst absorption across all workers before producers observe
/// backpressure.
pub const DEFAULT_GLOBAL_QUEUE_CAPACITY: usize = 8192;
/// Default initial slot count for each worker's resizable local priority queue.
///
/// This is a retained-storage policy, not an admission bound. The Chase-Lev
/// queues grow when full and normalize this value to a supported power of two.
pub const DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY: usize = 128;

/// Configuration settings for executor behavior and performance characteristics.
///
/// This struct encapsulates all tunable parameters that affect executor operation,
/// including thread pool sizes, queue capacities, and various performance optimizations.
#[allow(clippy::module_name_repetitions)]
pub struct ExecutorConfig {
    /// Number of worker threads for parallel tasks
    pub worker_threads: usize,
    /// Number of threads dedicated to async tasks
    pub async_threads: usize,
    /// Maximum aggregate size of the workers' external admission queues.
    ///
    /// Executor construction partitions this bound across workers without
    /// exceeding it. The value must supply at least two slots per worker.
    pub max_global_queue_size: usize,
    /// Initial slot count for each resizable per-worker local priority queue.
    ///
    /// Values below the deque minimum normalize upward. Local queues grow when
    /// full; [`Self::max_global_queue_size`] is the external admission bound.
    pub local_queue_initial_capacity: usize,
    /// Thread name prefix for worker threads
    pub thread_name_prefix: String,
    /// Whether each worker is confined to one logical processor.
    ///
    /// [`WorkerPlacement::Pinned`] can fail executor construction; see
    /// [`ExecutorError::WorkerPlacementFailed`](crate::error::ExecutorError::WorkerPlacementFailed).
    pub worker_placement: WorkerPlacement,
    /// Whether to enable metrics collection
    #[cfg(feature = "metrics")]
    pub enable_metrics: bool,
    /// Task preemption configuration
    pub preemption: PreemptionConfig,
    /// Memory management configuration
    pub memory: MemoryConfig,
    /// Task cleanup configuration
    pub cleanup: CleanupConfig,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            worker_threads: super::logical_parallelism(),
            async_threads: (super::logical_parallelism() / 4).max(1),
            max_global_queue_size: DEFAULT_GLOBAL_QUEUE_CAPACITY,
            local_queue_initial_capacity: DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY,
            thread_name_prefix: "moirai-worker".into(),
            worker_placement: WorkerPlacement::default(),
            #[cfg(feature = "metrics")]
            enable_metrics: true,
            preemption: PreemptionConfig::default(),
            memory: MemoryConfig::default(),
            cleanup: CleanupConfig::default(),
        }
    }
}

/// Configuration for task preemption.
#[derive(Debug, Clone)]
pub struct PreemptionConfig {
    /// Whether to enable cooperative preemption
    pub enabled: bool,
    /// Time slice for each task before preemption (microseconds)
    pub time_slice_us: u64,
    /// Whether to preempt based on priority
    pub priority_based: bool,
    /// Minimum execution time before preemption (microseconds)
    pub min_execution_time_us: u64,
}

impl Default for PreemptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            time_slice_us: 10_000, // 10ms
            priority_based: true,
            min_execution_time_us: 1_000, // 1ms
        }
    }
}

/// Configuration for memory management.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Whether to use memory pools
    pub use_memory_pools: bool,
    /// Size of small object pool (bytes)
    pub small_pool_size: usize,
    /// Size of medium object pool (bytes)
    pub medium_pool_size: usize,
    /// Size of large object pool (bytes)
    pub large_pool_size: usize,
    /// Whether to track memory usage per task
    pub track_per_task_memory: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            use_memory_pools: true,
            small_pool_size: SMALL_POOL_SIZE,
            medium_pool_size: MEDIUM_POOL_SIZE,
            large_pool_size: LARGE_POOL_SIZE,
            track_per_task_memory: cfg!(feature = "metrics"),
        }
    }
}

/// Configuration for completed-task retention.
///
/// The executor releases the state of finished tasks in whole blocks of 1,024
/// tasks as it registers new ones, so retained memory follows the spawn rate
/// without a background thread. A block is released once every task in it has
/// finished and either its newest completion is older than
/// `task_retention_duration` or more than `max_retained_tasks` finished tasks
/// are retained. One long-running task therefore keeps its own block resident.
///
/// A released task still reports as completed to `wait_for_task` and accepts
/// `cancel_task` as a no-op; only its status and statistics are gone, so
/// `task_status` and `task_stats` return `None`.
#[derive(Debug, Clone)]
pub struct CleanupConfig {
    /// How long finished-task metadata stays observable.
    ///
    /// # Default: 5 minutes
    pub task_retention_duration: core::time::Duration,

    /// Whether the executor releases finished-task metadata at all.
    ///
    /// When disabled every task's metadata is retained for the executor's
    /// lifetime and memory grows with the number of tasks spawned.
    /// # Default: true
    pub enable_automatic_cleanup: bool,

    /// Most finished tasks retained regardless of age, rounded up to whole
    /// blocks of 1,024 tasks.
    ///
    /// This is the hard bound on retained memory for a workload that finishes
    /// tasks faster than `task_retention_duration` elapses.
    /// # Default: 10,000 tasks
    pub max_retained_tasks: usize,
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            task_retention_duration: core::time::Duration::from_mins(5),
            enable_automatic_cleanup: true,
            max_retained_tasks: 10_000,
        }
    }
}
