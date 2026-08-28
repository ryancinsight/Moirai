//! Configuration settings for executor behavior.

use crate::platform::String;

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
/// Default bound for each worker's local queue (tasks, not bytes). Small so
/// idle workers can steal instead of one worker hoarding a deep queue.
pub const DEFAULT_LOCAL_QUEUE_CAPACITY: usize = 256;

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
    /// Maximum size of per-thread local queues
    pub max_local_queue_size: usize,
    /// Thread name prefix for worker threads
    pub thread_name_prefix: String,
    /// Whether to enable NUMA-aware thread placement
    #[cfg(feature = "numa")]
    pub numa_aware: bool,
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
            worker_threads: super::num_cpus(),
            async_threads: (super::num_cpus() / 4).max(1),
            max_global_queue_size: DEFAULT_GLOBAL_QUEUE_CAPACITY,
            max_local_queue_size: DEFAULT_LOCAL_QUEUE_CAPACITY,
            thread_name_prefix: "moirai-worker".into(),
            #[cfg(feature = "numa")]
            numa_aware: true,
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

/// Configuration for task metadata cleanup.
///
/// Controls how and when completed task metadata is removed from memory
/// to prevent memory leaks in long-running executors.
#[derive(Debug, Clone)]
pub struct CleanupConfig {
    /// How long to keep completed task metadata before cleanup
    ///
    /// # Default: 5 minutes
    /// # Range: 1 second to `task_retention_duration`
    pub task_retention_duration: core::time::Duration,

    /// How often to run the cleanup process
    ///
    /// # Default: 30 seconds  
    /// # Range: 1 second to `task_retention_duration`
    pub cleanup_interval: core::time::Duration,

    /// Whether to enable automatic cleanup
    ///
    /// If disabled, cleanup must be triggered manually via `cleanup_completed_tasks()`
    /// # Default: true
    pub enable_automatic_cleanup: bool,

    /// Maximum number of completed tasks to retain regardless of age
    ///
    /// This provides a hard limit to prevent unbounded memory growth
    /// # Default: 10,000 tasks
    pub max_retained_tasks: usize,
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            task_retention_duration: core::time::Duration::from_mins(5),
            cleanup_interval: core::time::Duration::from_secs(30), // 30 seconds
            enable_automatic_cleanup: true,
            max_retained_tasks: 10_000,
        }
    }
}
