//! Thread scheduler construction and worker-set initialization.

use std::{
    sync::{atomic::AtomicUsize, Arc, Mutex, OnceLock},
    thread,
};

use moirai_core::{
    error::{ExecutorError, ExecutorResult},
    executor::{
        config::{DEFAULT_GLOBAL_QUEUE_CAPACITY, DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY},
        ExecutorConfig,
    },
};
use moirai_scheduler::DequeCapacity;
use moirai_utils::cache::CacheAligned;

use super::super::super::job::ScheduledJob;
use super::super::types::{SchedulerInner, ThreadScheduler};
use super::super::worker::lock_mutex;

struct SchedulerConstruction<'config> {
    worker_count: usize,
    thread_name_prefix: &'config str,
    max_global_queue_size: usize,
    local_queue_initial_capacity: usize,
    numa_aware: bool,
}

impl ThreadScheduler<256, 8192> {
    /// Start a scheduler with a compute worker set and a lazy blocking lane.
    ///
    /// # Errors
    ///
    /// Returns an error when the default queue policy cannot be constructed or
    /// a worker thread cannot be started.
    pub fn new(worker_count: usize, thread_name_prefix: &str) -> ExecutorResult<Self> {
        Self::new_with_local_queue_initial_capacity(
            worker_count,
            thread_name_prefix,
            DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY,
        )
    }
}

impl<const BLOCKING_QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>
    ThreadScheduler<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>
{
    /// Start a scheduler with a runtime-selected local queue initial capacity.
    ///
    /// The first const parameter remains the blocking lane's hard per-queue
    /// admission bound. `local_queue_initial_capacity` controls only initial
    /// retained storage for resizable compute-local priority queues.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutorError::InvalidLocalQueueInitialCapacity`] when the
    /// requested capacity cannot normalize or form the required scheduled-job
    /// deque allocation layouts, or propagates worker thread construction
    /// failures.
    pub fn new_with_local_queue_initial_capacity(
        worker_count: usize,
        thread_name_prefix: &str,
        local_queue_initial_capacity: usize,
    ) -> ExecutorResult<Self> {
        Self::from_construction(SchedulerConstruction {
            worker_count,
            thread_name_prefix,
            max_global_queue_size: DEFAULT_GLOBAL_QUEUE_CAPACITY,
            local_queue_initial_capacity,
            numa_aware: true,
        })
    }

    pub(crate) fn from_executor_config(
        config: &ExecutorConfig,
        numa_aware: bool,
    ) -> ExecutorResult<Self> {
        Self::from_construction(SchedulerConstruction {
            worker_count: config.worker_threads,
            thread_name_prefix: &config.thread_name_prefix,
            max_global_queue_size: config.max_global_queue_size,
            local_queue_initial_capacity: config.local_queue_initial_capacity,
            numa_aware,
        })
    }

    fn from_construction(config: SchedulerConstruction<'_>) -> ExecutorResult<Self> {
        let SchedulerConstruction {
            worker_count,
            thread_name_prefix,
            max_global_queue_size,
            local_queue_initial_capacity,
            numa_aware,
        } = config;
        let worker_count = worker_count.max(1);
        let injector_capacity = partition_global_queue(max_global_queue_size, worker_count)?;
        let local_queue_capacity = DequeCapacity::<ScheduledJob>::try_from(
            local_queue_initial_capacity,
        )
        .map_err(|error| ExecutorError::InvalidLocalQueueInitialCapacity {
            requested: error.requested(),
        })?;
        let mut queue_owners = Vec::with_capacity(worker_count);
        let workers = (0..worker_count)
            .map(|_| {
                let (owner, queues) = super::super::super::queue::WorkerQueues::new(
                    injector_capacity,
                    local_queue_capacity,
                );
                queue_owners.push(owner);
                Arc::new(super::super::types::WorkerState::new(queues))
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        // Detect NUMA topology once at construction; derive a per-worker node
        // assignment so `steal_job` can prefer same-node victims without runtime
        // discovery overhead.  Falls back to `None` on single-node / VM systems.
        let topology: Option<moirai_scheduler::numa::CpuTopology> = if numa_aware {
            #[cfg(miri)]
            {
                // Miri cannot execute the platform topology FFI. `None` is the
                // scheduler's normal no-topology fallback and leaves queue and
                // stealing semantics available to the interpreter.
                None
            }
            #[cfg(not(miri))]
            {
                moirai_scheduler::numa::CpuTopology::detect()
            }
        } else {
            None
        };
        let worker_numa_nodes: Box<[Option<usize>]> = if let Some(ref topo) = topology {
            (0..worker_count)
                .map(|id| {
                    // Use CPU core ID equal to worker ID (modular wrap on many-core
                    // systems so indices stay in-bounds regardless of worker count).
                    let core_id = id % topo.logical_cores.max(1);
                    topo.core_to_numa_node(core_id)
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        } else {
            vec![None; worker_count].into_boxed_slice()
        };

        let inner = Arc::new(SchedulerInner {
            workers,
            handles: std::sync::Mutex::new(Vec::with_capacity(worker_count)),
            external_handles: AtomicUsize::new(1),
            pending_tasks: CacheAligned::new(AtomicUsize::new(0)),
            active_workers: CacheAligned::new(AtomicUsize::new(0)),
            blocking_pending_tasks: CacheAligned::new(AtomicUsize::new(0)),
            blocking_active_workers: CacheAligned::new(AtomicUsize::new(0)),
            completed_tasks: CacheAligned::new(std::sync::atomic::AtomicU64::new(0)),
            failed_tasks: CacheAligned::new(std::sync::atomic::AtomicU64::new(0)),
            admission_caller_runs: CacheAligned::new(std::sync::atomic::AtomicU64::new(0)),
            shutdown: CacheAligned::new(std::sync::atomic::AtomicBool::new(false)),
            shutdown_join_state: std::sync::atomic::AtomicU8::new(0),
            join_waiters: CacheAligned::new(AtomicUsize::new(0)),
            wait_lock: std::sync::Mutex::new(()),
            wait_signal: std::sync::Condvar::new(),
            idle_workers: super::super::idle::IdleBitset::new(worker_count),
            worker_numa_nodes,
            blocking_lane: OnceLock::new(),
            blocking_lane_init: Mutex::new(()),
            blocking_lane_prefix: thread_name_prefix.into(),
            #[cfg(test)]
            shutdown_started_barrier: OnceLock::new(),
            lifetime_owner: OnceLock::new(),
        });

        for (worker_id, owner) in queue_owners.into_iter().enumerate() {
            let worker_inner = Arc::clone(&inner);
            let thread_name = format!("{thread_name_prefix}-{worker_id}");
            let handle = thread::Builder::new()
                .name(thread_name)
                .spawn(move || {
                    super::super::worker::worker_loop::<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>(
                        worker_inner,
                        worker_id,
                        owner,
                    )
                })
                .map_err(|_| ExecutorError::ThreadPoolCreationFailed)?;

            lock_mutex(&inner.handles).push(handle);
        }

        // Wait until all workers have registered their thread handles
        for worker in &inner.workers {
            while worker.thread.get().is_none() {
                thread::yield_now();
            }
        }

        Ok(Self { inner })
    }

    /// Keep an executor-owned allocation alive until every scheduler worker
    /// and queued job has released the shared scheduler state.
    pub(crate) fn retain_lifetime_owner<T>(&self, owner: T)
    where
        T: Send + Sync + 'static,
    {
        let owner: Box<dyn std::any::Any + Send + Sync> = Box::new(owner);
        assert!(
            self.inner.lifetime_owner.set(owner).is_ok(),
            "invariant: scheduler lifetime owner is installed once"
        );
    }
}

fn partition_global_queue(
    max_global_queue_size: usize,
    worker_count: usize,
) -> ExecutorResult<usize> {
    let partition = max_global_queue_size / worker_count;
    if partition < 2 {
        return Err(ExecutorError::InvalidConfiguration);
    }

    Ok(1usize << partition.ilog2())
}
