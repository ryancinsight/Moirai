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
    #[cfg(test)]
    failure_probe: Option<ConstructionFailureProbe>,
}

#[cfg(test)]
struct ConstructionFailureProbe {
    worker_id: usize,
    lifetime_owner: Box<dyn std::any::Any + Send + Sync>,
    worker_exit_gate: Option<WorkerExitGate>,
}

#[cfg(test)]
struct WorkerExitGate {
    reached: std::sync::mpsc::SyncSender<()>,
    release: std::sync::mpsc::Receiver<()>,
}

impl ThreadScheduler<256, 8192> {
    /// Start a scheduler with a compute worker set and a lazy blocking lane.
    ///
    /// # Errors
    ///
    /// Returns an error when the default queue policy cannot be constructed or
    /// a worker thread cannot be started. A partially started worker set is
    /// shut down and joined before the error returns.
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
    /// failures after shutting down and joining any workers already started by
    /// the same construction attempt.
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
            #[cfg(test)]
            failure_probe: None,
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
            #[cfg(test)]
            failure_probe: None,
        })
    }

    #[cfg(test)]
    fn with_worker_construction_failure<T>(
        worker_count: usize,
        failing_worker_id: usize,
        lifetime_owner: T,
        worker_exit_gate: WorkerExitGate,
    ) -> ExecutorResult<Self>
    where
        T: Send + Sync + 'static,
    {
        assert!(
            failing_worker_id < worker_count.max(1),
            "invariant: injected worker failure targets the construction set"
        );
        Self::from_construction(SchedulerConstruction {
            worker_count,
            thread_name_prefix: "test-partial-spawn",
            max_global_queue_size: DEFAULT_GLOBAL_QUEUE_CAPACITY,
            local_queue_initial_capacity: DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY,
            numa_aware: false,
            failure_probe: Some(ConstructionFailureProbe {
                worker_id: failing_worker_id,
                lifetime_owner: Box::new(lifetime_owner),
                worker_exit_gate: Some(worker_exit_gate),
            }),
        })
    }

    fn from_construction(config: SchedulerConstruction<'_>) -> ExecutorResult<Self> {
        let SchedulerConstruction {
            worker_count,
            thread_name_prefix,
            max_global_queue_size,
            local_queue_initial_capacity,
            numa_aware,
            #[cfg(test)]
            mut failure_probe,
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
        let scheduler = Self { inner };

        #[cfg(test)]
        let failing_worker_id = failure_probe.as_ref().map(|probe| probe.worker_id);
        #[cfg(test)]
        let mut worker_exit_gate = failure_probe
            .as_mut()
            .and_then(|probe| probe.worker_exit_gate.take());
        #[cfg(test)]
        if let Some(probe) = failure_probe {
            assert!(
                scheduler
                    .inner
                    .lifetime_owner
                    .set(probe.lifetime_owner)
                    .is_ok(),
                "invariant: construction failure owner is installed once"
            );
        }

        for (worker_id, owner) in queue_owners.into_iter().enumerate() {
            let worker_inner = Arc::clone(&scheduler.inner);
            let thread_name = format!("{thread_name_prefix}-{worker_id}");
            #[cfg(test)]
            let inject_failure = failing_worker_id == Some(worker_id);
            #[cfg(not(test))]
            let inject_failure = false;
            #[cfg(test)]
            let exit_gate = if worker_id == 0 {
                worker_exit_gate.take()
            } else {
                None
            };
            let spawn_result = if inject_failure {
                Err(std::io::Error::other(
                    "injected worker construction failure",
                ))
            } else {
                thread::Builder::new().name(thread_name).spawn(move || {
                    super::super::worker::worker_loop::<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>(
                        worker_inner,
                        worker_id,
                        owner,
                    );
                    #[cfg(test)]
                    if let Some(gate) = exit_gate {
                        gate.reached
                            .send(())
                            .expect("test controller must observe worker exit");
                        gate.release
                            .recv()
                            .expect("test controller must release worker exit");
                    }
                })
            };
            let handle = match spawn_result {
                Ok(handle) => handle,
                Err(_) => {
                    scheduler.shutdown();
                    return Err(ExecutorError::ThreadPoolCreationFailed);
                }
            };

            lock_mutex(&scheduler.inner.handles).push(handle);
        }

        // Wait until all workers have registered their thread handles
        for worker in &scheduler.inner.workers {
            while worker.thread.get().is_none() {
                thread::yield_now();
            }
        }

        Ok(scheduler)
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

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            mpsc, Arc,
        },
        thread,
        time::Duration,
    };

    use moirai_core::error::ExecutorError;

    use super::{ThreadScheduler, WorkerExitGate};

    const TEST_EVENT_DEADLINE: Duration = Duration::from_secs(5);

    struct DropSignal {
        dropped: Arc<AtomicBool>,
    }

    impl Drop for DropSignal {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Release);
        }
    }

    #[test]
    fn partial_worker_spawn_failure_joins_started_workers() {
        let released = Arc::new(AtomicBool::new(false));
        let (exit_reached_sender, exit_reached_receiver) = mpsc::sync_channel(0);
        let (exit_release_sender, exit_release_receiver) = mpsc::sync_channel(0);
        let (result_sender, result_receiver) = mpsc::sync_channel(1);
        let released_by_constructor = Arc::clone(&released);
        let constructor = thread::spawn(move || {
            let result = ThreadScheduler::<256, 8192>::with_worker_construction_failure(
                2,
                1,
                DropSignal {
                    dropped: released_by_constructor,
                },
                WorkerExitGate {
                    reached: exit_reached_sender,
                    release: exit_release_receiver,
                },
            );
            result_sender
                .send(matches!(
                    result,
                    Err(ExecutorError::ThreadPoolCreationFailed)
                ))
                .expect("test controller must collect construction result");
        });

        exit_reached_receiver
            .recv_timeout(TEST_EVENT_DEADLINE)
            .expect("partial worker must observe shutdown and reach exit");
        assert!(
            matches!(result_receiver.try_recv(), Err(mpsc::TryRecvError::Empty)),
            "constructor must not return before its partial worker exits"
        );
        exit_release_sender
            .send(())
            .expect("test controller releases the partial worker");
        assert!(
            result_receiver
                .recv_timeout(TEST_EVENT_DEADLINE)
                .expect("constructor must return after its partial worker exits"),
            "construction must preserve the typed spawn failure"
        );
        constructor
            .join()
            .expect("construction test controller must not panic");
        assert!(
            released.load(Ordering::Acquire),
            "error return must release retained scheduler state"
        );
    }
}
