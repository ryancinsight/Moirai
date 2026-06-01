#[test]
fn scheduler_submission_diagnostics_stay_static_and_value_checked() {
    let runtime_source = read_benchmark("../moirai-executor/src/schedule/runtime/mod.rs");
    let diagnostics_source = read_benchmark("benches/result_handle_diagnostics/scheduler_paths.rs");
    let row_source =
        read_benchmark("benches/result_handle_diagnostics/scheduler_submission_diagnostics.rs");
    let benchmark_source = read_benchmark("benches/result_handle_diagnostics/benchmark.rs");
    let module_source = read_benchmark("benches/result_handle_diagnostics/mod.rs");
    let support_source = read_benchmark("tests/benchmark_contracts/support.rs");

    for required in [
        "pub trait DiagnosticWakeDecision: diagnostic_wake::Sealed + Send + Sync + 'static",
        "pub struct EmptyWakeDecision;",
        "pub struct ContendedWakeDecision;",
        "pub struct SaturatedWakeDecision;",
        "trait ContendedWakePolicy: contended_wake::Sealed + Send + Sync + 'static",
        "struct BoundedContendedWake;",
        "impl ContendedWakePolicy for BoundedContendedWake",
        "const WAKE_LIMIT: usize = 2;",
        "impl DiagnosticWakeDecision for EmptyWakeDecision",
        "impl DiagnosticWakeDecision for ContendedWakeDecision",
        "impl DiagnosticWakeDecision for SaturatedWakeDecision",
        "fn wake_contended_workers<P>(",
        "P: ContendedWakePolicy",
        "wake_contended_workers::<BoundedContendedWake>(",
        "fn diagnostic_publish_work_available(",
        "pub fn diagnostic_submission_queue_publication<C>(",
        "C: WorkClass",
        "AtomicUsize::new(0)",
        "self.select_worker_for_state::<C>(",
        "pending_tasks.fetch_add(1, Ordering::Release)",
        "queues.push(priority, ScheduledJob::new(|_| {}))",
        "job.execute(worker_index)",
        "pub fn diagnostic_wake_decision<P>(&self, worker_index: usize) -> usize",
        "P: DiagnosticWakeDecision",
        "diagnostic_publish_work_available(",
        "self.inner.as_ref()",
        "P::previous_pending(worker_count)",
    ] {
        assert!(
            runtime_source.contains(required),
            "scheduler submission diagnostic must retain static production primitives through {required}"
        );
    }

    for required in [
        "fn direct_scheduler_submission_queue_publication(scheduler: &ThreadScheduler) -> usize",
        "fn direct_scheduler_empty_wake_decision(scheduler: &ThreadScheduler) -> usize",
        "fn direct_scheduler_contended_wake_decision(scheduler: &ThreadScheduler) -> usize",
        "fn direct_scheduler_saturated_wake_decision(scheduler: &ThreadScheduler) -> usize",
        "fn direct_spawn_metrics_before_scheduler_submission(",
        "fn direct_spawn_metrics_after_scheduler_submission(",
        "assert_eq!(observed, 2)",
        "assert_eq!(after, before + 1)",
        "verify_ready_value(result)",
    ] {
        assert!(
            diagnostics_source.contains(required),
            "scheduler submission diagnostic row must retain value checking through {required}"
        );
    }

    for required in [
        "direct_scheduler_submission_queue_publication",
        "direct_scheduler_empty_wake_decision",
        "direct_scheduler_contended_wake_decision",
        "direct_scheduler_saturated_wake_decision",
        "direct_spawn_metrics_before_scheduler_submission",
        "direct_spawn_metrics_after_scheduler_submission",
    ] {
        assert!(
            row_source.contains(required),
            "scheduler submission benchmark registration must include {required}"
        );
    }

    assert!(
        benchmark_source
            .contains("benchmark_scheduler_submission_diagnostics(&mut group, &scheduler)"),
        "result-handle diagnostics must register scheduler submission rows"
    );
    assert!(
        module_source.contains("include!(\"scheduler_submission_diagnostics.rs\");"),
        "result-handle diagnostics module must include the submission leaf"
    );
    assert!(
        support_source
            .contains("benches/result_handle_diagnostics/scheduler_submission_diagnostics.rs"),
        "benchmark contract support must read the submission diagnostics leaf"
    );

    for source in [runtime_source, diagnostics_source, row_source] {
        for prohibited in ["dyn WorkClass", "Box<dyn WorkClass"] {
            assert!(
                !source.contains(prohibited),
                "scheduler submission diagnostics must not use runtime work-class dispatch through {prohibited}"
            );
        }
    }
}

#[test]
fn public_scheduler_task_surface_uses_scheduled_task_erasure() {
    let core_scheduler = read_benchmark("../moirai-core/src/scheduler.rs");
    let core_task = read_benchmark("../moirai-core/src/scheduler/task.rs");
    let core_lib = read_benchmark("../moirai-core/src/lib.rs");
    let task_source = read_benchmark("../moirai-core/src/task.rs");
    let scheduler_source = read_benchmark("../moirai-scheduler/src/lib.rs");
    let numa_source = read_benchmark("../moirai-scheduler/src/numa_scheduler.rs");
    let audit = read_benchmark("../docs/rayon_tokio_gap_audit.md");

    for required in [
        "pub const INLINE_SCHEDULED_TASK_WORDS: usize = 14;",
        "pub struct ScheduledTask",
        "storage: UnsafeCell<ScheduledTaskStorage>",
        "execute: unsafe fn(*mut ScheduledTaskStorage)",
        "drop_task: unsafe fn(*mut ScheduledTaskStorage)",
        "context: unsafe fn(*const ScheduledTaskStorage) -> *const TaskContext",
        "pub fn new<T>(task: T) -> Self",
        "T: Task",
        "Self::new_boxed(task)",
        "execute_inline_task::<T>",
        "drop_inline_task::<T>",
        "context_inline_task::<T>",
        "execute_boxed_task::<T>",
        "drop_boxed_task::<T>",
        "context_boxed_task::<T>",
        "scheduled_task_storage_budget_is_static_and_bounded",
        "scheduled_task_executes_inline_and_oversized_tasks",
    ] {
        assert!(
            core_task.contains(required),
            "core ScheduledTask erasure must retain {required}"
        );
    }

    for required in [
        "pub use task::{ScheduledTask, INLINE_SCHEDULED_TASK_WORDS};",
        "fn schedule(&self, task: ScheduledTask) -> SchedulerResult<()>",
        "fn schedule_task<T>(&self, task: T) -> SchedulerResult<()>",
        "fn next_task(&self) -> SchedulerResult<Option<ScheduledTask>>",
        "fn try_steal<S>(&self, victim: &S) -> SchedulerResult<Option<ScheduledTask>>",
        "pub struct WorkStealingCoordinator<S: Scheduler>",
        "schedulers: Vec<S>",
        "injector: Arc<WorkStealingDeque<ScheduledTask>>",
    ] {
        assert!(
            core_scheduler.contains(required),
            "core scheduler surface must retain static task dispatch through {required}"
        );
    }

    for required in [
        "pub use scheduler::{ScheduledTask, Scheduler, SchedulerConfig, SchedulerId};",
        "Priority, Task, TaskBuilder, TaskContext, TaskExt, TaskFuture, TaskHandle, TaskId",
    ] {
        assert!(
            core_lib.contains(required),
            "core lib re-export surface must retain {required}"
        );
    }

    for required in [
        "impl<T> Task for Box<T>",
        "T: Task",
        "fn execute(self) -> Self::Output",
        "(*self).execute()",
    ] {
        assert!(
            task_source.contains(required),
            "typed Box<T> task forwarding must remain static through {required}"
        );
    }

    for required in [
        "data: Box<[UnsafeCell<MaybeUninit<T>>]>",
        "unsafe fn write(&self, index: isize, item: T)",
        "unsafe fn read(&self, index: isize) -> T",
        "unsafe fn copy_slot_to(&self, target: &Self, index: isize)",
        "retired_arrays: Mutex<Vec<*mut Array<T>>>",
        "pub trait DequeReclaimPolicy: reclaim_policy::Sealed + Copy + Default",
        "type State: DequeReclaimState;",
        "pub trait DequeReclaimState: Default + Send + Sync",
        "pub struct QuiescentState;",
        "pub struct QuiescentReclaim;",
        "type State = QuiescentState;",
        "pub struct SharedEpochReclaim;",
        "type State = SharedEpochState;",
        "active_accesses: AtomicUsize",
        "let _guard = self.reclaim.enter();",
        "impl DequeReclaimPolicy for QuiescentReclaim",
        "pub fn reclaim_memory(&mut self, _policy: P)",
        "pub fn try_reclaim_shared(&self, _policy: SharedEpochReclaim) -> bool",
        "if !self.reclaim.can_reclaim_shared()",
        "drop(array.read(index));",
        "chase_lev_deque_resizes_without_per_item_heap_nodes",
        "chase_lev_deque_reclaims_retired_arrays_after_quiescence",
        "chase_lev_deque_reclamation_policies_are_static",
        "chase_lev_deque_shared_epoch_reclaim_waits_for_active_access",
        "chase_lev_deque_drops_each_inline_item_once",
        "local_queue: ChaseLevDeque<ScheduledTask>",
        "global_queue: Mutex<VecDeque<ScheduledTask>>",
        "pub fn schedule_task<T>(&self, task: T) -> SchedulerResult<()>",
        "self.schedule(ScheduledTask::new(task))",
        "pub fn try_steal_from(&self, other: &WorkStealingScheduler) -> StealResult<ScheduledTask>",
        "fn execute_task(&self, task: ScheduledTask)",
        "task.execute();",
        "fn schedule(&self, task: ScheduledTask) -> SchedulerResult<()>",
        "fn next_task(&self) -> SchedulerResult<Option<ScheduledTask>>",
        "fn try_steal<S>(&self, victim: &S) -> SchedulerResult<Option<ScheduledTask>>",
        ") -> Option<ScheduledTask>",
    ] {
        assert!(
            scheduler_source.contains(required),
            "moirai-scheduler work-stealing surface must retain {required}"
        );
    }

    for required in [
        "_local_queue: crate::ChaseLevDeque<ScheduledTask>",
        "priority_queues: [crate::ChaseLevDeque<ScheduledTask>; 4]",
        "fn push_task(&self, task: ScheduledTask, priority: Priority)",
        "fn pop_task(&self) -> Option<ScheduledTask>",
        "fn steal_task(&self) -> Option<ScheduledTask>",
        "pub fn schedule_task<T>(&self, task: T) -> SchedulerResult<()>",
        "pub fn schedule_on_node<T>",
        "fn schedule_erased_on_node(",
        "pub fn steal_with_locality(&self, worker_id: usize) -> Option<ScheduledTask>",
        "fn schedule(&self, task: ScheduledTask) -> SchedulerResult<()>",
        "fn next_task(&self) -> SchedulerResult<Option<ScheduledTask>>",
        "fn try_steal<S>(&self, _victim: &S) -> SchedulerResult<Option<ScheduledTask>>",
    ] {
        assert!(
            numa_source.contains(required),
            "NUMA scheduler surface must retain {required}"
        );
    }

    for source in [
        &core_scheduler,
        &core_task,
        &task_source,
        &scheduler_source,
        &numa_source,
    ] {
        for prohibited in [
            "BoxedTask",
            "Box<dyn BoxedTask",
            "Box<dyn Scheduler",
            "dyn Scheduler",
            "TaskSlot",
            "Pin<Box<dyn Future",
            "data: Box<[AtomicPtr<T>]>",
            "fn get(&self, index: isize) -> *mut T",
            "fn put(&self, index: isize, item: *mut T)",
            "let item_ptr = Box::into_raw(Box::new(item))",
            "Box::from_raw(item_ptr)",
            "pub unsafe fn reclaim_memory(&self)",
        ] {
            assert!(
                !source.contains(prohibited),
                "public scheduler task path must not reintroduce {prohibited}"
            );
        }
    }

    for required in [
        "Public scheduler task dispatch is concrete",
        "ScheduledTask",
        "INLINE_SCHEDULED_TASK_WORDS",
        "WorkStealingScheduler",
        "NumaAwareScheduler",
    ] {
        assert!(
            audit.contains(required),
            "Rayon/Tokio gap audit must track scheduler task erasure through {required}"
        );
    }
}
