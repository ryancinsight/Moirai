#[test]
fn gpu_task_adapter_uses_moirai_block_on_not_pollster() {
    let manifest = read_benchmark("../moirai-gpu/Cargo.toml");
    let gpu_task = read_benchmark("../moirai-gpu/src/task.rs");
    let executor = read_benchmark("../moirai-executor/src/lib.rs");
    let dependency_section = manifest_section(&manifest, "[dependencies]");
    let feature_section = manifest_section(&manifest, "[features]");

    assert!(
        manifest_section_declares_dependency(dependency_section, "moirai-executor"),
        "moirai-gpu must depend on the Moirai-owned executor boundary"
    );
    assert!(
        feature_section.contains("\"dep:moirai-executor\""),
        "wgpu-backend must activate moirai-executor for sync GPU task waits"
    );
    assert!(
        executor.contains("pub fn block_on<F>(future: F) -> F::Output")
            && executor.contains("schedule::wake::block_on_current_thread(future)"),
        "moirai-executor must expose the current-thread parking block_on boundary"
    );
    assert!(
        gpu_task.contains("moirai_executor::block_on(self.gpu_task.execute_gpu(&self.device))"),
        "GPU task adapter must run synchronous waits through Moirai"
    );

    for prohibited in ["pollster", "pollster::block_on", "\"dep:pollster\""] {
        assert!(
            !manifest.contains(prohibited) && !gpu_task.contains(prohibited),
            "moirai-gpu must not reintroduce {prohibited}"
        );
    }
}

#[test]
fn public_result_handle_comparison_uses_real_join_handles() {
    let source = read_benchmark("benches/public_result_handle_comparison.rs");

    for required in [
        "WakeOnce",
        "CAPTURE_WORDS",
        "OVERSIZED_CAPTURE_WORDS",
        "moirai_spawn_join_ready",
        "tokio_spawn_join_ready",
        "moirai_spawn_join_captured_ready",
        "tokio_spawn_join_captured_ready",
        "moirai_spawn_join_oversized_captured_ready",
        "tokio_spawn_join_oversized_captured_ready",
        "moirai_spawn_async_ready",
        "moirai_spawn_async_wake_once",
        "tokio_spawn_async_wake_once",
        "moirai_scope_single_ready",
        "rayon_scope_single_ready",
        "verify_ready_value",
        "tokio::spawn",
        ".scope(|scope|",
        ".join()",
        ".await",
        "wake_by_ref",
        "Poll::Pending",
    ] {
        assert!(
            source.contains(required),
            "public result-handle benchmark must contain {required}"
        );
    }

    assert!(
        !source.contains(&["tokio_spawn_", "async_ready"].concat()),
        "public result-handle benchmark must not duplicate the ready Tokio JoinHandle baseline under another row name"
    );
}

#[test]
fn registry_lifecycle_keeps_qpc_out_of_production_path() {
    let source = read_benchmark("../moirai-executor/src/registry/mod.rs");

    for required in [
        "fn elapsed_nanos_since(origin: Instant) -> u64",
        "elapsed_nanos_since(self.created_at)",
        "mark_completed_since(self.started_after_ns)",
        "completed_after_ns >= started_after_ns",
        "Duration::from_nanos(completed_after_ns - started_after_ns)",
        "pub(crate) fn complete(self) -> Duration",
        "token.completed = true;",
        "mark_completed_since(token.started_after_ns)",
    ] {
        assert!(
            source.contains(required),
            "registry lifecycle timing must retain the production Instant policy through {required}"
        );
    }

    let prohibited = "completed_after_ns.saturating_sub(started_after_ns)";
    assert!(
        !source.contains(prohibited),
        "registry lifecycle completion must expose the monotonic timestamp invariant through {prohibited}"
    );

    let prohibited = "token.complete_once().unwrap_or(Duration::ZERO)";
    assert!(
        !source.contains(prohibited),
        "explicit running lifecycle completion must not route through the drop-path Option branch"
    );

    for prohibited in [
        "QueryPerformanceCounter",
        "QueryPerformanceFrequency",
        "qpc_created_ticks",
        "qpc_ticks_per_second: i64",
        "qpc_ticks_to_nanos(ticks, self.qpc_ticks_per_second)",
        "OnceLock<i64>",
        "AtomicI64",
    ] {
        assert!(
            !source.contains(prohibited),
            "registry lifecycle timing must keep rejected QPC production policy out through {prohibited}"
        );
    }
}

#[test]
fn task_result_wait_uses_zero_sized_policy_and_load_gated_take() {
    let source = read_benchmark("../moirai-core/src/task.rs");

    for required in [
        "trait ResultWaitPolicy",
        "struct BlockingResultWait",
        "impl ResultWaitPolicy for BlockingResultWait",
        "slot.wait::<BlockingResultWait>()",
        "for _ in 0..P::SPIN_ATTEMPTS",
        "try_take_observed_ready",
        "self.state.load(Ordering::Relaxed) == RESULT_READY",
        "compare_exchange(",
        "pub fn diagnostic_result_slot_ready_take() -> usize",
        "pub fn diagnostic_result_slot_spin_miss() -> usize",
        "pub fn diagnostic_result_slot_register_waiter() -> usize",
        "pub fn diagnostic_result_slot_complete_waiting() -> usize",
        "result_wait_policy_is_zero_sized_and_const_bounded",
        "size_of::<BlockingResultWait>()",
    ] {
        assert!(
            source.contains(required),
            "task result wait path must retain {required}"
        );
    }

    for prohibited in ["dyn ResultWaitPolicy", "Box<dyn ResultWaitPolicy"] {
        assert!(
            !source.contains(prohibited),
            "task result wait path must not introduce runtime policy dispatch through {prohibited}"
        );
    }
}

#[test]
fn scheduled_job_storage_keeps_two_cache_line_inline_budget() {
    let source = read_benchmark("../moirai-executor/src/schedule/job/mod.rs");

    for required in [
        "const INLINE_JOB_WORDS: usize = 14",
        "#[repr(C, align(64))]",
        "pub(crate) struct ScheduledJob",
        "job: InlineJob",
        "InlineJob::new(boxed_job(task))",
        "fn boxed_job<F>(task: F) -> impl FnOnce(usize) + Send",
        "drop_consumed",
        "inline_job_uses_two_cache_line_budget",
        "maximum_two_cache_line_job_uses_inline_storage",
        "oversized_job_uses_boxed_inline_trampoline",
    ] {
        assert!(
            source.contains(required),
            "scheduled job storage must retain {required}"
        );
    }

    for prohibited in [
        "pub(crate) enum ScheduledJob",
        "struct HeapJob",
        "Box<dyn FnOnce",
        "execute_heap",
        "drop_heap",
    ] {
        assert!(
            !source.contains(prohibited),
            "scheduled job storage must not reintroduce {prohibited}"
        );
    }
}

#[test]
fn scheduler_scope_buffers_inline_scheduled_jobs() {
    let source = read_benchmark("../moirai-executor/src/schedule/runtime/mod.rs");

    for required in [
        "jobs: RefCell<Vec<ScheduledJob>>",
        "self.state().register_task()",
        "let scoped_task = move |worker_id|",
        "ScheduledJob::new_scoped(scoped_task)",
        "fn schedule_single(&self, job: ScheduledJob)",
        // Admission leaves a refused job in the caller's slot so `flush` can run
        // it on the calling lane instead of dropping it (ISSUE-221).
        ".admit_job::<C>(self.priority, self.locality_hint, &mut job)",
        "fn run_if_refused",
        "self.scheduler.record_admission_caller_run()",
        "job.execute(self.scheduler.caller_lane_id())",
        "fn schedule_chunk(&self, jobs: Vec<ScheduledJob>)",
        "let _ = job.execute(worker_id)",
    ] {
        assert!(
            source.contains(required),
            "scheduler scope must retain inline scheduled-job buffering through {required}"
        );
    }

    for prohibited in ["ScopedJobFn", "Box<dyn FnOnce", "Box::new(move |_|"] {
        assert!(
            !source.contains(prohibited),
            "scheduler scope must not reintroduce dynamic scoped-job dispatch through {prohibited}"
        );
    }
}

#[test]
fn executor_registry_registration_rejects_regressed_lock_free_allocator() {
    let registry_source = read_benchmark("../moirai-executor/src/registry/mod.rs");
    let hybrid_source = read_benchmark("../moirai-executor/src/hybrid/mod.rs");

    for required in [
        "blocks: Vec<TaskStateBlock>",
        // Dense inline slots, now interior-mutable so a lifecycle token's stable
        // `NonNull<TaskState>` is sound under the aliasing model while the
        // registry mutates sibling slots. `UnsafeCell` is zero-cost: same dense
        // layout and address stability, no per-task allocation.
        "slots: Box<[UnsafeCell<Option<TaskState>>]>",
        "fn ensure_block(&mut self, block_index: usize)",
        "TaskLifecycleToken {",
    ] {
        assert!(
            registry_source.contains(required),
            "registry must retain the accepted dense-block lifecycle policy through {required}"
        );
    }

    for required in [
        "task_registry: Arc<Mutex<TaskRegistry>>",
        "task_registry: Arc::new(Mutex::new(TaskRegistry::new()))",
        "let mut registry = self.task_registry.lock().map_err",
        "let (task_id, lifecycle) = registry.register_next_task();",
        "Ok((TaskId::new(task_id), lifecycle))",
    ] {
        assert!(
            hybrid_source.contains(required),
            "hybrid executor must retain accepted registry access through {required}"
        );
    }

    for prohibited in [
        "pub(crate) struct ConcurrentTaskRegistry",
        "register_unique_task_with_id",
        "fn register_unique(&self, id: u64)",
        "next_task_id: AtomicU64",
        "fn allocate_task_id(&self) -> TaskId",
        "registry.register_task_with_id(task_id.0)",
    ] {
        assert!(
            !registry_source.contains(prohibited) && !hybrid_source.contains(prohibited),
            "regressed lock-free registry allocator must not be retained through {prohibited}"
        );
    }
}

#[test]
fn registry_hot_path_diagnostics_use_production_registry_paths() {
    let registry_source = format!(
        "{}\n{}\n{}",
        read_benchmark("../moirai-executor/src/registry/registry.rs"),
        read_benchmark("../moirai-executor/src/registry/diagnostics.rs"),
        read_benchmark("../moirai-executor/src/registry/state.rs")
    );
    let diagnostics_source = read_result_handle_diagnostics();
    let registry_diagnostics_source =
        read_benchmark("benches/result_handle_diagnostics/registry_paths.rs");

    for required in [
        "pub fn diagnostic_block_lookup(&mut self) -> u64",
        "pub fn diagnostic_slot_initialize(&mut self) -> u64",
        "pub fn diagnostic_lifecycle_timestamp_publication() -> Duration",
        "self.ensure_block(block_index);",
        // Diagnostics and production register/lookup now share the same
        // `TaskStateBlock` accessors (interior-mutable `UnsafeCell` slots), so
        // the diagnostic block-lookup and slot-init paths still exercise the
        // exact production code rather than a divergent copy.
        "self.blocks[block_index].get(slot_index)",
        "self.blocks[block_index].insert(slot_index)",
        "fn snapshot(&self, id: u64) -> TaskMetadata",
        "id,\n            created_at",
        "let started_after_ns = state.mark_started(0);",
        "state.mark_completed_since(started_after_ns)",
        "lifecycle.start(0).complete()",
        "pub fn diagnostic_register_next_and_complete_with_token(&mut self) -> Duration",
        "pub fn diagnostic_register_next_and_complete_with_token_id(&mut self) -> (u64, Duration)",
    ] {
        assert!(
            registry_source.contains(required),
            "registry diagnostics must remain backed by production registry code through {required}"
        );
    }

    for required in [
        "fn registry_mutex_lock_only(registry: &Mutex<TaskRegistry>) -> usize",
        "fn registry_block_lookup(registry: &mut TaskRegistry) -> usize",
        "fn registry_slot_initialize(registry: &mut TaskRegistry) -> usize",
        "fn registry_lifecycle_timestamp_publication() -> usize",
        "fn registry_elapsed_nanos_since_origin(origin: Instant) -> usize",
        "fn registry_start_release_publication(",
        "fn registry_completion_release_publication(completed_after_ns: &AtomicUsize) -> usize",
        "fn registry_duration_offset_math() -> usize",
        "fn direct_registry_token_lifecycle(",
        "fn direct_scheduled_public_token_wrapper_components(",
        "fn direct_scheduled_public_registry_token_wrapper_components(",
        "fn direct_scheduled_public_registry_token_wrapper_after_send_quiescent(",
        "fn direct_scheduled_public_registry_token_wrapper_local_metrics_quiescent(",
        "fn direct_scheduled_public_token_wrapper_without_metrics(",
        "fn direct_scheduled_public_registry_token_wrapper_without_metrics(",
        "fn direct_scheduled_public_token_wrapper_without_catch(",
        "fn direct_scheduled_public_token_wrapper_atomic_result(",
        "fn direct_scheduled_public_token_wrapper_without_lifecycle(",
        "fn direct_scheduled_public_token_wrapper_oversized_components(",
        "fn direct_scheduled_public_token_wrapper_oversized_storage_only(",
        "fn direct_scheduled_public_token_wrapper_oversized_read_one_components(",
        "fn direct_scheduled_public_token_wrapper_oversized_without_metrics(",
        "diagnostic_block_lookup()",
        "diagnostic_slot_initialize()",
        "diagnostic_lifecycle_timestamp_publication()",
        "diagnostic_register_next_and_complete_with_token()",
        "diagnostic_register_next_and_complete_with_token_id()",
        "let (id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();",
        "fn direct_public_token_wrapper_components(",
        "fn direct_public_token_wrapper_after_send_components(",
        "started_after_ns.store(offset, Ordering::Release)",
        "worker_id.store(black_box(BLOCKING_NORMAL_WORKER), Ordering::Release)",
        "completed_after_ns.store(offset, Ordering::Release)",
        "completed_after_ns - started_after_ns",
    ] {
        assert!(
            diagnostics_source.contains(required),
            "registry diagnostic benchmark must retain row {required}"
        );
    }

    assert!(
        !registry_diagnostics_source
            .contains("completed_after_ns.saturating_sub(started_after_ns)"),
        "registry duration math diagnostic must match production monotonic subtraction"
    );

    assert!(
        !registry_diagnostics_source.contains("READY_VALUE.saturating_add(CAPTURED_READY_VALUE)"),
        "registry duration diagnostic must not add defensive saturating arithmetic to monotonic fixture setup"
    );

    for prohibited in [
        "pub fn diagnostic_register_external_task_with_id",
        "pub fn diagnostic_restart_and_complete_with_token(&mut self, id: u64)",
        "fn direct_external_id_registry_register(",
        "fn direct_registry_external_token_lifecycle(",
        "diagnostic_register_external_task_with_id(id)",
        "diagnostic_restart_and_complete_with_token(id)",
        "registry.diagnostic_restart_and_complete_with_token(id)",
    ] {
        assert!(
            !registry_source.contains(prohibited)
                && !diagnostics_source.contains(prohibited)
                && !registry_diagnostics_source.contains(prohibited),
            "registry lifecycle diagnostics must use registry-owned task IDs, not external-ID accounting through {prohibited}"
        );
    }

    assert!(
        !registry_source.contains("pub(crate) struct TaskState {\n    id: u64,"),
        "dense direct-indexed registry slots must not store a redundant task id"
    );
}

#[test]
fn work_class_routing_stays_zero_sized_and_static() {
    let class_source = read_benchmark("../moirai-executor/src/schedule/class/mod.rs");
    let runtime_source = read_benchmark("../moirai-executor/src/schedule/runtime/mod.rs");
    let queue_source = read_benchmark("../moirai-executor/src/schedule/queue/mod.rs");
    let hybrid_source = read_benchmark("../moirai-executor/src/hybrid/mod.rs");
    let diagnostics_source = read_benchmark("benches/result_handle_diagnostics/scheduler_paths.rs");

    for required in [
        "mod sealed",
        "pub trait WorkClass: sealed::Sealed + Send + Sync + 'static",
        "const SERIAL_AFFINITY_OFFSET: usize",
        "pub struct SyncTask;",
        "pub struct AsyncTask;",
        "pub struct BlockingTask;",
        "impl sealed::Sealed for SyncTask",
        "impl sealed::Sealed for AsyncTask",
        "impl sealed::Sealed for BlockingTask",
        "assert_eq!(core::mem::size_of::<SyncTask>(), 0)",
        "assert_eq!(core::mem::size_of::<AsyncTask>(), 0)",
        "assert_eq!(core::mem::size_of::<BlockingTask>(), 0)",
    ] {
        assert!(
            class_source.contains(required),
            "work-class markers must retain sealed ZST routing through {required}"
        );
    }

    for required in [
        "pub fn schedule<C, F>",
        "C: WorkClass",
        "fn select_worker_for_state<C>",
        "C::SERIAL_AFFINITY_OFFSET",
        "C::AFFINITY_OFFSET",
        "active_workers.fetch_add(1, Ordering::Release)",
        "pending_tasks.fetch_sub(1, Ordering::Release)",
        "completed_tasks.fetch_add(1, Ordering::Relaxed)",
        "failed_tasks.fetch_add(1, Ordering::Relaxed)",
        // SeqCst, not AcqRel: this decrement is one half of the join()
        // quiescence Dekker handshake closed in 4d790a9 ("Close join()
        // quiescence lost-wakeup (AcqRel -> SeqCst)"), loom-verified by
        // tests/loom_join_quiescence.rs. AcqRel permits the StoreLoad
        // reordering that causes the lost wakeup; do not regress this back
        // to AcqRel.
        "active_workers.fetch_sub(1, Ordering::SeqCst)",
    ] {
        assert!(
            runtime_source.contains(required),
            "scheduler runtime must retain monomorphized work-class routing through {required}"
        );
    }

    for required in [
        "Queue contents are synchronized by `state`",
        "scheduler quiescence",
        "self.len.fetch_add(1, Ordering::Relaxed)",
        "self.len.fetch_sub(1, Ordering::Relaxed)",
        "self.len.load(Ordering::Relaxed) == 0",
    ] {
        assert!(
            queue_source.contains(required),
            "worker queue length must remain an advisory relaxed counter through {required}"
        );
    }

    for prohibited in [
        "self.len.fetch_add(1, Ordering::Release)",
        "self.len.fetch_sub(1, Ordering::AcqRel)",
        "self.len.load(Ordering::Acquire) == 0",
    ] {
        assert!(
            !queue_source.contains(prohibited),
            "worker queue length must not regain synchronization responsibility through {prohibited}"
        );
    }

    for required in [
        "spawn_result::<SyncTask, _>",
        ".schedule::<C, _>",
        ".schedule::<BlockingTask, _>",
        ".schedule::<AsyncTask, _>",
    ] {
        assert!(
            hybrid_source.contains(required),
            "hybrid executor must retain static scheduler calls through {required}"
        );
    }

    for source in [&class_source, &runtime_source, &hybrid_source] {
        for prohibited in ["dyn WorkClass", "Box<dyn WorkClass"] {
            assert!(
                !source.contains(prohibited),
                "work-class routing must not use runtime dispatch through {prohibited}"
            );
        }
    }

    for required in [
        "pub fn diagnostic_select_worker_for_state<C>",
        "pub fn diagnostic_pending_counter_pair(&self) -> usize",
        "pub fn diagnostic_worker_unpark(&self, worker_index: usize) -> usize",
        "pub fn diagnostic_priority_queue_push_pop(priority: Priority) -> usize",
        "pub fn diagnostic_worker_execute_ready_job(&self, worker_index: usize) -> usize",
        "pub fn diagnostic_worker_local_dequeue_execute(&self, worker_index: usize) -> usize",
        "pub fn diagnostic_max_inline_job_construct_drop() -> usize",
        "pub fn diagnostic_max_inline_job_construct_execute() -> usize",
        "pub fn diagnostic_oversized_job_construct_drop() -> usize",
        "pub fn diagnostic_oversized_job_construct_execute() -> usize",
        "pub fn diagnostic_max_inline_queue_push_pop_execute() -> usize",
        "pub fn diagnostic_oversized_queue_push_pop_execute() -> usize",
        "pub fn diagnostic_worker_local_max_inline_dequeue_execute(&self, worker_index: usize)",
        "pub fn diagnostic_worker_local_oversized_dequeue_execute(&self, worker_index: usize)",
        "pub fn diagnostic_join_fast_spin_quiescent(&self) -> usize",
        "pub fn diagnostic_join_fast_spin_pending(&self) -> usize",
    ] {
        assert!(
            runtime_source.contains(required),
            "scheduler diagnostics must expose production primitive attribution through {required}"
        );
    }

    for required in [
        "fn direct_scheduler_select_worker_serial(scheduler: &ThreadScheduler) -> usize",
        "fn direct_scheduler_pending_counter_pair(scheduler: &ThreadScheduler) -> usize",
        "fn direct_scheduler_worker_unpark(scheduler: &ThreadScheduler) -> usize",
        "fn direct_scheduler_priority_queue_push_pop() -> usize",
        "fn direct_scheduler_worker_execute_ready_job(scheduler: &ThreadScheduler) -> usize",
        "fn direct_scheduler_worker_local_dequeue_execute(scheduler: &ThreadScheduler) -> usize",
        "fn direct_scheduler_worker_start_signal(scheduler: &ThreadScheduler) -> usize",
        "fn direct_scheduler_worker_start_then_result_slot(scheduler: &ThreadScheduler) -> usize",
        "fn direct_scheduler_max_inline_job_construct_drop() -> usize",
        "fn direct_scheduler_max_inline_job_construct_execute() -> usize",
        "fn direct_scheduler_oversized_job_construct_drop() -> usize",
        "fn direct_scheduler_oversized_job_construct_execute() -> usize",
        "fn direct_scheduler_max_inline_queue_push_pop_execute() -> usize",
        "fn direct_scheduler_oversized_queue_push_pop_execute() -> usize",
        "fn direct_scheduler_worker_local_max_inline_dequeue_execute(",
        "fn direct_scheduler_worker_local_oversized_dequeue_execute(scheduler: &ThreadScheduler) -> usize",
        "fn direct_scheduler_join_fast_spin_quiescent(scheduler: &ThreadScheduler) -> usize",
        "fn direct_scheduler_join_fast_spin_pending(scheduler: &ThreadScheduler) -> usize",
    ] {
        assert!(
            diagnostics_source.contains(required),
            "result-handle diagnostics must retain scheduler primitive row {required}"
        );
    }
}

#[test]
fn rejected_scheduler_inline_handoff_candidate_stays_removed() {
    let executor_manifest = read_benchmark("../moirai-executor/Cargo.toml");
    let benchmark_manifest = read_benchmark("Cargo.toml");
    let runtime_source = read_benchmark("../moirai-executor/src/schedule/runtime/mod.rs");

    for prohibited in [
        "scheduler-inline-handoff",
        "InlineHandoffSlot",
        "HANDOFF_EMPTY",
        "HANDOFF_WRITING",
        "HANDOFF_READY",
        "try_publish_handoff",
        "take_handoff(&self) -> Option<ScheduledJob>",
    ] {
        assert!(
            !executor_manifest.contains(prohibited)
                && !benchmark_manifest.contains(prohibited)
                && !runtime_source.contains(prohibited),
            "rejected scheduler inline handoff candidate must stay removed: {prohibited}"
        );
    }
}
