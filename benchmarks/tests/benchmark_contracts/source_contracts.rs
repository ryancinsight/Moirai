#[test]
fn gpu_tasks_use_the_hephaestus_device_seam() {
    let manifest = read_benchmark("../moirai-gpu/Cargo.toml");
    let gpu_task = read_benchmark("../moirai-gpu/src/task/mod.rs");
    let gpu_context = read_benchmark("../moirai-gpu/src/device/context.rs");
    let gpu_source = read_benchmark("../moirai-gpu/src/lib.rs");
    let dependency_section = manifest_section(&manifest, "[dependencies]");
    let feature_section = manifest_section(&manifest, "[features]");
    let wgpu_feature = feature_section
        .lines()
        .find(|line| line.trim_start().starts_with("wgpu-backend ="))
        .unwrap_or_default();

    assert!(
        manifest_section_declares_dependency(dependency_section, "hephaestus-core"),
        "moirai-gpu must depend on the Hephaestus device contract"
    );
    assert!(
        wgpu_feature.contains("\"dep:hephaestus-wgpu\""),
        "wgpu-backend must activate the Hephaestus WGPU provider"
    );
    assert!(
        gpu_task.contains("pub trait GpuTask")
            && gpu_task.contains("fn execute_gpu(self, device: &Self::Device)"),
        "GPU tasks must carry a typed Hephaestus device seam"
    );
    assert!(
        gpu_context.contains("self.device.upload(host)")
            && gpu_context.contains("eunomia::Pod"),
        "GPU context transfers must delegate to Eunomia-bounded Hephaestus APIs"
    );

    for prohibited in [
        "moirai-executor",
        "moirai::block_on",
        "Box<dyn Future",
        "wgpu::",
        "bytemuck",
    ] {
        assert!(
            !contains_prohibited(&manifest, prohibited)
                && !contains_prohibited(&gpu_task, prohibited)
                && !contains_prohibited(&gpu_context, prohibited)
                && !contains_prohibited(&gpu_source, prohibited),
            "moirai-gpu must not reintroduce {prohibited}"
        );
    }
}

fn contains_prohibited(source: &str, needle: &str) -> bool {
    if needle != "wgpu::" {
        return source.contains(needle);
    }

    source.match_indices(needle).any(|(index, _)| {
        index == 0 || source.as_bytes().get(index - 1) != Some(&b'_')
    })
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
        "elapsed_nanos_since(self.created_at).max(started_after_ns)",
        "Duration::from_nanos(completed_after_ns - started_after_ns)",
        "pub(crate) fn complete(mut self) -> Duration",
        "self.completed = true;",
        "mark_completed_since(self.started_after_ns)",
    ] {
        assert!(
            source.contains(required),
            "registry lifecycle timing must retain the production Instant policy through {required}"
        );
    }

    let prohibited = "completed_after_ns.saturating_sub(started_after_ns)";
    assert!(
        !source.contains(prohibited),
        "registry lifecycle completion must clamp the sampled timestamp before plain subtraction, not use {prohibited}"
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
fn executor_registry_registration_rejects_regressed_lock_free_allocator() {
    let registry_source = read_benchmark("../moirai-executor/src/registry/mod.rs");
    let hybrid_source = read_benchmark("../moirai-executor/src/hybrid/mod.rs");
    let scheduler_source = format!(
        "{}\n{}\n{}\n{}",
        read_benchmark("../moirai-executor/src/schedule/runtime/types.rs"),
        read_benchmark("../moirai-executor/src/schedule/runtime/scheduler/construction.rs"),
        read_benchmark("../moirai-executor/src/schedule/runtime/scheduler/core.rs"),
        read_benchmark("../moirai-executor/src/schedule/runtime/worker.rs")
    );

    for required in [
        // Dense blocks are the point of this contract: one block allocation
        // per 1024 tasks, never one per task. The directory gained a lock so
        // registration takes `&self` (ADR 0005, 2026-08-29 revision); the
        // block granularity it guards is unchanged.
        "blocks: RwLock<Vec<Arc<TaskStateBlock>>>",
        // Dense inline slots remain one block allocation per 1024 tasks. Async
        // tokens retain the block; scheduler-bounded tokens use a non-owning
        // lifetime policy without restoring one allocation per task.
        "slots: Box<[UnsafeCell<Option<TaskState>>]>",
        "block: Arc<TaskStateBlock>",
        "token_active: AtomicBool",
        "pub(crate) struct OwnedStateLease",
        "pub(crate) struct SchedulerStateLease",
        "pub(super) fn ensure_block(&self, block_index: usize) -> Arc<TaskStateBlock>",
        "TaskLifecycleToken::new_owned(block, state)",
        "TaskLifecycleToken::new_scheduled(state)",
    ] {
        assert!(
            registry_source.contains(required),
            "registry must retain the accepted dense-block lifecycle policy through {required}"
        );
    }

    for required in [
        "task_registry: Arc<TaskRegistry>,",
        "let task_registry = Arc::new(TaskRegistry::new())",
        "scheduler.retain_lifetime_owner((Arc::clone(&task_registry), Arc::clone(&metrics)))",
        "let registry = &self.task_registry;",
        "unsafe { registry.register_next_scheduled_task() }",
        "Ok((TaskId::new(task_id), lifecycle))",
    ] {
        assert!(
            hybrid_source.contains(required),
            "hybrid executor must retain accepted registry access through {required}"
        );
    }

    for required in [
        "lifetime_owner: OnceLock<Box<dyn Any + Send + Sync>>",
        "pub(crate) fn retain_lifetime_owner<T>(&self, owner: T)",
        "pub(super) fn join_other_threads(handles: &mut Vec<JoinHandle<()>>)",
        "handle.thread().id() != current",
    ] {
        assert!(
            scheduler_source.contains(required),
            "scheduler must retain the registry through every job through {required}"
        );
    }

    // The 2026-05 rejection stands on its own terms and stays enforced: the
    // design refused there allocated one `Arc<TaskState>` per task and split id
    // allocation from registration, so an id could exist unregistered. The
    // 2026-08-29 revision changes only where the lock sits — registration is
    // still a single call that allocates the id and claims the slot together,
    // and slots are still dense blocks. Every marker below is still banned.
    for prohibited in [
        "Arc<TaskState>",
        "pub(crate) struct ConcurrentTaskRegistry",
        "register_unique_task_with_id",
        "fn register_unique(&self, id: u64)",
        "fn allocate_task_id(&self) -> TaskId",
        "registry.register_task_with_id(task_id.0)",
        // Re-introducing the global registry mutex would restore the
        // multi-producer collapse the revision removed: spawn throughput fell
        // from 3.13 to 3.08 M/s between one and eight producers, where it now
        // rises to 4.90 M/s.
        "task_registry: Arc<Mutex<TaskRegistry>>",
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
        "pub fn diagnostic_block_lookup(&self) -> u64",
        "pub fn diagnostic_slot_initialize(&self) -> u64",
        "pub fn diagnostic_lifecycle_timestamp_publication() -> Duration",
        "self.ensure_block(block_index);",
        // Diagnostics and production register/lookup now share the same
        // `TaskStateBlock` accessors (interior-mutable `UnsafeCell` slots), so
        // the diagnostic block-lookup and slot-init paths still exercise the
        // exact production code rather than a divergent copy.
        "let slot_occupied = block.get(slot_index).is_some();",
        "self.ensure_block(block_index).insert(slot_index)",
        "fn snapshot(&self, id: u64) -> TaskMetadata",
        "id,\n            created_at",
        "let started_after_ns = state.mark_started(0);",
        "state.mark_completed_since(started_after_ns)",
        "lifecycle.start(0).complete()",
        "pub fn diagnostic_register_next_and_complete_with_token(&self) -> Duration",
        "pub fn diagnostic_register_next_and_complete_with_retained_token(&self) -> Duration",
        "pub fn diagnostic_register_next_and_complete_with_token_id(&self) -> (u64, Duration)",
    ] {
        assert!(
            registry_source.contains(required),
            "registry diagnostics must remain backed by production registry code through {required}"
        );
    }

    for required in [
        "fn registry_shared_acquire_only(registry: &TaskRegistry) -> usize",
        "fn registry_block_lookup(registry: &TaskRegistry) -> usize",
        "fn registry_slot_initialize(registry: &TaskRegistry) -> usize",
        "fn registry_lifecycle_timestamp_publication() -> usize",
        "fn registry_elapsed_nanos_since_origin(origin: Instant) -> usize",
        "fn registry_start_release_publication(",
        "fn registry_completion_release_publication(completed_after_ns: &AtomicUsize) -> usize",
        "fn registry_duration_offset_math() -> usize",
        "fn direct_registry_token_lifecycle(",
        "fn direct_registry_retained_token_lifecycle(",
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
