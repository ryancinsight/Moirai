#[test]
fn current_performance_artifacts_do_not_report_non_executable_estimates() {
    for relative in [
        "../PERFORMANCE_RESULTS.md",
        "../GAP_ANALYSIS.md",
        "../CHANGELOG.md",
        "benches/industry_comparison.rs",
        "benches/thread_schedule_comparison.rs",
        "benches/public_result_handle_comparison.rs",
        "benches/transport_archive_comparison.rs",
        "benches/channel_matrix.rs",
        "benches/example_pattern_comparison.rs",
        "benches/iterator_adapter_comparison.rs",
        "benches/async_iterator_comparison.rs",
        "benches/sorting_comparison.rs",
        "benches/async_fs_comparison.rs",
        "benches/async_udp_comparison.rs",
        "benches/async_tcp_comparison.rs",
        "benches/async_tcp_backpressure_comparison.rs",
        "benches/async_tcp_readiness_comparison.rs",
        "benches/async_tcp_cancel_safety_comparison.rs",
        "benches/async_io_compat_comparison.rs",
    ] {
        let content = read_benchmark(relative);
        let lowered = content.to_lowercase();
        let stale_estimate_label = ["simu", "lated"].concat();
        let stale_estimate_noun = ["simu", "lation"].concat();
        assert!(
            !lowered.contains(&stale_estimate_label) && !lowered.contains(&stale_estimate_noun),
            "{relative} must not contain non-executable benchmark claims"
        );
    }
}

#[test]
fn competitive_comparison_benchmarks_exclude_non_equivalent_diagnostic_rows() {
    let industry = read_benchmark("benches/industry_comparison.rs");
    let scheduler = read_benchmark("benches/thread_schedule_comparison.rs");

    for prohibited in [
        concat!("BenchmarkId::new(\"moirai_", "sync\""),
        concat!("bench_function(\"moirai_", "sync\""),
        concat!("BenchmarkId::new(\"moirai_", "indexed\","),
        concat!("bench_function(\"moirai_", "indexed\""),
        concat!("moirai_", "async_ready"),
        concat!("industry_", "cpu_fanout"),
        "std_thread",
    ] {
        assert!(
            !industry.contains(prohibited),
            "industry comparison must not include non-equivalent diagnostic row {prohibited}"
        );
        assert!(
            !scheduler.contains(prohibited),
            "scheduler comparison must not include non-equivalent diagnostic row {prohibited}"
        );
    }
}

#[test]
fn competitive_benchmarks_keep_value_assertions() {
    let industry = read_benchmark("benches/industry_comparison.rs");
    let scheduler = read_benchmark("benches/thread_schedule_comparison.rs");
    let channel = read_benchmark("benches/channel_matrix.rs");

    for required in [
        "verify_ready_sum",
        "verify_cpu_work_sum",
        "rayon_into_par_iter",
        "tokio_spawn",
        "moirai_scope",
        "moirai_indexed_reduce",
    ] {
        assert!(
            industry.contains(required),
            "industry comparison must contain {required}"
        );
    }

    assert!(
        scheduler.contains("verify_ready_sum")
            && scheduler.contains("moirai_scope")
            && scheduler.contains("indexed_reduce_schedule")
            && scheduler.contains("mixed_unified_schedule")
            && scheduler.contains("verify_mixed_sum")
            && scheduler.contains("moirai_unified_mixed")
            && scheduler.contains("tokio_rayon_mixed")
            && scheduler.contains("real_application_mixed_workload")
            && scheduler.contains("verify_real_app_sum")
            && scheduler.contains("expected_real_app_sum")
            && scheduler.contains("moirai_real_app_pipeline")
            && scheduler.contains("tokio_rayon_real_app_pipeline")
            && scheduler.contains("spsc::<usize>(capacity)")
            && scheduler.contains("tokio::sync::mpsc::channel::<usize>(capacity)")
            && scheduler.contains("standalone_deque_reclaim_policy")
            && scheduler.contains("moirai_quiescent_reclaim")
            && scheduler.contains("moirai_shared_epoch_reclaim")
            && scheduler.contains("moirai_deque_quiescent_reclaim_sum")
            && scheduler.contains("moirai_deque_shared_epoch_reclaim_sum"),
        "scheduler comparison must retain value assertions for scoped, indexed, mixed unified, real-app mixed, channel, and deque reclaim work"
    );

    assert!(
        channel.contains("verify_sum")
            && channel.contains("bounded_channel_matrix")
            && channel.contains("tokio_mpsc")
            && channel.contains("moirai_mpmc")
            && channel.contains("mpsc::channel::<usize>(capacity)")
            && channel.contains("mpmc::<usize>(capacity)"),
        "channel comparison must retain value assertions and bounded Tokio/Moirai channel paths"
    );
}

#[test]
fn criterion_benchmarks_are_executable_and_bounded() {
    let manifest = read_benchmark("Cargo.toml");

    for name in [
        "industry_comparison",
        "simd_benchmarks",
        "thread_schedule_comparison",
        "performance_benchmarks",
        "moirai_benchmarks",
        "public_result_handle_comparison",
        "example_pattern_comparison",
        "channel_matrix",
        "transport_archive_comparison",
        "result_handle_diagnostics",
        "iterator_adapter_comparison",
        "async_iterator_comparison",
        "sorting_comparison",
        "async_fs_comparison",
        "async_udp_comparison",
        "async_tcp_comparison",
        "async_tcp_backpressure_comparison",
        "async_tcp_readiness_comparison",
        "async_tcp_cancel_safety_comparison",
        "async_io_compat_comparison",
    ] {
        let section = format!("name = \"{name}\"");
        assert!(
            manifest.contains(&section),
            "benchmark manifest must include {name}"
        );
    }

    let harness_false_count = manifest.matches("harness = false").count();
    assert!(
        harness_false_count >= 5,
        "criterion benchmark targets must disable the default bench harness"
    );

    for relative in [
        "benches/industry_comparison.rs",
        "benches/thread_schedule_comparison.rs",
        "benches/performance_benchmarks.rs",
        "benches/moirai_benchmarks.rs",
        "benches/simd_benchmarks.rs",
        "benches/public_result_handle_comparison.rs",
        "benches/example_pattern_comparison.rs",
        "benches/channel_matrix.rs",
        "benches/transport_archive_comparison.rs",
        "benches/result_handle_diagnostics.rs",
        "benches/iterator_adapter_comparison.rs",
        "benches/async_iterator_comparison.rs",
        "benches/sorting_comparison.rs",
        "benches/async_fs_comparison.rs",
        "benches/async_udp_comparison.rs",
        "benches/async_tcp_comparison.rs",
        "benches/async_tcp_backpressure_comparison.rs",
        "benches/async_tcp_readiness_comparison.rs",
        "benches/async_tcp_cancel_safety_comparison.rs",
        "benches/async_io_compat_comparison.rs",
    ] {
        let source = read_benchmark(relative);
        assert!(
            source.contains("sample_size")
                && source.contains("measurement_time")
                && source.contains("warm_up_time"),
            "{relative} must bound Criterion sampling and measurement windows"
        );
    }

    let performance = read_benchmark("benches/performance_benchmarks.rs");
    assert!(
        performance.contains("without_plots"),
        "performance_benchmarks must disable plot generation so cargo bench exits under the verification gate"
    );
}

#[test]
fn version_artifacts_are_synchronized_for_current_target() {
    let manifest = read_benchmark("../Cargo.toml");
    let checklist = read_benchmark("../docs/checklist.md");
    let changelog = read_benchmark("../CHANGELOG.md");

    for required in [
        "version = \"0.2.0\"",
        "moirai-core = { path = \"moirai-core\", version = \"0.2.0\" }",
        "moirai-executor = { path = \"moirai-executor\", version = \"0.2.0\" }",
        "moirai-iter = { path = \"moirai-iter\", version = \"0.2.0\" }",
    ] {
        assert!(
            manifest.contains(required),
            "workspace manifest must retain synchronized target version entry {required}"
        );
    }

    assert!(
        checklist.contains("**Target Version**: 0.2.0"),
        "checklist target version must match Cargo workspace version"
    );
    assert!(
        changelog.contains("## [0.2.0] - 2026-05-24"),
        "changelog must contain a synchronized 0.2.0 section"
    );
}

#[test]
fn result_handle_diagnostics_separates_slot_and_scheduler_costs() {
    let source = read_result_handle_diagnostics();

    for required in [
        "direct_ready_result_slot",
        "direct_send_then_join_result_slot",
        "direct_cross_thread_result_slot",
        "direct_result_slot_ready_take",
        "direct_result_slot_spin_miss",
        "direct_result_slot_register_waiter",
        "direct_result_slot_complete_waiting",
        "direct_oversized_capture_read_one",
        "direct_oversized_captured_sum",
        "direct_boxed_oversized_capture_allocate_drop",
        "direct_boxed_oversized_capture_execute",
        "moirai_spawn_join_ready",
        "moirai_peer_spawn_join_ready",
        "moirai_spawn_join_captured_ready",
        "moirai_spawn_join_oversized_captured_ready",
        "moirai_peer_spawn_join_oversized_captured_ready",
        "moirai_spawn_join_oversized_capture_read_one",
        "moirai_spawn_blocking_ready",
        "moirai_spawn_blocking_oversized_captured_ready",
        "moirai_spawn_async_ready",
        "moirai_spawn_async_wake_once",
        "direct_async_idle_to_queued_state_claim",
        "direct_async_polling_to_notified_state_claim",
        "direct_async_notified_to_polling_state_claim",
        "direct_async_polling_to_idle_state_release",
        "direct_async_waker_from_arc",
        "direct_async_wake_by_ref_polling_notification",
        "direct_async_completed_state_store",
        "direct_async_future_present_drop_flag",
        "direct_async_lifecycle_complete",
        "direct_async_sender_cell_take_send_join",
        "direct_async_ready_completion_components",
        "hybrid_spawn_blocking_ready",
        "hybrid_peer_spawn_blocking_ready",
        "arc_hybrid_spawn_blocking_ready",
        "hybrid_spawn_blocking_captured_ready",
        "hybrid_spawn_blocking_oversized_captured_ready",
        "hybrid_peer_spawn_blocking_oversized_captured_ready",
        "arc_hybrid_spawn_blocking_oversized_captured_ready",
        "hybrid_spawn_blocking_oversized_capture_read_one",
        "moirai_spawn_join_ready_with_quiescent_barrier",
        "direct_scheduler_submit_join",
        "direct_scheduler_submission_queue_publication",
        "direct_scheduler_worker_execute_ready_job",
        "direct_scheduler_worker_local_dequeue_execute",
        "direct_scheduler_max_inline_job_construct_drop",
        "direct_scheduler_max_inline_job_construct_execute",
        "direct_scheduler_oversized_job_construct_drop",
        "direct_scheduler_oversized_job_construct_execute",
        "direct_scheduler_max_inline_queue_push_pop_execute",
        "direct_scheduler_oversized_queue_push_pop_execute",
        "direct_scheduler_worker_local_max_inline_dequeue_execute",
        "direct_scheduler_worker_local_oversized_dequeue_execute",
        "direct_scheduler_join_fast_spin_quiescent",
        "direct_scheduler_join_fast_spin_pending",
        "direct_scheduler_empty_wake_decision",
        "direct_scheduler_contended_wake_decision",
        "direct_scheduler_saturated_wake_decision",
        "direct_spawn_metrics_before_scheduler_submission",
        "direct_spawn_metrics_after_scheduler_submission",
        "direct_scheduler_ready_atomic_join",
        "direct_scheduler_max_inline_atomic_join",
        "direct_scheduler_oversized_atomic_join",
        "direct_scheduler_worker_start_signal",
        "direct_scheduler_worker_start_then_result_slot",
        "direct_scheduler_result_slot",
        "direct_scheduler_boxed_ready_result_slot",
        "direct_scheduler_captured_result_slot",
        "direct_scheduler_max_inline_captured_result_slot",
        "direct_scheduler_oversized_captured_result_slot",
        "direct_scheduler_oversized_capture_read_one_result_slot",
        "direct_scheduler_oversized_result_slot_with_quiescent_barrier",
        "direct_scheduler_result_slot_with_metrics_tail",
        "direct_scheduler_oversized_result_slot_with_metrics_tail",
        "direct_scheduler_lifecycle_before_send_result_slot",
        "direct_scheduler_lifecycle_elapsed_only_result_slot",
        "direct_scheduler_lifecycle_atomic_only_result_slot",
        "direct_scheduler_lifecycle_start_instant_result_slot",
        "direct_scheduler_lifecycle_cached_clock_result_slot",
        "direct_scheduler_lifecycle_qpc_result_slot",
        "direct_scheduler_lifecycle_duration_only_result_slot",
        "direct_scheduler_lifecycle_after_send_result_slot",
        "direct_scheduler_oversized_lifecycle_before_send_result_slot",
        "direct_scheduler_oversized_lifecycle_elapsed_only_result_slot",
        "direct_scheduler_oversized_lifecycle_atomic_only_result_slot",
        "direct_scheduler_oversized_lifecycle_start_instant_result_slot",
        "direct_scheduler_oversized_lifecycle_cached_clock_result_slot",
        "direct_scheduler_oversized_lifecycle_qpc_result_slot",
        "direct_scheduler_oversized_lifecycle_duration_only_result_slot",
        "direct_scheduler_oversized_lifecycle_after_send_result_slot",
        "ElapsedOnlyLifecycle",
        "AtomicOnlyLifecycle",
        "StartInstantLifecycle",
        "CachedClockLifecycle",
        "CachedLifecycleClockGuard",
        "QpcLifecycle",
        "DurationOnlyLifecycle",
        "direct_scheduler_tail_after_send_result_slot",
        "direct_scheduler_tail_after_send_with_quiescent_barrier",
        "direct_scheduler_pinned_oversized_captured_result_slot",
        "direct_scheduler_pinned_oversized_capture_read_one_result_slot",
        "direct_scheduler_affinity_oversized_captured_result_slot",
        "BLOCKING_NORMAL_WORKER",
        "direct_scheduler_result_slot_with_quiescent_barrier",
        "direct_task_id_allocate",
        "direct_metrics_record_task_spawned",
        "direct_metrics_record_task_completed",
        "direct_public_wrapper_without_metrics",
        "direct_public_wrapper_components",
        "direct_public_token_wrapper_components",
        "direct_public_token_wrapper_after_send_components",
        "direct_public_wrapper_oversized_captured_components",
        "direct_public_wrapper_oversized_capture_read_one_components",
        "direct_scheduled_public_token_wrapper_components",
        "direct_scheduled_public_token_wrapper_without_metrics",
        "direct_scheduled_public_token_wrapper_without_catch",
        "direct_scheduled_public_token_wrapper_atomic_result",
        "direct_scheduled_public_token_wrapper_without_lifecycle",
        "direct_scheduled_public_token_wrapper_oversized_components",
        "direct_scheduled_public_token_wrapper_oversized_storage_only",
        "direct_scheduled_public_token_wrapper_oversized_read_one_components",
        "direct_scheduled_public_token_wrapper_oversized_without_metrics",
        "direct_registry_lifecycle",
        "direct_external_id_registry_register",
        "registry_mutex_lock_only",
        "registry_block_lookup",
        "registry_slot_initialize",
        "registry_lifecycle_timestamp_publication",
        "registry_elapsed_nanos_since_origin",
        "registry_start_release_publication",
        "registry_completion_release_publication",
        "registry_duration_offset_math",
        "registry_task_state_construct",
        "registry_mark_started_existing_slot",
        "registry_mark_completed_existing_slot",
        "diagnostic_block_lookup",
        "diagnostic_slot_initialize",
        "diagnostic_lifecycle_timestamp_publication",
        "diagnostic_task_state_construct",
        "diagnostic_mark_started",
        "diagnostic_mark_completed_since",
        "mutex_registry_register",
        "oversized_capture_read_one",
        "oversized_capture_sum",
        "boxed_ready_value",
        "max_inline_capture_sum",
        "MAX_INLINE_CAPTURE_WORDS",
        "ExecutorMetrics::new",
        "record_task_spawned",
        "record_task_completed",
        "catch_unwind(AssertUnwindSafe",
        "CAPTURE_WORDS",
        "OVERSIZED_CAPTURE_WORDS",
        "TaskHandle::ready",
        "TaskHandle::new_pending",
        "HybridExecutor::new",
        "ExecutorConfig",
        ".spawn_blocking",
        "TaskRegistry::new",
        "ThreadScheduler::new",
        "schedule::<BlockingTask",
        "thread::spawn",
        "Moirai::builder",
        ".spawn_fn",
        ".join()",
        "verify_ready_value",
        "verify_captured_ready_value",
        "verify_oversized_captured_ready_value",
        "without_plots",
    ] {
        assert!(
            source.contains(required),
            "result-handle diagnostic benchmark must contain {required}"
        );
    }
}
