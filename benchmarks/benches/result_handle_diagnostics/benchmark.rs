pub(crate) fn benchmark_result_handle_diagnostics(c: &mut Criterion) {
    let mut group = c.benchmark_group("result_handle_diagnostics");
    group.sample_size(BENCHMARK_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(BENCHMARK_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(BENCHMARK_WARM_UP_MILLIS));

    let moirai = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("Moirai runtime must start");
    let moirai_peer = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("peer Moirai runtime must start");
    let mut executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: WORKER_THREADS,
        thread_name_prefix: "result-handle-hybrid".into(),
        ..ExecutorConfig::default()
    })
    .expect("HybridExecutor diagnostic runtime must start");
    let mut executor_peer = HybridExecutor::new(ExecutorConfig {
        worker_threads: WORKER_THREADS,
        thread_name_prefix: "result-handle-hybrid-peer".into(),
        ..ExecutorConfig::default()
    })
    .expect("peer HybridExecutor diagnostic runtime must start");
    let arc_executor = Arc::new(
        HybridExecutor::new(ExecutorConfig {
            worker_threads: WORKER_THREADS,
            thread_name_prefix: "result-handle-arc-hybrid".into(),
            ..ExecutorConfig::default()
        })
        .expect("Arc HybridExecutor diagnostic runtime must start"),
    );
    let scheduler = ThreadScheduler::new(WORKER_THREADS, "result-handle-diagnostic")
        .expect("diagnostic scheduler must start");

    assert_eq!(moirai_spawn_join_ready(&moirai_peer), READY_VALUE);
    assert_eq!(hybrid_spawn_blocking_ready(&executor_peer), READY_VALUE);

    group.bench_function("direct_ready_result_slot", |bench| {
        bench.iter(direct_ready_result_slot);
    });

    group.bench_function("direct_send_then_join_result_slot", |bench| {
        bench.iter(direct_send_then_join_result_slot);
    });

    group.bench_function("direct_cross_thread_result_slot", |bench| {
        bench.iter(direct_cross_thread_result_slot);
    });

    #[cfg(feature = "result-diagnostics")]
    group.bench_function("direct_result_slot_ready_take", |bench| {
        bench.iter(direct_result_slot_ready_take);
    });

    #[cfg(feature = "result-diagnostics")]
    group.bench_function("direct_result_slot_spin_miss", |bench| {
        bench.iter(direct_result_slot_spin_miss);
    });

    #[cfg(feature = "result-diagnostics")]
    group.bench_function("direct_result_slot_register_waiter", |bench| {
        bench.iter(direct_result_slot_register_waiter);
    });

    #[cfg(feature = "result-diagnostics")]
    group.bench_function("direct_result_slot_complete_waiting", |bench| {
        bench.iter(direct_result_slot_complete_waiting);
    });

    group.bench_function("direct_oversized_capture_read_one", |bench| {
        bench.iter(direct_oversized_capture_read_one);
    });

    group.bench_function("direct_oversized_captured_sum", |bench| {
        bench.iter(direct_oversized_captured_sum);
    });

    group.bench_function("direct_boxed_oversized_capture_allocate_drop", |bench| {
        bench.iter(direct_boxed_oversized_capture_allocate_drop);
    });

    group.bench_function("direct_boxed_oversized_capture_execute", |bench| {
        bench.iter(direct_boxed_oversized_capture_execute);
    });

    group.bench_function("moirai_spawn_join_ready", |bench| {
        bench.iter(|| moirai_spawn_join_ready(&moirai));
    });

    group.bench_function("moirai_peer_spawn_join_ready", |bench| {
        bench.iter(|| moirai_spawn_join_ready(&moirai_peer));
    });

    group.bench_function("moirai_spawn_join_captured_ready", |bench| {
        bench.iter(|| moirai_spawn_join_captured_ready(&moirai));
    });

    group.bench_function("moirai_spawn_join_oversized_captured_ready", |bench| {
        bench.iter(|| moirai_spawn_join_oversized_captured_ready(&moirai));
    });

    group.bench_function(
        "moirai_peer_spawn_join_oversized_captured_ready",
        |bench| {
            bench.iter(|| moirai_spawn_join_oversized_captured_ready(&moirai_peer));
        },
    );

    group.bench_function("moirai_spawn_join_oversized_capture_read_one", |bench| {
        bench.iter(|| moirai_spawn_join_oversized_capture_read_one(&moirai));
    });

    group.bench_function("moirai_spawn_blocking_ready", |bench| {
        bench.iter(|| moirai_spawn_blocking_ready(&moirai));
    });

    group.bench_function(
        "moirai_spawn_blocking_oversized_captured_ready",
        |bench| {
            bench.iter(|| moirai_spawn_blocking_oversized_captured_ready(&moirai));
        },
    );

    benchmark_async_state_diagnostics(&mut group, &moirai);

    group.bench_function("hybrid_spawn_blocking_ready", |bench| {
        bench.iter(|| hybrid_spawn_blocking_ready(&executor));
    });

    group.bench_function("hybrid_peer_spawn_blocking_ready", |bench| {
        bench.iter(|| hybrid_spawn_blocking_ready(&executor_peer));
    });

    group.bench_function("arc_hybrid_spawn_blocking_ready", |bench| {
        bench.iter(|| arc_hybrid_spawn_blocking_ready(&arc_executor));
    });

    group.bench_function("hybrid_spawn_blocking_captured_ready", |bench| {
        bench.iter(|| hybrid_spawn_blocking_captured_ready(&executor));
    });

    group.bench_function("hybrid_spawn_blocking_oversized_captured_ready", |bench| {
        bench.iter(|| hybrid_spawn_blocking_oversized_captured_ready(&executor));
    });

    group.bench_function(
        "hybrid_peer_spawn_blocking_oversized_captured_ready",
        |bench| {
            bench.iter(|| hybrid_spawn_blocking_oversized_captured_ready(&executor_peer));
        },
    );

    group.bench_function(
        "arc_hybrid_spawn_blocking_oversized_captured_ready",
        |bench| {
            bench.iter(|| arc_hybrid_spawn_blocking_oversized_captured_ready(&arc_executor));
        },
    );

    group.bench_function(
        "hybrid_spawn_blocking_oversized_capture_read_one",
        |bench| {
            bench.iter(|| hybrid_spawn_blocking_oversized_capture_read_one(&executor));
        },
    );

    group.bench_function("moirai_spawn_join_ready_with_quiescent_barrier", |bench| {
        bench.iter(|| moirai_spawn_join_ready_with_quiescent_barrier(&moirai));
    });

    group.bench_function("direct_scheduler_submit_join", |bench| {
        bench.iter(|| direct_scheduler_submit_join(&scheduler));
    });

    benchmark_scheduler_submission_diagnostics(&mut group, &scheduler);

    group.bench_function("direct_scheduler_ready_atomic_join", |bench| {
        bench.iter(|| direct_scheduler_ready_atomic_join(&scheduler));
    });

    group.bench_function("direct_scheduler_max_inline_atomic_join", |bench| {
        bench.iter(|| direct_scheduler_max_inline_atomic_join(&scheduler));
    });

    group.bench_function("direct_scheduler_oversized_atomic_join", |bench| {
        bench.iter(|| direct_scheduler_oversized_atomic_join(&scheduler));
    });

    group.bench_function("direct_scheduler_worker_start_signal", |bench| {
        bench.iter(|| direct_scheduler_worker_start_signal(&scheduler));
    });

    group.bench_function("direct_scheduler_worker_start_then_result_slot", |bench| {
        bench.iter(|| direct_scheduler_worker_start_then_result_slot(&scheduler));
    });

    group.bench_function("direct_scheduler_result_slot", |bench| {
        bench.iter(|| direct_scheduler_result_slot(&scheduler));
    });

    group.bench_function("direct_scheduler_boxed_ready_result_slot", |bench| {
        bench.iter(|| direct_scheduler_boxed_ready_result_slot(&scheduler));
    });

    group.bench_function("direct_scheduler_captured_result_slot", |bench| {
        bench.iter(|| direct_scheduler_captured_result_slot(&scheduler));
    });

    group.bench_function(
        "direct_scheduler_max_inline_captured_result_slot",
        |bench| {
            bench.iter(|| direct_scheduler_max_inline_captured_result_slot(&scheduler));
        },
    );

    group.bench_function("direct_scheduler_oversized_captured_result_slot", |bench| {
        bench.iter(|| direct_scheduler_oversized_captured_result_slot(&scheduler));
    });

    group.bench_function(
        "direct_scheduler_oversized_capture_read_one_result_slot",
        |bench| {
            bench.iter(|| direct_scheduler_oversized_capture_read_one_result_slot(&scheduler));
        },
    );

    group.bench_function(
        "direct_scheduler_oversized_result_slot_with_quiescent_barrier",
        |bench| {
            bench
                .iter(|| direct_scheduler_oversized_result_slot_with_quiescent_barrier(&scheduler));
        },
    );

    group.bench_function("direct_scheduler_result_slot_with_metrics_tail", |bench| {
        let metrics = Arc::new(ExecutorMetrics::new());
        bench.iter(|| direct_scheduler_result_slot_with_metrics_tail(&scheduler, &metrics));
    });

    group.bench_function(
        "direct_scheduler_oversized_result_slot_with_metrics_tail",
        |bench| {
            let metrics = Arc::new(ExecutorMetrics::new());
            bench.iter(|| {
                direct_scheduler_oversized_result_slot_with_metrics_tail(&scheduler, &metrics)
            });
        },
    );

    group.bench_function(
        "direct_scheduler_lifecycle_before_send_result_slot",
        |bench| {
            bench.iter(|| direct_scheduler_lifecycle_before_send_result_slot(&scheduler));
        },
    );

    group.bench_function(
        "direct_scheduler_lifecycle_elapsed_only_result_slot",
        |bench| {
            bench.iter(|| direct_scheduler_lifecycle_elapsed_only_result_slot(&scheduler));
        },
    );

    group.bench_function(
        "direct_scheduler_lifecycle_atomic_only_result_slot",
        |bench| {
            bench.iter(|| direct_scheduler_lifecycle_atomic_only_result_slot(&scheduler));
        },
    );

    group.bench_function(
        "direct_scheduler_lifecycle_start_instant_result_slot",
        |bench| {
            bench.iter(|| direct_scheduler_lifecycle_start_instant_result_slot(&scheduler));
        },
    );

    {
        let cached_lifecycle_clock = CachedLifecycleClockGuard::start();
        group.bench_function(
            "direct_scheduler_lifecycle_cached_clock_result_slot",
            |bench| {
                bench.iter(|| {
                    direct_scheduler_lifecycle_cached_clock_result_slot(
                        &scheduler,
                        cached_lifecycle_clock.clock(),
                    )
                });
            },
        );
    }

    #[cfg(windows)]
    group.bench_function("direct_scheduler_lifecycle_qpc_result_slot", |bench| {
        bench.iter(|| direct_scheduler_lifecycle_qpc_result_slot(&scheduler));
    });

    group.bench_function(
        "direct_scheduler_lifecycle_duration_only_result_slot",
        |bench| {
            bench.iter(|| direct_scheduler_lifecycle_duration_only_result_slot(&scheduler));
        },
    );

    group.bench_function(
        "direct_scheduler_lifecycle_after_send_result_slot",
        |bench| {
            bench.iter(|| direct_scheduler_lifecycle_after_send_result_slot(&scheduler));
        },
    );

    group.bench_function(
        "direct_scheduler_oversized_lifecycle_before_send_result_slot",
        |bench| {
            bench.iter(|| direct_scheduler_oversized_lifecycle_before_send_result_slot(&scheduler));
        },
    );

    group.bench_function(
        "direct_scheduler_oversized_lifecycle_elapsed_only_result_slot",
        |bench| {
            bench
                .iter(|| direct_scheduler_oversized_lifecycle_elapsed_only_result_slot(&scheduler));
        },
    );

    group.bench_function(
        "direct_scheduler_oversized_lifecycle_atomic_only_result_slot",
        |bench| {
            bench.iter(|| direct_scheduler_oversized_lifecycle_atomic_only_result_slot(&scheduler));
        },
    );

    group.bench_function(
        "direct_scheduler_oversized_lifecycle_start_instant_result_slot",
        |bench| {
            bench.iter(|| {
                direct_scheduler_oversized_lifecycle_start_instant_result_slot(&scheduler)
            });
        },
    );

    {
        let cached_lifecycle_clock = CachedLifecycleClockGuard::start();
        group.bench_function(
            "direct_scheduler_oversized_lifecycle_cached_clock_result_slot",
            |bench| {
                bench.iter(|| {
                    direct_scheduler_oversized_lifecycle_cached_clock_result_slot(
                        &scheduler,
                        cached_lifecycle_clock.clock(),
                    )
                });
            },
        );
    }

    #[cfg(windows)]
    group.bench_function(
        "direct_scheduler_oversized_lifecycle_qpc_result_slot",
        |bench| {
            bench.iter(|| direct_scheduler_oversized_lifecycle_qpc_result_slot(&scheduler));
        },
    );

    group.bench_function(
        "direct_scheduler_oversized_lifecycle_duration_only_result_slot",
        |bench| {
            bench.iter(|| {
                direct_scheduler_oversized_lifecycle_duration_only_result_slot(&scheduler)
            });
        },
    );

    group.bench_function(
        "direct_scheduler_oversized_lifecycle_after_send_result_slot",
        |bench| {
            bench.iter(|| direct_scheduler_oversized_lifecycle_after_send_result_slot(&scheduler));
        },
    );

    group.bench_function("direct_scheduler_tail_after_send_result_slot", |bench| {
        bench.iter(|| direct_scheduler_tail_after_send_result_slot(&scheduler));
    });

    group.bench_function(
        "direct_scheduler_tail_after_send_with_quiescent_barrier",
        |bench| {
            bench.iter(|| direct_scheduler_tail_after_send_with_quiescent_barrier(&scheduler));
        },
    );

    group.bench_function(
        "direct_scheduler_pinned_oversized_captured_result_slot",
        |bench| {
            bench.iter(|| direct_scheduler_pinned_oversized_captured_result_slot(&scheduler));
        },
    );

    group.bench_function(
        "direct_scheduler_pinned_oversized_capture_read_one_result_slot",
        |bench| {
            bench.iter(|| {
                direct_scheduler_pinned_oversized_capture_read_one_result_slot(&scheduler)
            });
        },
    );

    group.bench_function(
        "direct_scheduler_affinity_oversized_captured_result_slot",
        |bench| {
            bench.iter(|| direct_scheduler_affinity_oversized_captured_result_slot(&scheduler));
        },
    );

    group.bench_function(
        "direct_scheduler_result_slot_with_quiescent_barrier",
        |bench| {
            bench.iter(|| direct_scheduler_result_slot_with_quiescent_barrier(&scheduler));
        },
    );

    group.bench_function("direct_task_id_allocate", |bench| {
        let next_task_id = AtomicU64::new(1);
        bench.iter(|| direct_task_id_allocate(&next_task_id));
    });

    group.bench_function("direct_metrics_record_task_spawned", |bench| {
        let metrics = ExecutorMetrics::new();
        bench.iter(|| direct_metrics_record_task_spawned(&metrics));
    });

    group.bench_function("direct_metrics_record_task_completed", |bench| {
        let metrics = ExecutorMetrics::new();
        bench.iter(|| direct_metrics_record_task_completed(&metrics));
    });

    group.bench_function("direct_public_wrapper_without_metrics", |bench| {
        let mut registry = TaskRegistry::new();
        bench.iter(|| direct_public_wrapper_without_metrics(&mut registry));
    });

    group.bench_function("direct_public_wrapper_components", |bench| {
        let mut registry = TaskRegistry::new();
        let metrics = ExecutorMetrics::new();
        bench.iter(|| direct_public_wrapper_components(&mut registry, &metrics));
    });

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function("direct_public_token_wrapper_components", |bench| {
        let mut registry = TaskRegistry::new();
        let next_task_id = AtomicU64::new(1);
        let metrics = ExecutorMetrics::new();
        bench.iter(|| {
            direct_public_token_wrapper_components(&mut registry, &next_task_id, &metrics)
        });
    });

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function("direct_public_token_wrapper_after_send_components", |bench| {
        let mut registry = TaskRegistry::new();
        let next_task_id = AtomicU64::new(1);
        let metrics = ExecutorMetrics::new();
        bench.iter(|| {
            direct_public_token_wrapper_after_send_components(&mut registry, &next_task_id, &metrics)
        });
    });

    group.bench_function(
        "direct_public_wrapper_oversized_captured_components",
        |bench| {
            let mut registry = TaskRegistry::new();
            let metrics = ExecutorMetrics::new();
            bench.iter(|| {
                direct_public_wrapper_oversized_captured_components(&mut registry, &metrics)
            });
        },
    );

    group.bench_function(
        "direct_public_wrapper_oversized_capture_read_one_components",
        |bench| {
            let mut registry = TaskRegistry::new();
            let metrics = ExecutorMetrics::new();
            bench.iter(|| {
                direct_public_wrapper_oversized_capture_read_one_components(&mut registry, &metrics)
            });
        },
    );

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function("direct_scheduled_public_token_wrapper_components", |bench| {
        let mut registry = TaskRegistry::new();
        let next_task_id = AtomicU64::new(1);
        let metrics = Arc::new(ExecutorMetrics::new());
        bench.iter(|| {
            direct_scheduled_public_token_wrapper_components(
                &scheduler,
                &mut registry,
                &next_task_id,
                &metrics,
            )
        });
    });

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function(
        "direct_scheduled_public_token_wrapper_without_metrics",
        |bench| {
            let mut registry = TaskRegistry::new();
            let next_task_id = AtomicU64::new(1);
            bench.iter(|| {
                direct_scheduled_public_token_wrapper_without_metrics(
                    &scheduler,
                    &mut registry,
                    &next_task_id,
                )
            });
        },
    );

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function(
        "direct_scheduled_public_token_wrapper_without_catch",
        |bench| {
            let mut registry = TaskRegistry::new();
            let next_task_id = AtomicU64::new(1);
            let metrics = Arc::new(ExecutorMetrics::new());
            bench.iter(|| {
                direct_scheduled_public_token_wrapper_without_catch(
                    &scheduler,
                    &mut registry,
                    &next_task_id,
                    &metrics,
                )
            });
        },
    );

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function(
        "direct_scheduled_public_token_wrapper_atomic_result",
        |bench| {
            let mut registry = TaskRegistry::new();
            let next_task_id = AtomicU64::new(1);
            let metrics = Arc::new(ExecutorMetrics::new());
            bench.iter(|| {
                direct_scheduled_public_token_wrapper_atomic_result(
                    &scheduler,
                    &mut registry,
                    &next_task_id,
                    &metrics,
                )
            });
        },
    );

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function(
        "direct_scheduled_public_token_wrapper_without_lifecycle",
        |bench| {
            let next_task_id = AtomicU64::new(1);
            bench.iter(|| {
                direct_scheduled_public_token_wrapper_without_lifecycle(
                    &scheduler,
                    &next_task_id,
                )
            });
        },
    );

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function(
        "direct_scheduled_public_token_wrapper_oversized_components",
        |bench| {
            let mut registry = TaskRegistry::new();
            let next_task_id = AtomicU64::new(1);
            let metrics = Arc::new(ExecutorMetrics::new());
            bench.iter(|| {
                direct_scheduled_public_token_wrapper_oversized_components(
                    &scheduler,
                    &mut registry,
                    &next_task_id,
                    &metrics,
                )
            });
        },
    );

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function(
        "direct_scheduled_public_token_wrapper_oversized_storage_only",
        |bench| {
            let mut registry = TaskRegistry::new();
            let next_task_id = AtomicU64::new(1);
            let metrics = Arc::new(ExecutorMetrics::new());
            bench.iter(|| {
                direct_scheduled_public_token_wrapper_oversized_storage_only(
                    &scheduler,
                    &mut registry,
                    &next_task_id,
                    &metrics,
                )
            });
        },
    );

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function(
        "direct_scheduled_public_token_wrapper_oversized_read_one_components",
        |bench| {
            let mut registry = TaskRegistry::new();
            let next_task_id = AtomicU64::new(1);
            let metrics = Arc::new(ExecutorMetrics::new());
            bench.iter(|| {
                direct_scheduled_public_token_wrapper_oversized_read_one_components(
                    &scheduler,
                    &mut registry,
                    &next_task_id,
                    &metrics,
                )
            });
        },
    );

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function(
        "direct_scheduled_public_token_wrapper_oversized_without_metrics",
        |bench| {
            let mut registry = TaskRegistry::new();
            let next_task_id = AtomicU64::new(1);
            bench.iter(|| {
                direct_scheduled_public_token_wrapper_oversized_without_metrics(
                    &scheduler,
                    &mut registry,
                    &next_task_id,
                )
            });
        },
    );

    group.bench_function("direct_registry_lifecycle", |bench| {
        let mut registry = TaskRegistry::new();
        bench.iter(|| direct_registry_lifecycle(&mut registry));
    });

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function("direct_external_id_registry_register", |bench| {
        let mut registry = TaskRegistry::new();
        let next_task_id = AtomicU64::new(1);
        bench.iter(|| direct_external_id_registry_register(&mut registry, &next_task_id));
    });

    group.bench_function("registry_mutex_lock_only", |bench| {
        let registry = Mutex::new(TaskRegistry::new());
        bench.iter(|| registry_mutex_lock_only(&registry));
    });

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function("registry_block_lookup", |bench| {
        let mut registry = TaskRegistry::new();
        bench.iter(|| registry_block_lookup(&mut registry));
    });

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function("registry_slot_initialize", |bench| {
        let mut registry = TaskRegistry::new();
        bench.iter(|| registry_slot_initialize(&mut registry));
    });

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function("registry_lifecycle_timestamp_publication", |bench| {
        bench.iter(registry_lifecycle_timestamp_publication);
    });

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function("registry_elapsed_nanos_since_origin", |bench| {
        let origin = Instant::now();
        bench.iter(|| registry_elapsed_nanos_since_origin(origin));
    });

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function("registry_start_release_publication", |bench| {
        let started_after_ns = AtomicUsize::new(LIFECYCLE_TIMESTAMP_NOT_RECORDED);
        let worker_id = AtomicUsize::new(usize::MAX);
        bench.iter(|| registry_start_release_publication(&started_after_ns, &worker_id));
    });

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function("registry_completion_release_publication", |bench| {
        let completed_after_ns = AtomicUsize::new(LIFECYCLE_TIMESTAMP_NOT_RECORDED);
        bench.iter(|| registry_completion_release_publication(&completed_after_ns));
    });

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function("registry_duration_offset_math", |bench| {
        bench.iter(registry_duration_offset_math);
    });

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function("registry_task_state_construct", |bench| {
        bench.iter(registry_task_state_construct);
    });

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function("registry_mark_started_existing_slot", |bench| {
        let mut registry = TaskRegistry::new();
        let task_id = registry.register_task();
        bench.iter(|| registry_mark_started_existing_slot(&registry, task_id));
    });

    #[cfg(feature = "registry-diagnostics")]
    group.bench_function("registry_mark_completed_existing_slot", |bench| {
        let mut registry = TaskRegistry::new();
        let task_id = registry.register_task();
        let started_after_ns = registry.diagnostic_mark_started(task_id, 0);
        bench.iter(|| {
            registry_mark_completed_existing_slot(&registry, task_id, started_after_ns)
        });
    });

    group.bench_function("mutex_registry_register", |bench| {
        let registry = Mutex::new(TaskRegistry::new());
        bench.iter(|| mutex_registry_register(&registry));
    });

    group.finish();
    scheduler.shutdown();
    executor_peer
        .shutdown()
        .expect("peer HybridExecutor diagnostic runtime must shut down");
    executor
        .shutdown()
        .expect("HybridExecutor diagnostic runtime must shut down");
    moirai_peer.shutdown();
    moirai.shutdown();
}
