use criterion::{measurement::WallTime, BenchmarkGroup};

pub(crate) fn benchmark_scheduler_submission_diagnostics(
    group: &mut BenchmarkGroup<'_, WallTime>,
    scheduler: &ThreadScheduler,
) {
    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_select_worker_serial", |bench| {
        bench.iter(|| direct_scheduler_select_worker_serial(scheduler));
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_pending_counter_pair", |bench| {
        bench.iter(|| direct_scheduler_pending_counter_pair(scheduler));
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_worker_unpark", |bench| {
        bench.iter(|| direct_scheduler_worker_unpark(scheduler));
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_priority_queue_push_pop", |bench| {
        bench.iter(direct_scheduler_priority_queue_push_pop);
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_submission_queue_publication", |bench| {
        bench.iter(|| direct_scheduler_submission_queue_publication(scheduler));
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_worker_execute_ready_job", |bench| {
        bench.iter(|| direct_scheduler_worker_execute_ready_job(scheduler));
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_worker_local_dequeue_execute", |bench| {
        bench.iter(|| direct_scheduler_worker_local_dequeue_execute(scheduler));
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_max_inline_job_construct_drop", |bench| {
        bench.iter(direct_scheduler_max_inline_job_construct_drop);
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_max_inline_job_construct_execute", |bench| {
        bench.iter(direct_scheduler_max_inline_job_construct_execute);
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_oversized_job_construct_drop", |bench| {
        bench.iter(direct_scheduler_oversized_job_construct_drop);
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_oversized_job_construct_execute", |bench| {
        bench.iter(direct_scheduler_oversized_job_construct_execute);
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_max_inline_queue_push_pop_execute", |bench| {
        bench.iter(direct_scheduler_max_inline_queue_push_pop_execute);
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_oversized_queue_push_pop_execute", |bench| {
        bench.iter(direct_scheduler_oversized_queue_push_pop_execute);
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function(
        "direct_scheduler_worker_local_max_inline_dequeue_execute",
        |bench| {
            bench.iter(|| direct_scheduler_worker_local_max_inline_dequeue_execute(scheduler));
        },
    );

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function(
        "direct_scheduler_worker_local_oversized_dequeue_execute",
        |bench| {
            bench.iter(|| direct_scheduler_worker_local_oversized_dequeue_execute(scheduler));
        },
    );

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_join_fast_spin_quiescent", |bench| {
        bench.iter(|| direct_scheduler_join_fast_spin_quiescent(scheduler));
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_join_fast_spin_pending", |bench| {
        bench.iter(|| direct_scheduler_join_fast_spin_pending(scheduler));
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_empty_wake_decision", |bench| {
        bench.iter(|| direct_scheduler_empty_wake_decision(scheduler));
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_contended_wake_decision", |bench| {
        bench.iter(|| direct_scheduler_contended_wake_decision(scheduler));
    });

    #[cfg(feature = "scheduler-diagnostics")]
    group.bench_function("direct_scheduler_saturated_wake_decision", |bench| {
        bench.iter(|| direct_scheduler_saturated_wake_decision(scheduler));
    });

    group.bench_function("direct_spawn_metrics_before_scheduler_submission", |bench| {
        let metrics = ExecutorMetrics::new();
        bench.iter(|| direct_spawn_metrics_before_scheduler_submission(scheduler, &metrics));
    });

    group.bench_function("direct_spawn_metrics_after_scheduler_submission", |bench| {
        let metrics = ExecutorMetrics::new();
        bench.iter(|| direct_spawn_metrics_after_scheduler_submission(scheduler, &metrics));
    });
}
