fn direct_task_id_allocate(next_task_id: &AtomicU64) -> usize {
    black_box(next_task_id.fetch_add(1, Ordering::Relaxed) as usize)
}

fn direct_metrics_record_task_spawned(metrics: &ExecutorMetrics) -> usize {
    metrics.record_task_spawned();
    black_box(metrics.tasks_spawned.load(Ordering::Relaxed) as usize)
}

fn direct_metrics_record_task_completed(metrics: &ExecutorMetrics) -> usize {
    metrics.record_task_completed(Duration::from_nanos(black_box(READY_VALUE as u64)));
    black_box(metrics.tasks_completed.load(Ordering::Relaxed) as usize)
}
