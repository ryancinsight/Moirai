#[cfg(feature = "registry-diagnostics")]
fn direct_scheduled_public_token_wrapper_components(
    scheduler: &ThreadScheduler,
    registry: &TaskRegistry,
    metrics: &Arc<ExecutorMetrics>,
) -> usize {
    let (id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();
    let worker_metrics = Arc::clone(metrics);

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));
            match task_result {
                Ok(value) => {
                    sender.send(Ok(value));
                    worker_metrics.record_task_completed(execution_time);
                }
                Err(_) => {
                    sender.send(Err(TaskError::Panicked));
                    worker_metrics.record_task_failed();
                }
            }
        })
        .expect("scheduler must accept scheduled token wrapper diagnostic job");

    let result = handle
        .join()
        .expect("scheduled token wrapper handle must be attached")
        .expect("scheduled token wrapper handle must contain a value");

    verify_ready_value(result)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_scheduled_public_registry_token_wrapper_components(
    scheduler: &ThreadScheduler,
    registry: &TaskRegistry,
    metrics: &Arc<ExecutorMetrics>,
) -> usize {
    let (id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();
    let worker_metrics = Arc::clone(metrics);

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));
            match task_result {
                Ok(value) => {
                    sender.send(Ok(value));
                    worker_metrics.record_task_completed(execution_time);
                }
                Err(_) => {
                    sender.send(Err(TaskError::Panicked));
                    worker_metrics.record_task_failed();
                }
            }
        })
        .expect("scheduler must accept scheduled registry-token wrapper diagnostic job");

    let result = handle
        .join()
        .expect("scheduled registry-token wrapper handle must be attached")
        .expect("scheduled registry-token wrapper handle must contain a value");

    verify_ready_value(result)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_scheduled_public_registry_token_wrapper_after_send_quiescent(
    scheduler: &ThreadScheduler,
    registry: &TaskRegistry,
    metrics: &Arc<ExecutorMetrics>,
) -> usize {
    let (id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();
    let worker_metrics = Arc::clone(metrics);

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));
            match task_result {
                Ok(value) => {
                    sender.send(Ok(value));
                    worker_metrics.record_task_completed(execution_time);
                }
                Err(_) => {
                    sender.send(Err(TaskError::Panicked));
                    worker_metrics.record_task_failed();
                }
            }
        })
        .expect("scheduler must accept scheduled registry-token after-send quiescent wrapper job");

    let result = handle
        .join()
        .expect("scheduled registry-token after-send quiescent wrapper handle must be attached")
        .expect(
            "scheduled registry-token after-send quiescent wrapper handle must contain a value",
        );
    scheduler
        .join()
        .expect("scheduler must reach quiescence after registry-token metrics tail");

    verify_ready_value(result)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_scheduled_public_registry_token_wrapper_local_metrics_quiescent(
    scheduler: &ThreadScheduler,
    registry: &TaskRegistry,
) -> usize {
    let (id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    let spawned_count = black_box(1u64);

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));
            let mut completed_count = 0u64;
            let mut failed_count = 0u64;
            let mut total_execution_ns = 0u64;
            match task_result {
                Ok(value) => {
                    sender.send(Ok(value));
                    completed_count += 1;
                    total_execution_ns += execution_time.as_nanos() as u64;
                }
                Err(_) => {
                    sender.send(Err(TaskError::Panicked));
                    failed_count += 1;
                }
            }
            black_box((
                spawned_count,
                completed_count,
                failed_count,
                total_execution_ns,
            ));
        })
        .expect("scheduler must accept scheduled registry-token local-metrics wrapper job");

    let result = handle
        .join()
        .expect("scheduled registry-token local-metrics wrapper handle must be attached")
        .expect("scheduled registry-token local-metrics wrapper handle must contain a value");
    scheduler
        .join()
        .expect("scheduler must reach quiescence after registry-token local metrics tail");

    verify_ready_value(result)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_scheduled_public_token_wrapper_without_metrics(
    scheduler: &ThreadScheduler,
    registry: &TaskRegistry,
) -> usize {
    let (id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            black_box(execution_time);
            let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));
            match task_result {
                Ok(value) => sender.send(Ok(value)),
                Err(_) => sender.send(Err(TaskError::Panicked)),
            }
        })
        .expect("scheduler must accept scheduled token wrapper without metrics job");

    let result = handle
        .join()
        .expect("scheduled token wrapper without metrics handle must be attached")
        .expect("scheduled token wrapper without metrics handle must contain a value");

    verify_ready_value(result)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_scheduled_public_registry_token_wrapper_without_metrics(
    scheduler: &ThreadScheduler,
    registry: &TaskRegistry,
) -> usize {
    let (id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            black_box(execution_time);
            let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));
            match task_result {
                Ok(value) => sender.send(Ok(value)),
                Err(_) => sender.send(Err(TaskError::Panicked)),
            }
        })
        .expect("scheduler must accept scheduled registry-token wrapper without metrics job");

    let result = handle
        .join()
        .expect("scheduled registry-token wrapper without metrics handle must be attached")
        .expect("scheduled registry-token wrapper without metrics handle must contain a value");

    verify_ready_value(result)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_scheduled_public_token_wrapper_without_catch(
    scheduler: &ThreadScheduler,
    registry: &TaskRegistry,
    metrics: &Arc<ExecutorMetrics>,
) -> usize {
    let (id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();
    let worker_metrics = Arc::clone(metrics);

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            sender.send(Ok(black_box(READY_VALUE)));
            worker_metrics.record_task_completed(execution_time);
        })
        .expect("scheduler must accept scheduled token wrapper without catch diagnostic job");

    let result = handle
        .join()
        .expect("scheduled token wrapper without catch handle must be attached")
        .expect("scheduled token wrapper without catch handle must contain a value");

    verify_ready_value(result)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_scheduled_public_token_wrapper_atomic_result(
    scheduler: &ThreadScheduler,
    registry: &TaskRegistry,
    metrics: &Arc<ExecutorMetrics>,
) -> usize {
    let (_id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();
    metrics.record_task_spawned();
    let worker_metrics = Arc::clone(metrics);
    let result = Arc::new(AtomicUsize::new(0));
    let worker_result = Arc::clone(&result);

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));
            match task_result {
                Ok(value) => {
                    worker_result.store(value, Ordering::Release);
                    worker_metrics.record_task_completed(execution_time);
                }
                Err(_) => {
                    worker_result.store(usize::MAX, Ordering::Release);
                    worker_metrics.record_task_failed();
                }
            }
        })
        .expect("scheduler must accept scheduled token wrapper atomic-result job");
    scheduler
        .join()
        .expect("scheduler must reach quiescence after atomic-result wrapper job");

    verify_ready_value(result.load(Ordering::Acquire))
}

#[cfg(feature = "registry-diagnostics")]
fn direct_scheduled_public_token_wrapper_without_lifecycle(
    scheduler: &ThreadScheduler,
    next_task_id: &AtomicU64,
) -> usize {
    let id = next_task_id.fetch_add(1, Ordering::Relaxed);
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));
            match task_result {
                Ok(value) => sender.send(Ok(value)),
                Err(_) => sender.send(Err(TaskError::Panicked)),
            }
        })
        .expect("scheduler must accept scheduled token wrapper without lifecycle job");

    let result = handle
        .join()
        .expect("scheduled token wrapper without lifecycle handle must be attached")
        .expect("scheduled token wrapper without lifecycle handle must contain a value");

    verify_ready_value(result)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_scheduled_public_token_wrapper_oversized_components(
    scheduler: &ThreadScheduler,
    registry: &TaskRegistry,
    metrics: &Arc<ExecutorMetrics>,
) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();
    let worker_metrics = Arc::clone(metrics);

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            let task_result = catch_unwind(AssertUnwindSafe(|| oversized_capture_sum(words)));
            match task_result {
                Ok(value) => {
                    sender.send(Ok(value));
                    worker_metrics.record_task_completed(execution_time);
                }
                Err(_) => {
                    sender.send(Err(TaskError::Panicked));
                    worker_metrics.record_task_failed();
                }
            }
        })
        .expect("scheduler must accept scheduled oversized token wrapper diagnostic job");

    let result = handle
        .join()
        .expect("scheduled oversized token wrapper handle must be attached")
        .expect("scheduled oversized token wrapper handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_scheduled_public_token_wrapper_oversized_storage_only(
    scheduler: &ThreadScheduler,
    registry: &TaskRegistry,
    metrics: &Arc<ExecutorMetrics>,
) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();
    let worker_metrics = Arc::clone(metrics);

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            black_box(words);
            let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));
            match task_result {
                Ok(value) => {
                    sender.send(Ok(value));
                    worker_metrics.record_task_completed(execution_time);
                }
                Err(_) => {
                    sender.send(Err(TaskError::Panicked));
                    worker_metrics.record_task_failed();
                }
            }
        })
        .expect("scheduler must accept scheduled oversized storage-only token wrapper job");

    let result = handle
        .join()
        .expect("scheduled oversized storage-only token wrapper handle must be attached")
        .expect("scheduled oversized storage-only token wrapper handle must contain a value");

    verify_ready_value(result)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_scheduled_public_token_wrapper_oversized_read_one_components(
    scheduler: &ThreadScheduler,
    registry: &TaskRegistry,
    metrics: &Arc<ExecutorMetrics>,
) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();
    let worker_metrics = Arc::clone(metrics);

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            let task_result = catch_unwind(AssertUnwindSafe(|| oversized_capture_read_one(words)));
            match task_result {
                Ok(value) => {
                    sender.send(Ok(value));
                    worker_metrics.record_task_completed(execution_time);
                }
                Err(_) => {
                    sender.send(Err(TaskError::Panicked));
                    worker_metrics.record_task_failed();
                }
            }
        })
        .expect("scheduler must accept scheduled oversized read-one token wrapper diagnostic job");

    let result = handle
        .join()
        .expect("scheduled oversized read-one token wrapper handle must be attached")
        .expect("scheduled oversized read-one token wrapper handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_scheduled_public_token_wrapper_oversized_without_metrics(
    scheduler: &ThreadScheduler,
    registry: &TaskRegistry,
) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            black_box(execution_time);
            let task_result = catch_unwind(AssertUnwindSafe(|| oversized_capture_sum(words)));
            match task_result {
                Ok(value) => sender.send(Ok(value)),
                Err(_) => sender.send(Err(TaskError::Panicked)),
            }
        })
        .expect("scheduler must accept scheduled oversized token wrapper without metrics job");

    let result = handle
        .join()
        .expect("scheduled oversized token wrapper without metrics handle must be attached")
        .expect("scheduled oversized token wrapper without metrics handle must contain a value");

    verify_oversized_captured_ready_value(result)
}
