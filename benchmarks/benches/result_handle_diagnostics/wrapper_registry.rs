fn direct_scheduler_tail_after_send_result_slot(scheduler: &ThreadScheduler) -> usize {
    let tail_words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            sender.send(Ok(black_box(READY_VALUE)));
            black_box(oversized_capture_sum(tail_words));
        })
        .expect("scheduler must accept tail-after-send result-slot diagnostic job");
    let result = handle
        .join()
        .expect("scheduled tail-after-send result handle must be attached")
        .expect("scheduled tail-after-send result handle must contain a value");

    verify_ready_value(result)
}

fn direct_scheduler_tail_after_send_with_quiescent_barrier(scheduler: &ThreadScheduler) -> usize {
    let tail_words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            sender.send(Ok(black_box(READY_VALUE)));
            black_box(oversized_capture_sum(tail_words));
        })
        .expect("scheduler must accept quiescent tail-after-send result-slot diagnostic job");
    let result = handle
        .join()
        .expect("scheduled quiescent tail-after-send result handle must be attached")
        .expect("scheduled quiescent tail-after-send result handle must contain a value");
    scheduler
        .join()
        .expect("scheduler must reach quiescence after tail-after-send result-slot join");

    verify_ready_value(result)
}

fn direct_scheduler_pinned_oversized_captured_result_slot(scheduler: &ThreadScheduler) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, Some(0), move |_| {
            sender.send(Ok(oversized_capture_sum(words)));
        })
        .expect("scheduler must accept pinned oversized captured result-slot diagnostic job");
    let result = handle
        .join()
        .expect("scheduled pinned oversized captured result handle must be attached")
        .expect("scheduled pinned oversized captured result handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn direct_scheduler_pinned_oversized_capture_read_one_result_slot(
    scheduler: &ThreadScheduler,
) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, Some(0), move |_| {
            sender.send(Ok(oversized_capture_read_one(words)));
        })
        .expect("scheduler must accept pinned oversized read-one result-slot diagnostic job");
    let result = handle
        .join()
        .expect("scheduled pinned oversized read-one result handle must be attached")
        .expect("scheduled pinned oversized read-one result handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn direct_scheduler_affinity_oversized_captured_result_slot(scheduler: &ThreadScheduler) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(
            moirai_core::Priority::Normal,
            Some(BLOCKING_NORMAL_WORKER),
            move |_| {
                sender.send(Ok(oversized_capture_sum(words)));
            },
        )
        .expect("scheduler must accept affinity oversized captured result-slot diagnostic job");
    let result = handle
        .join()
        .expect("scheduled affinity oversized captured result handle must be attached")
        .expect("scheduled affinity oversized captured result handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn direct_scheduler_result_slot_with_quiescent_barrier(scheduler: &ThreadScheduler) -> usize {
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            sender.send(Ok(black_box(READY_VALUE)));
        })
        .expect("scheduler must accept result-slot diagnostic job");
    let result = handle
        .join()
        .expect("scheduled result handle must be attached")
        .expect("scheduled result handle must contain a value");
    scheduler
        .join()
        .expect("scheduler must reach quiescence after result-slot join");

    verify_ready_value(result)
}

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

fn direct_public_wrapper_without_metrics(registry: &mut TaskRegistry) -> usize {
    let id = registry.register_task();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));

    registry.mark_started(id, 0);
    let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));
    registry.mark_completed(id);

    match task_result {
        Ok(value) => sender.send(Ok(value)),
        Err(_) => sender.send(Err(TaskError::Panicked)),
    }

    let result = handle
        .join()
        .expect("direct public wrapper without metrics handle must be attached")
        .expect("direct public wrapper without metrics handle must contain a value");

    verify_ready_value(result)
}

fn direct_public_wrapper_components(
    registry: &mut TaskRegistry,
    metrics: &ExecutorMetrics,
) -> usize {
    let id = registry.register_task();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();

    registry.mark_started(id, 0);
    let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));
    registry.mark_completed(id);
    let execution_time = registry
        .get_metadata(id)
        .and_then(|metadata| metadata.execution_duration())
        .expect("direct public wrapper lifecycle must record execution duration");

    match task_result {
        Ok(value) => {
            sender.send(Ok(value));
            metrics.record_task_completed(execution_time);
        }
        Err(_) => {
            sender.send(Err(TaskError::Panicked));
            metrics.record_task_failed();
        }
    }

    let result = handle
        .join()
        .expect("direct public wrapper handle must be attached")
        .expect("direct public wrapper handle must contain a value");

    verify_ready_value(result)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_public_token_wrapper_components(
    registry: &mut TaskRegistry,
    next_task_id: &AtomicU64,
    metrics: &ExecutorMetrics,
) -> usize {
    let id = next_task_id.fetch_add(1, Ordering::Relaxed);
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();

    let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));
    let execution_time = registry.diagnostic_restart_and_complete_with_token(id);

    match task_result {
        Ok(value) => {
            sender.send(Ok(value));
            metrics.record_task_completed(execution_time);
        }
        Err(_) => {
            sender.send(Err(TaskError::Panicked));
            metrics.record_task_failed();
        }
    }

    let result = handle
        .join()
        .expect("direct token wrapper handle must be attached")
        .expect("direct token wrapper handle must contain a value");

    verify_ready_value(result)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_public_token_wrapper_after_send_components(
    registry: &mut TaskRegistry,
    next_task_id: &AtomicU64,
    metrics: &ExecutorMetrics,
) -> usize {
    let id = next_task_id.fetch_add(1, Ordering::Relaxed);
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();

    let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));

    match task_result {
        Ok(value) => {
            sender.send(Ok(value));
            metrics.record_task_completed(registry.diagnostic_restart_and_complete_with_token(id));
        }
        Err(_) => {
            sender.send(Err(TaskError::Panicked));
            black_box(registry.diagnostic_restart_and_complete_with_token(id));
            metrics.record_task_failed();
        }
    }

    let result = handle
        .join()
        .expect("direct token after-send wrapper handle must be attached")
        .expect("direct token after-send wrapper handle must contain a value");

    verify_ready_value(result)
}

fn direct_public_wrapper_oversized_captured_components(
    registry: &mut TaskRegistry,
    metrics: &ExecutorMetrics,
) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let id = registry.register_task();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();

    registry.mark_started(id, 0);
    let task_result = catch_unwind(AssertUnwindSafe(|| oversized_capture_sum(words)));
    registry.mark_completed(id);
    let execution_time = registry
        .get_metadata(id)
        .and_then(|metadata| metadata.execution_duration())
        .expect("direct oversized wrapper lifecycle must record execution duration");

    match task_result {
        Ok(value) => {
            sender.send(Ok(value));
            metrics.record_task_completed(execution_time);
        }
        Err(_) => {
            sender.send(Err(TaskError::Panicked));
            metrics.record_task_failed();
        }
    }

    let result = handle
        .join()
        .expect("direct oversized wrapper handle must be attached")
        .expect("direct oversized wrapper handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn direct_public_wrapper_oversized_capture_read_one_components(
    registry: &mut TaskRegistry,
    metrics: &ExecutorMetrics,
) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let id = registry.register_task();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();

    registry.mark_started(id, 0);
    let task_result = catch_unwind(AssertUnwindSafe(|| oversized_capture_read_one(words)));
    registry.mark_completed(id);
    let execution_time = registry
        .get_metadata(id)
        .and_then(|metadata| metadata.execution_duration())
        .expect("direct oversized read-one wrapper lifecycle must record execution duration");

    match task_result {
        Ok(value) => {
            sender.send(Ok(value));
            metrics.record_task_completed(execution_time);
        }
        Err(_) => {
            sender.send(Err(TaskError::Panicked));
            metrics.record_task_failed();
        }
    }

    let result = handle
        .join()
        .expect("direct oversized read-one wrapper handle must be attached")
        .expect("direct oversized read-one wrapper handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_scheduled_public_token_wrapper_components(
    scheduler: &ThreadScheduler,
    registry: &mut TaskRegistry,
    next_task_id: &AtomicU64,
    metrics: &Arc<ExecutorMetrics>,
) -> usize {
    let id = next_task_id.fetch_add(1, Ordering::Relaxed);
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();
    let worker_metrics = Arc::clone(metrics);
    let execution_time = registry.diagnostic_restart_and_complete_with_token(id);

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
fn direct_scheduled_public_token_wrapper_oversized_components(
    scheduler: &ThreadScheduler,
    registry: &mut TaskRegistry,
    next_task_id: &AtomicU64,
    metrics: &Arc<ExecutorMetrics>,
) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let id = next_task_id.fetch_add(1, Ordering::Relaxed);
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();
    let worker_metrics = Arc::clone(metrics);
    let execution_time = registry.diagnostic_restart_and_complete_with_token(id);

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
fn direct_scheduled_public_token_wrapper_oversized_read_one_components(
    scheduler: &ThreadScheduler,
    registry: &mut TaskRegistry,
    next_task_id: &AtomicU64,
    metrics: &Arc<ExecutorMetrics>,
) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let id = next_task_id.fetch_add(1, Ordering::Relaxed);
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();
    let worker_metrics = Arc::clone(metrics);
    let execution_time = registry.diagnostic_restart_and_complete_with_token(id);

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
    registry: &mut TaskRegistry,
    next_task_id: &AtomicU64,
) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let id = next_task_id.fetch_add(1, Ordering::Relaxed);
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    let execution_time = registry.diagnostic_restart_and_complete_with_token(id);

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

fn direct_registry_lifecycle(registry: &mut TaskRegistry) -> usize {
    let id = registry.register_task();
    registry.mark_started(id, 0);
    registry.mark_completed(id);

    verify_ready_value(READY_VALUE)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_external_id_registry_register(
    registry: &mut TaskRegistry,
    next_task_id: &AtomicU64,
) -> usize {
    let id = next_task_id.fetch_add(1, Ordering::Relaxed);
    black_box(registry.diagnostic_register_external_task_with_id(id) as usize)
}

fn registry_mutex_lock_only(registry: &Mutex<TaskRegistry>) -> usize {
    let guard = registry
        .lock()
        .expect("diagnostic registry lock must not be poisoned");
    drop(guard);

    verify_ready_value(READY_VALUE)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_block_lookup(registry: &mut TaskRegistry) -> usize {
    black_box(registry.diagnostic_block_lookup() as usize)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_slot_initialize(registry: &mut TaskRegistry) -> usize {
    black_box(registry.diagnostic_slot_initialize() as usize)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_lifecycle_timestamp_publication() -> usize {
    let duration = TaskRegistry::diagnostic_lifecycle_timestamp_publication();
    black_box(duration.as_nanos() as usize)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_task_state_construct() -> usize {
    black_box(TaskRegistry::diagnostic_task_state_construct())
}

#[cfg(feature = "registry-diagnostics")]
fn registry_mark_started_existing_slot(registry: &TaskRegistry, task_id: u64) -> usize {
    black_box(registry.diagnostic_mark_started(task_id, 0) as usize)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_mark_completed_existing_slot(
    registry: &TaskRegistry,
    task_id: u64,
    started_after_ns: u64,
) -> usize {
    let duration = registry.diagnostic_mark_completed_since(task_id, started_after_ns);
    black_box(duration.as_nanos() as usize)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_elapsed_nanos_since_origin(origin: Instant) -> usize {
    black_box(elapsed_nanos_since(origin))
}

#[cfg(feature = "registry-diagnostics")]
fn registry_start_release_publication(
    started_after_ns: &AtomicUsize,
    worker_id: &AtomicUsize,
) -> usize {
    let offset = black_box(READY_VALUE);
    started_after_ns.store(offset, Ordering::Release);
    worker_id.store(black_box(BLOCKING_NORMAL_WORKER), Ordering::Release);
    black_box(offset)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_completion_release_publication(completed_after_ns: &AtomicUsize) -> usize {
    let offset = black_box(READY_VALUE);
    completed_after_ns.store(offset, Ordering::Release);
    black_box(offset)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_duration_offset_math() -> usize {
    let started_after_ns = black_box(READY_VALUE);
    let completed_after_ns = black_box(READY_VALUE + CAPTURED_READY_VALUE);
    debug_assert!(
        completed_after_ns >= started_after_ns,
        "diagnostic completion offset must not precede start offset"
    );
    black_box(completed_after_ns - started_after_ns)
}

fn mutex_registry_register(registry: &Mutex<TaskRegistry>) -> usize {
    let id = registry
        .lock()
        .expect("diagnostic registry lock must not be poisoned")
        .register_task();

    black_box(id as usize)
}
