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
    metrics: &ExecutorMetrics,
) -> usize {
    let (id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();

    let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));

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
    metrics: &ExecutorMetrics,
) -> usize {
    let (id, execution_time) = registry.diagnostic_register_next_and_complete_with_token_id();
    let (handle, sender) = TaskHandle::new_pending(TaskId(id));
    metrics.record_task_spawned();

    let task_result = catch_unwind(AssertUnwindSafe(|| black_box(READY_VALUE)));

    match task_result {
        Ok(value) => {
            sender.send(Ok(value));
            metrics.record_task_completed(execution_time);
        }
        Err(_) => {
            sender.send(Err(TaskError::Panicked));
            black_box(execution_time);
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
