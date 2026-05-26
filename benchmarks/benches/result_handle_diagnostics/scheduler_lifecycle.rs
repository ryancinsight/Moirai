fn direct_scheduler_lifecycle_before_send_result_slot(scheduler: &ThreadScheduler) -> usize {
    let lifecycle = Arc::new(DiagnosticLifecycle::new());
    let worker_lifecycle = Arc::clone(&lifecycle);
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let started_after_ns = worker_lifecycle.start(worker_id);
            black_box(worker_lifecycle.complete_since(started_after_ns));
            sender.send(Ok(black_box(READY_VALUE)));
        })
        .expect("scheduler must accept lifecycle-before-send diagnostic job");
    let result = handle
        .join()
        .expect("scheduled lifecycle-before-send result handle must be attached")
        .expect("scheduled lifecycle-before-send result handle must contain a value");

    verify_ready_value(result)
}

fn direct_scheduler_lifecycle_elapsed_only_result_slot(scheduler: &ThreadScheduler) -> usize {
    let lifecycle = Arc::new(ElapsedOnlyLifecycle::new());
    let worker_lifecycle = Arc::clone(&lifecycle);
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let started_after_ns = worker_lifecycle.start(worker_id);
            black_box(worker_lifecycle.complete_since(started_after_ns));
            sender.send(Ok(black_box(READY_VALUE)));
        })
        .expect("scheduler must accept elapsed-only lifecycle diagnostic job");
    let result = handle
        .join()
        .expect("scheduled elapsed-only lifecycle result handle must be attached")
        .expect("scheduled elapsed-only lifecycle result handle must contain a value");

    verify_ready_value(result)
}

fn direct_scheduler_lifecycle_atomic_only_result_slot(scheduler: &ThreadScheduler) -> usize {
    let lifecycle = Arc::new(AtomicOnlyLifecycle::new());
    let worker_lifecycle = Arc::clone(&lifecycle);
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let started_after_ns = worker_lifecycle.start(worker_id);
            black_box(worker_lifecycle.complete_since(started_after_ns));
            sender.send(Ok(black_box(READY_VALUE)));
        })
        .expect("scheduler must accept atomic-only lifecycle diagnostic job");
    let result = handle
        .join()
        .expect("scheduled atomic-only lifecycle result handle must be attached")
        .expect("scheduled atomic-only lifecycle result handle must contain a value");

    verify_ready_value(result)
}

fn direct_scheduler_lifecycle_start_instant_result_slot(scheduler: &ThreadScheduler) -> usize {
    let lifecycle = Arc::new(StartInstantLifecycle::new());
    let worker_lifecycle = Arc::clone(&lifecycle);
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let running = worker_lifecycle.start(worker_id);
            black_box(running.complete());
            sender.send(Ok(black_box(READY_VALUE)));
        })
        .expect("scheduler must accept start-instant lifecycle diagnostic job");
    let result = handle
        .join()
        .expect("scheduled start-instant lifecycle result handle must be attached")
        .expect("scheduled start-instant lifecycle result handle must contain a value");

    verify_ready_value(result)
}

fn direct_scheduler_lifecycle_cached_clock_result_slot(
    scheduler: &ThreadScheduler,
    clock: Arc<CachedLifecycleClock>,
) -> usize {
    let lifecycle = Arc::new(CachedClockLifecycle::new(clock));
    let worker_lifecycle = Arc::clone(&lifecycle);
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let started_after_ns = worker_lifecycle.start(worker_id);
            black_box(worker_lifecycle.complete_since(started_after_ns));
            sender.send(Ok(black_box(READY_VALUE)));
        })
        .expect("scheduler must accept cached-clock lifecycle diagnostic job");
    let result = handle
        .join()
        .expect("scheduled cached-clock lifecycle result handle must be attached")
        .expect("scheduled cached-clock lifecycle result handle must contain a value");

    verify_ready_value(result)
}

#[cfg(windows)]
fn direct_scheduler_lifecycle_qpc_result_slot(scheduler: &ThreadScheduler) -> usize {
    let lifecycle = Arc::new(QpcLifecycle::new());
    let worker_lifecycle = Arc::clone(&lifecycle);
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let started_after_ns = worker_lifecycle.start(worker_id);
            black_box(worker_lifecycle.complete_since(started_after_ns));
            sender.send(Ok(black_box(READY_VALUE)));
        })
        .expect("scheduler must accept QPC lifecycle diagnostic job");
    let result = handle
        .join()
        .expect("scheduled QPC lifecycle result handle must be attached")
        .expect("scheduled QPC lifecycle result handle must contain a value");

    verify_ready_value(result)
}

fn direct_scheduler_lifecycle_duration_only_result_slot(scheduler: &ThreadScheduler) -> usize {
    let lifecycle = Arc::new(DurationOnlyLifecycle::new());
    let worker_lifecycle = Arc::clone(&lifecycle);
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            worker_lifecycle.start(worker_id);
            black_box(worker_lifecycle.complete());
            sender.send(Ok(black_box(READY_VALUE)));
        })
        .expect("scheduler must accept duration-only lifecycle diagnostic job");
    let result = handle
        .join()
        .expect("scheduled duration-only lifecycle result handle must be attached")
        .expect("scheduled duration-only lifecycle result handle must contain a value");

    verify_ready_value(result)
}

fn direct_scheduler_lifecycle_after_send_result_slot(scheduler: &ThreadScheduler) -> usize {
    let lifecycle = Arc::new(DiagnosticLifecycle::new());
    let worker_lifecycle = Arc::clone(&lifecycle);
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let started_after_ns = worker_lifecycle.start(worker_id);
            sender.send(Ok(black_box(READY_VALUE)));
            black_box(worker_lifecycle.complete_since(started_after_ns));
        })
        .expect("scheduler must accept lifecycle-after-send diagnostic job");
    let result = handle
        .join()
        .expect("scheduled lifecycle-after-send result handle must be attached")
        .expect("scheduled lifecycle-after-send result handle must contain a value");

    verify_ready_value(result)
}

fn direct_scheduler_oversized_lifecycle_before_send_result_slot(
    scheduler: &ThreadScheduler,
) -> usize {
    let lifecycle = Arc::new(DiagnosticLifecycle::new());
    let worker_lifecycle = Arc::clone(&lifecycle);
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let result = oversized_capture_sum(words);
            let started_after_ns = worker_lifecycle.start(worker_id);
            black_box(worker_lifecycle.complete_since(started_after_ns));
            sender.send(Ok(result));
        })
        .expect("scheduler must accept oversized lifecycle-before-send diagnostic job");
    let result = handle
        .join()
        .expect("scheduled oversized lifecycle-before-send result handle must be attached")
        .expect("scheduled oversized lifecycle-before-send result handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn direct_scheduler_oversized_lifecycle_elapsed_only_result_slot(
    scheduler: &ThreadScheduler,
) -> usize {
    let lifecycle = Arc::new(ElapsedOnlyLifecycle::new());
    let worker_lifecycle = Arc::clone(&lifecycle);
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let result = oversized_capture_sum(words);
            let started_after_ns = worker_lifecycle.start(worker_id);
            black_box(worker_lifecycle.complete_since(started_after_ns));
            sender.send(Ok(result));
        })
        .expect("scheduler must accept oversized elapsed-only lifecycle diagnostic job");
    let result = handle
        .join()
        .expect("scheduled oversized elapsed-only lifecycle result handle must be attached")
        .expect("scheduled oversized elapsed-only lifecycle result handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn direct_scheduler_oversized_lifecycle_atomic_only_result_slot(
    scheduler: &ThreadScheduler,
) -> usize {
    let lifecycle = Arc::new(AtomicOnlyLifecycle::new());
    let worker_lifecycle = Arc::clone(&lifecycle);
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let result = oversized_capture_sum(words);
            let started_after_ns = worker_lifecycle.start(worker_id);
            black_box(worker_lifecycle.complete_since(started_after_ns));
            sender.send(Ok(result));
        })
        .expect("scheduler must accept oversized atomic-only lifecycle diagnostic job");
    let result = handle
        .join()
        .expect("scheduled oversized atomic-only lifecycle result handle must be attached")
        .expect("scheduled oversized atomic-only lifecycle result handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn direct_scheduler_oversized_lifecycle_start_instant_result_slot(
    scheduler: &ThreadScheduler,
) -> usize {
    let lifecycle = Arc::new(StartInstantLifecycle::new());
    let worker_lifecycle = Arc::clone(&lifecycle);
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let result = oversized_capture_sum(words);
            let running = worker_lifecycle.start(worker_id);
            black_box(running.complete());
            sender.send(Ok(result));
        })
        .expect("scheduler must accept oversized start-instant lifecycle diagnostic job");
    let result = handle
        .join()
        .expect("scheduled oversized start-instant lifecycle result handle must be attached")
        .expect("scheduled oversized start-instant lifecycle result handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn direct_scheduler_oversized_lifecycle_cached_clock_result_slot(
    scheduler: &ThreadScheduler,
    clock: Arc<CachedLifecycleClock>,
) -> usize {
    let lifecycle = Arc::new(CachedClockLifecycle::new(clock));
    let worker_lifecycle = Arc::clone(&lifecycle);
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let result = oversized_capture_sum(words);
            let started_after_ns = worker_lifecycle.start(worker_id);
            black_box(worker_lifecycle.complete_since(started_after_ns));
            sender.send(Ok(result));
        })
        .expect("scheduler must accept oversized cached-clock lifecycle diagnostic job");
    let result = handle
        .join()
        .expect("scheduled oversized cached-clock lifecycle result handle must be attached")
        .expect("scheduled oversized cached-clock lifecycle result handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

#[cfg(windows)]
fn direct_scheduler_oversized_lifecycle_qpc_result_slot(scheduler: &ThreadScheduler) -> usize {
    let lifecycle = Arc::new(QpcLifecycle::new());
    let worker_lifecycle = Arc::clone(&lifecycle);
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let result = oversized_capture_sum(words);
            let started_after_ns = worker_lifecycle.start(worker_id);
            black_box(worker_lifecycle.complete_since(started_after_ns));
            sender.send(Ok(result));
        })
        .expect("scheduler must accept oversized QPC lifecycle diagnostic job");
    let result = handle
        .join()
        .expect("scheduled oversized QPC lifecycle result handle must be attached")
        .expect("scheduled oversized QPC lifecycle result handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn direct_scheduler_oversized_lifecycle_duration_only_result_slot(
    scheduler: &ThreadScheduler,
) -> usize {
    let lifecycle = Arc::new(DurationOnlyLifecycle::new());
    let worker_lifecycle = Arc::clone(&lifecycle);
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let result = oversized_capture_sum(words);
            worker_lifecycle.start(worker_id);
            black_box(worker_lifecycle.complete());
            sender.send(Ok(result));
        })
        .expect("scheduler must accept oversized duration-only lifecycle diagnostic job");
    let result = handle
        .join()
        .expect("scheduled oversized duration-only lifecycle result handle must be attached")
        .expect("scheduled oversized duration-only lifecycle result handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn direct_scheduler_oversized_lifecycle_after_send_result_slot(
    scheduler: &ThreadScheduler,
) -> usize {
    let lifecycle = Arc::new(DiagnosticLifecycle::new());
    let worker_lifecycle = Arc::clone(&lifecycle);
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |worker_id| {
            let started_after_ns = worker_lifecycle.start(worker_id);
            sender.send(Ok(oversized_capture_sum(words)));
            black_box(worker_lifecycle.complete_since(started_after_ns));
        })
        .expect("scheduler must accept oversized lifecycle-after-send diagnostic job");
    let result = handle
        .join()
        .expect("scheduled oversized lifecycle-after-send result handle must be attached")
        .expect("scheduled oversized lifecycle-after-send result handle must contain a value");

    verify_oversized_captured_ready_value(result)
}
