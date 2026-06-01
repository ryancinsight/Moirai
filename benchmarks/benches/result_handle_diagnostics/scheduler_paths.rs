fn direct_scheduler_submit_join(scheduler: &ThreadScheduler) -> usize {
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, |_| {
            black_box(READY_VALUE);
        })
        .expect("scheduler must accept diagnostic job");
    scheduler
        .join()
        .expect("scheduler must drain diagnostic job");

    verify_ready_value(READY_VALUE)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_select_worker_serial(scheduler: &ThreadScheduler) -> usize {
    scheduler.diagnostic_select_worker_for_state::<BlockingTask>(
        black_box(moirai_core::Priority::Normal),
        None,
        black_box(0),
        black_box(0),
    )
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_pending_counter_pair(scheduler: &ThreadScheduler) -> usize {
    scheduler.diagnostic_pending_counter_pair()
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_worker_unpark(scheduler: &ThreadScheduler) -> usize {
    scheduler.diagnostic_worker_unpark(black_box(BLOCKING_NORMAL_WORKER))
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_priority_queue_push_pop() -> usize {
    ThreadScheduler::<256, 256>::diagnostic_priority_queue_push_pop(black_box(
        moirai_core::Priority::Normal,
    ))
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_submission_queue_publication(scheduler: &ThreadScheduler) -> usize {
    let observed = scheduler.diagnostic_submission_queue_publication::<BlockingTask>(
        black_box(moirai_core::Priority::Normal),
        None,
    );
    assert_eq!(observed, BLOCKING_NORMAL_WORKER + 1);
    black_box(observed)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_worker_execute_ready_job(scheduler: &ThreadScheduler) -> usize {
    let observed = scheduler.diagnostic_worker_execute_ready_job(black_box(BLOCKING_NORMAL_WORKER));
    assert_eq!(observed, BLOCKING_NORMAL_WORKER);
    black_box(observed)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_worker_local_dequeue_execute(scheduler: &ThreadScheduler) -> usize {
    let observed =
        scheduler.diagnostic_worker_local_dequeue_execute(black_box(BLOCKING_NORMAL_WORKER));
    assert_eq!(observed, BLOCKING_NORMAL_WORKER + 1);
    black_box(observed)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_max_inline_job_construct_drop() -> usize {
    assert_eq!(
        ThreadScheduler::<256, 256>::diagnostic_max_inline_job_construct_drop(),
        MAX_INLINE_CAPTURE_WORDS
    );
    verify_ready_value(READY_VALUE)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_max_inline_job_construct_execute() -> usize {
    assert_eq!(
        ThreadScheduler::<256, 256>::diagnostic_max_inline_job_construct_execute(),
        1
    );
    verify_ready_value(READY_VALUE)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_oversized_job_construct_drop() -> usize {
    assert_eq!(
        ThreadScheduler::<256, 256>::diagnostic_oversized_job_construct_drop(),
        OVERSIZED_CAPTURE_WORDS
    );
    verify_ready_value(READY_VALUE)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_oversized_job_construct_execute() -> usize {
    assert_eq!(
        ThreadScheduler::<256, 256>::diagnostic_oversized_job_construct_execute(),
        1
    );
    verify_ready_value(READY_VALUE)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_max_inline_queue_push_pop_execute() -> usize {
    assert_eq!(
        ThreadScheduler::<256, 256>::diagnostic_max_inline_queue_push_pop_execute(),
        1
    );
    verify_ready_value(READY_VALUE)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_oversized_queue_push_pop_execute() -> usize {
    assert_eq!(
        ThreadScheduler::<256, 256>::diagnostic_oversized_queue_push_pop_execute(),
        1
    );
    verify_ready_value(READY_VALUE)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_worker_local_max_inline_dequeue_execute(
    scheduler: &ThreadScheduler,
) -> usize {
    let observed = scheduler.diagnostic_worker_local_max_inline_dequeue_execute(black_box(
        BLOCKING_NORMAL_WORKER,
    ));
    assert_eq!(observed, BLOCKING_NORMAL_WORKER + 1);
    black_box(observed)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_worker_local_oversized_dequeue_execute(scheduler: &ThreadScheduler) -> usize {
    let observed = scheduler.diagnostic_worker_local_oversized_dequeue_execute(black_box(
        BLOCKING_NORMAL_WORKER,
    ));
    assert_eq!(observed, BLOCKING_NORMAL_WORKER + 1);
    black_box(observed)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_join_fast_spin_quiescent(scheduler: &ThreadScheduler) -> usize {
    let observed = scheduler.diagnostic_join_fast_spin_quiescent();
    assert_eq!(observed, 1);
    black_box(observed)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_join_fast_spin_pending(scheduler: &ThreadScheduler) -> usize {
    let observed = scheduler.diagnostic_join_fast_spin_pending();
    assert_eq!(observed, SCHEDULER_JOIN_FAST_SPIN_ATTEMPTS);
    black_box(observed)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_empty_wake_decision(scheduler: &ThreadScheduler) -> usize {
    let observed =
        scheduler.diagnostic_wake_decision::<EmptyWakeDecision>(black_box(BLOCKING_NORMAL_WORKER));
    assert_eq!(observed, 1);
    black_box(observed)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_contended_wake_decision(scheduler: &ThreadScheduler) -> usize {
    let observed = scheduler
        .diagnostic_wake_decision::<ContendedWakeDecision>(black_box(BLOCKING_NORMAL_WORKER));
    assert_eq!(observed, 2);
    black_box(observed)
}

#[cfg(feature = "scheduler-diagnostics")]
fn direct_scheduler_saturated_wake_decision(scheduler: &ThreadScheduler) -> usize {
    let observed = scheduler
        .diagnostic_wake_decision::<SaturatedWakeDecision>(black_box(BLOCKING_NORMAL_WORKER));
    assert_eq!(observed, 0);
    black_box(observed)
}

fn direct_spawn_metrics_before_scheduler_submission(
    scheduler: &ThreadScheduler,
    metrics: &ExecutorMetrics,
) -> usize {
    let before = metrics.tasks_spawned.load(Ordering::Relaxed);
    metrics.record_task_spawned();

    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            sender.send(Ok(black_box(READY_VALUE)));
        })
        .expect("scheduler must accept metrics-before diagnostic job");
    let result = handle
        .join()
        .expect("metrics-before result handle must be attached")
        .expect("metrics-before result handle must contain a value");
    let after = metrics.tasks_spawned.load(Ordering::Relaxed);
    assert_eq!(after, before + 1);

    verify_ready_value(result)
}

fn direct_spawn_metrics_after_scheduler_submission(
    scheduler: &ThreadScheduler,
    metrics: &ExecutorMetrics,
) -> usize {
    let before = metrics.tasks_spawned.load(Ordering::Relaxed);
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            sender.send(Ok(black_box(READY_VALUE)));
        })
        .expect("scheduler must accept metrics-after diagnostic job");

    metrics.record_task_spawned();
    let result = handle
        .join()
        .expect("metrics-after result handle must be attached")
        .expect("metrics-after result handle must contain a value");
    let after = metrics.tasks_spawned.load(Ordering::Relaxed);
    assert_eq!(after, before + 1);

    verify_ready_value(result)
}

fn direct_scheduler_ready_atomic_join(scheduler: &ThreadScheduler) -> usize {
    let result = Arc::new(AtomicUsize::new(0));
    let worker_result = Arc::clone(&result);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            worker_result.store(black_box(READY_VALUE), Ordering::Release);
        })
        .expect("scheduler must accept atomic ready diagnostic job");
    scheduler
        .join()
        .expect("scheduler must drain atomic ready diagnostic job");

    verify_ready_value(result.load(Ordering::Acquire))
}

fn direct_scheduler_max_inline_atomic_join(scheduler: &ThreadScheduler) -> usize {
    let result = Arc::new(AtomicUsize::new(0));
    let worker_result = Arc::clone(&result);
    let words = [1usize; MAX_INLINE_CAPTURE_WORDS];
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            worker_result.store(max_inline_capture_sum(words), Ordering::Release);
        })
        .expect("scheduler must accept max-inline atomic diagnostic job");
    scheduler
        .join()
        .expect("scheduler must drain max-inline atomic diagnostic job");

    verify_max_inline_captured_value(result.load(Ordering::Acquire))
}

fn direct_scheduler_oversized_atomic_join(scheduler: &ThreadScheduler) -> usize {
    let result = Arc::new(AtomicUsize::new(0));
    let worker_result = Arc::clone(&result);
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            worker_result.store(oversized_capture_sum(words), Ordering::Release);
        })
        .expect("scheduler must accept oversized atomic diagnostic job");
    scheduler
        .join()
        .expect("scheduler must drain oversized atomic diagnostic job");

    verify_oversized_captured_ready_value(result.load(Ordering::Acquire))
}

fn direct_scheduler_worker_start_signal(scheduler: &ThreadScheduler) -> usize {
    let started = Arc::new(AtomicUsize::new(0));
    let release = Arc::new(AtomicUsize::new(0));
    let worker_started = Arc::clone(&started);
    let worker_release = Arc::clone(&release);

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            worker_started.store(black_box(READY_VALUE), Ordering::Release);
            while worker_release.load(Ordering::Acquire) == 0 {
                core::hint::spin_loop();
            }
        })
        .expect("scheduler must accept worker-start signal diagnostic job");

    while started.load(Ordering::Acquire) != READY_VALUE {
        core::hint::spin_loop();
    }

    release.store(1, Ordering::Release);
    scheduler
        .join()
        .expect("scheduler must drain worker-start signal diagnostic job");

    verify_ready_value(started.load(Ordering::Acquire))
}

fn direct_scheduler_worker_start_then_result_slot(scheduler: &ThreadScheduler) -> usize {
    let started = Arc::new(AtomicUsize::new(0));
    let release = Arc::new(AtomicUsize::new(0));
    let worker_started = Arc::clone(&started);
    let worker_release = Arc::clone(&release);
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);

    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            worker_started.store(black_box(READY_VALUE), Ordering::Release);
            while worker_release.load(Ordering::Acquire) == 0 {
                core::hint::spin_loop();
            }
            sender.send(Ok(black_box(READY_VALUE)));
        })
        .expect("scheduler must accept worker-start result-slot diagnostic job");

    while started.load(Ordering::Acquire) != READY_VALUE {
        core::hint::spin_loop();
    }

    release.store(1, Ordering::Release);
    let result = handle
        .join()
        .expect("worker-start result handle must be attached")
        .expect("worker-start result handle must contain a value");

    verify_ready_value(result)
}

fn direct_scheduler_result_slot(scheduler: &ThreadScheduler) -> usize {
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

    verify_ready_value(result)
}

fn direct_scheduler_boxed_ready_result_slot(scheduler: &ThreadScheduler) -> usize {
    let task = boxed_ready_value();
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            sender.send(Ok(task()));
        })
        .expect("scheduler must accept boxed ready result-slot diagnostic job");
    let result = handle
        .join()
        .expect("scheduled boxed ready result handle must be attached")
        .expect("scheduled boxed ready result handle must contain a value");

    verify_ready_value(result)
}

fn direct_scheduler_captured_result_slot(scheduler: &ThreadScheduler) -> usize {
    let words = [1usize; CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            sender.send(Ok(black_box(words.iter().copied().sum::<usize>())));
        })
        .expect("scheduler must accept captured result-slot diagnostic job");
    let result = handle
        .join()
        .expect("scheduled captured result handle must be attached")
        .expect("scheduled captured result handle must contain a value");

    verify_captured_ready_value(result)
}

fn direct_scheduler_max_inline_captured_result_slot(scheduler: &ThreadScheduler) -> usize {
    let words = [1usize; MAX_INLINE_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            sender.send(Ok(max_inline_capture_sum(words)));
        })
        .expect("scheduler must accept max-inline captured result-slot diagnostic job");
    let result = handle
        .join()
        .expect("scheduled max-inline captured result handle must be attached")
        .expect("scheduled max-inline captured result handle must contain a value");

    verify_max_inline_captured_value(result)
}

fn direct_scheduler_oversized_captured_result_slot(scheduler: &ThreadScheduler) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            sender.send(Ok(oversized_capture_sum(words)));
        })
        .expect("scheduler must accept oversized captured result-slot diagnostic job");
    let result = handle
        .join()
        .expect("scheduled oversized captured result handle must be attached")
        .expect("scheduled oversized captured result handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn direct_scheduler_oversized_capture_read_one_result_slot(scheduler: &ThreadScheduler) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            sender.send(Ok(oversized_capture_read_one(words)));
        })
        .expect("scheduler must accept oversized read-one result-slot diagnostic job");
    let result = handle
        .join()
        .expect("scheduled oversized read-one result handle must be attached")
        .expect("scheduled oversized read-one result handle must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn direct_scheduler_oversized_result_slot_with_quiescent_barrier(
    scheduler: &ThreadScheduler,
) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            sender.send(Ok(oversized_capture_sum(words)));
        })
        .expect("scheduler must accept oversized quiescent result-slot diagnostic job");
    let result = handle
        .join()
        .expect("scheduled oversized quiescent result handle must be attached")
        .expect("scheduled oversized quiescent result handle must contain a value");
    scheduler
        .join()
        .expect("scheduler must reach quiescence after oversized result-slot join");

    verify_oversized_captured_ready_value(result)
}

fn direct_scheduler_result_slot_with_metrics_tail(
    scheduler: &ThreadScheduler,
    metrics: &Arc<ExecutorMetrics>,
) -> usize {
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    let worker_metrics = Arc::clone(metrics);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            sender.send(Ok(black_box(READY_VALUE)));
            worker_metrics.record_task_completed(Duration::from_nanos(READY_VALUE as u64));
        })
        .expect("scheduler must accept metrics-tail result-slot diagnostic job");
    let result = handle
        .join()
        .expect("scheduled metrics-tail result handle must be attached")
        .expect("scheduled metrics-tail result handle must contain a value");

    verify_ready_value(result)
}

fn direct_scheduler_oversized_result_slot_with_metrics_tail(
    scheduler: &ThreadScheduler,
    metrics: &Arc<ExecutorMetrics>,
) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    let worker_metrics = Arc::clone(metrics);
    scheduler
        .schedule::<BlockingTask, _>(moirai_core::Priority::Normal, None, move |_| {
            sender.send(Ok(oversized_capture_sum(words)));
            worker_metrics
                .record_task_completed(Duration::from_nanos(OVERSIZED_CAPTURED_READY_VALUE as u64));
        })
        .expect("scheduler must accept oversized metrics-tail result-slot diagnostic job");
    let result = handle
        .join()
        .expect("scheduled oversized metrics-tail result handle must be attached")
        .expect("scheduled oversized metrics-tail result handle must contain a value");

    verify_oversized_captured_ready_value(result)
}
