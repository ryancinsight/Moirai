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
