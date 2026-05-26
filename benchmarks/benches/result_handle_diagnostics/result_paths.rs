fn direct_ready_result_slot() -> usize {
    let handle = TaskHandle::ready(TASK_ID, Ok(black_box(READY_VALUE)));
    let result = handle
        .join()
        .expect("direct ready handle must be attached")
        .expect("direct ready handle must contain a value");

    verify_ready_value(result)
}

fn direct_send_then_join_result_slot() -> usize {
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    sender.send(Ok(black_box(READY_VALUE)));
    let result = handle
        .join()
        .expect("direct sent handle must be attached")
        .expect("direct sent handle must contain a value");

    verify_ready_value(result)
}

fn direct_cross_thread_result_slot() -> usize {
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    let worker = thread::spawn(move || {
        sender.send(Ok(black_box(READY_VALUE)));
    });
    let result = handle
        .join()
        .expect("cross-thread handle must be attached")
        .expect("cross-thread handle must contain a value");
    worker.join().expect("cross-thread sender must not panic");

    verify_ready_value(result)
}

#[cfg(feature = "result-diagnostics")]
fn direct_result_slot_ready_take() -> usize {
    verify_ready_value(moirai_core::task::diagnostic_result_slot_ready_take())
}

#[cfg(feature = "result-diagnostics")]
fn direct_result_slot_spin_miss() -> usize {
    assert_eq!(
        moirai_core::task::diagnostic_result_slot_spin_miss(),
        moirai_core::constants::MAX_SPIN_ATTEMPTS
    );
    verify_ready_value(READY_VALUE)
}

#[cfg(feature = "result-diagnostics")]
fn direct_result_slot_register_waiter() -> usize {
    assert_eq!(
        moirai_core::task::diagnostic_result_slot_register_waiter(),
        1
    );
    verify_ready_value(READY_VALUE)
}

#[cfg(feature = "result-diagnostics")]
fn direct_result_slot_complete_waiting() -> usize {
    verify_ready_value(moirai_core::task::diagnostic_result_slot_complete_waiting())
}

fn moirai_spawn_join_ready(moirai: &Moirai) -> usize {
    let handle = moirai.spawn_fn(|| black_box(READY_VALUE));
    let result = handle
        .join()
        .expect("Moirai handle must be attached")
        .expect("Moirai ready task must contain a value");

    verify_ready_value(result)
}

fn moirai_spawn_join_captured_ready(moirai: &Moirai) -> usize {
    let words = [1usize; CAPTURE_WORDS];
    let handle = moirai.spawn_fn(move || black_box(words.iter().copied().sum::<usize>()));
    let result = handle
        .join()
        .expect("Moirai captured handle must be attached")
        .expect("Moirai captured task must contain a value");

    verify_captured_ready_value(result)
}

fn moirai_spawn_join_oversized_captured_ready(moirai: &Moirai) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let handle = moirai.spawn_fn(move || oversized_capture_sum(words));
    let result = handle
        .join()
        .expect("Moirai oversized captured handle must be attached")
        .expect("Moirai oversized captured task must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn moirai_spawn_join_oversized_capture_read_one(moirai: &Moirai) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let handle = moirai.spawn_fn(move || oversized_capture_read_one(words));
    let result = handle
        .join()
        .expect("Moirai oversized read-one handle must be attached")
        .expect("Moirai oversized read-one task must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn moirai_spawn_blocking_ready(moirai: &Moirai) -> usize {
    let handle = moirai.spawn_blocking(|| black_box(READY_VALUE));
    let result = handle
        .join()
        .expect("Moirai blocking handle must be attached")
        .expect("Moirai blocking ready task must contain a value");

    verify_ready_value(result)
}

fn moirai_spawn_blocking_oversized_captured_ready(moirai: &Moirai) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let handle = moirai.spawn_blocking(move || oversized_capture_sum(words));
    let result = handle
        .join()
        .expect("Moirai blocking oversized handle must be attached")
        .expect("Moirai blocking oversized task must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn arc_hybrid_spawn_blocking_ready(executor: &Arc<HybridExecutor>) -> usize {
    let handle = executor
        .spawn_blocking(|| black_box(READY_VALUE))
        .expect("Arc hybrid executor must spawn ready blocking task");
    let result = handle
        .join()
        .expect("Arc hybrid handle must be attached")
        .expect("Arc hybrid ready task must contain a value");

    verify_ready_value(result)
}

fn arc_hybrid_spawn_blocking_oversized_captured_ready(
    executor: &Arc<HybridExecutor>,
) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let handle = executor
        .spawn_blocking(move || oversized_capture_sum(words))
        .expect("Arc hybrid executor must spawn oversized blocking task");
    let result = handle
        .join()
        .expect("Arc hybrid oversized handle must be attached")
        .expect("Arc hybrid oversized task must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn moirai_spawn_async_ready(moirai: &Moirai) -> usize {
    let handle = moirai.spawn_async(async { black_box(READY_VALUE) });
    let result = handle
        .join()
        .expect("Moirai async ready handle must be attached")
        .expect("Moirai async ready task must contain a value");

    verify_ready_value(result)
}

fn moirai_spawn_async_wake_once(moirai: &Moirai) -> usize {
    let handle = moirai.spawn_async(WakeOnce::default());
    let result = handle
        .join()
        .expect("Moirai wake-once handle must be attached")
        .expect("Moirai wake-once task must contain a value");

    verify_ready_value(result)
}

fn hybrid_spawn_blocking_ready(executor: &HybridExecutor) -> usize {
    let handle = executor
        .spawn_blocking(|| black_box(READY_VALUE))
        .expect("hybrid executor must spawn ready blocking task");
    let result = handle
        .join()
        .expect("hybrid handle must be attached")
        .expect("hybrid ready task must contain a value");

    verify_ready_value(result)
}

fn hybrid_spawn_blocking_captured_ready(executor: &HybridExecutor) -> usize {
    let words = [1usize; CAPTURE_WORDS];
    let handle = executor
        .spawn_blocking(move || black_box(words.iter().copied().sum::<usize>()))
        .expect("hybrid executor must spawn captured blocking task");
    let result = handle
        .join()
        .expect("hybrid captured handle must be attached")
        .expect("hybrid captured task must contain a value");

    verify_captured_ready_value(result)
}

fn hybrid_spawn_blocking_oversized_captured_ready(executor: &HybridExecutor) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let handle = executor
        .spawn_blocking(move || oversized_capture_sum(words))
        .expect("hybrid executor must spawn oversized captured blocking task");
    let result = handle
        .join()
        .expect("hybrid oversized captured handle must be attached")
        .expect("hybrid oversized captured task must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn hybrid_spawn_blocking_oversized_capture_read_one(executor: &HybridExecutor) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let handle = executor
        .spawn_blocking(move || oversized_capture_read_one(words))
        .expect("hybrid executor must spawn oversized read-one blocking task");
    let result = handle
        .join()
        .expect("hybrid oversized read-one handle must be attached")
        .expect("hybrid oversized read-one task must contain a value");

    verify_oversized_captured_ready_value(result)
}

fn moirai_spawn_join_ready_with_quiescent_barrier(moirai: &Moirai) -> usize {
    let handle = moirai.spawn_fn(|| black_box(READY_VALUE));
    let result = handle
        .join()
        .expect("Moirai handle must be attached")
        .expect("Moirai ready task must contain a value");
    moirai
        .join()
        .expect("Moirai runtime must reach quiescence after handle join");

    verify_ready_value(result)
}
