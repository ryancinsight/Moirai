#[test]
fn async_public_handle_path_uses_inline_future_state() {
    let source = read_benchmark("../moirai-executor/src/hybrid/mod.rs");

    for required in [
        "struct AsyncFutureState<S, F, L = OwnedStateLease>",
        "future: UnsafeCell<MaybeUninit<F>>",
        "lifecycle: UnsafeCell<AsyncLifecycle<L>>",
        "result_sender: UnsafeCell<Option<TaskResultSender<F::Output>>>",
        "future_present: UnsafeCell<bool>",
        "L: StateLease",
        "Pin::new_unchecked",
        "ASYNC_NOTIFIED",
        "ASYNC_INLINE_REPOLL_LIMIT",
        "finish_pending_poll",
        "fn take_result_sender(&self) -> Option<TaskResultSender<F::Output>>",
        "fn schedule_wake(self: &Arc<Self>)",
        "with_registration_context(self.registration(), || self.schedule_wake())",
        "impl<S, F, L> Wake for AsyncFutureState<S, F, L>",
        "self.register_scheduled_task(Priority::Normal)?",
    ] {
        assert!(
            source.contains(required),
            "async public-handle state must retain {required}"
        );
    }

    for prohibited in [
        "Box::pin",
        "Pin<Box",
        "dyn Future<Output",
        "struct AsyncFutureWaker",
        "result_sender: Mutex<Option<TaskResultSender",
        "future_present: AtomicBool",
        "fn future_is_present(&self) -> bool",
        "if !self.future_is_present()",
    ] {
        assert!(
            !source.contains(prohibited),
            "async public-handle state must not reintroduce {prohibited}"
        );
    }

    let async_state_start = source
        .find("pub(crate) struct AsyncFutureState<S, F, L = OwnedStateLease>")
        .expect("async future state must remain present");
    let async_state = &source[async_state_start..];
    let async_state_end = async_state
        .find("\n}\n\n// Safety: `state` serializes")
        .expect("async future state must retain its safety contract");
    let async_state = &async_state[..async_state_end];
    let lifecycle_field = async_state
        .find("lifecycle: UnsafeCell<AsyncLifecycle<L>>")
        .expect("async lifecycle field must remain present");
    let scheduler_field = async_state
        .find("scheduler: S,")
        .expect("async scheduler field must remain present");
    assert!(
        lifecycle_field < scheduler_field,
        "async lifecycle must drop before its scheduler-owned storage"
    );
}

#[test]
fn public_handle_metrics_reuse_lifecycle_duration() {
    let source = read_benchmark("../moirai-executor/src/hybrid/mod.rs");

    for required in [
        "self.scheduler.schedule::<BlockingTask, _>(\n                Priority::Normal,\n                None,\n                move |worker_id|",
        "running.complete()",
        "TaskEnd::Completed(result)",
        "fn send_task_result<T>",
        "record_task_completed(completion.duration())",
        "TaskEnd::Cancelled",
    ] {
        assert!(
            source.contains(required),
            "public handle completion must retain lifecycle duration reuse through {required}"
        );
    }

    for prohibited in ["let started_at = Instant::now()", "started_at.elapsed()"] {
        assert!(
            !source.contains(prohibited),
            "public handle completion must not reintroduce duplicate timing through {prohibited}"
        );
    }
}

#[test]
fn public_handle_paths_retain_panic_containment() {
    let source = read_benchmark("../moirai-executor/src/hybrid/mod.rs");

    for required in [
        "let result = catch_unwind(AssertUnwindSafe(func));",
        "self.spawn_result::<SyncTask, _>(priority, locality_hint, move || task.execute())",
        "TaskEnd::Completed(result)",
        "TaskEnd::Completed(Err(payload))",
        "Err(TaskError::Panicked), Some(payload)",
        "drop_after_settlement(payload)",
        "resume_callback(callback)",
        "fn spawn_blocking_reports_panicked_result()",
        "assert_eq!(handle.join(), Some(Err(moirai_core::TaskError::Panicked)))",
    ] {
        assert!(
            source.contains(required),
            "public handle panic containment must retain {required}"
        );
    }
}
