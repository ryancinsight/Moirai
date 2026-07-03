#[test]
fn async_public_handle_path_uses_inline_future_state() {
    let source = read_benchmark("../moirai-executor/src/hybrid/mod.rs");

    for required in [
        "struct AsyncFutureState<S, F>",
        "future: UnsafeCell<MaybeUninit<F>>",
        "lifecycle: UnsafeCell<AsyncLifecycle>",
        "result_sender: UnsafeCell<Option<TaskResultSender<F::Output>>>",
        "future_present: UnsafeCell<bool>",
        "Pin::new_unchecked",
        "ASYNC_NOTIFIED",
        "ASYNC_INLINE_REPOLL_LIMIT",
        "finish_pending_poll",
        "fn take_result_sender(&self) -> Option<TaskResultSender<F::Output>>",
        "fn schedule_by_ref(self: &Arc<Self>) -> ExecutorResult<()>",
        "let _ = self.schedule_by_ref();",
        "impl<S, F> Wake for AsyncFutureState<S, F>",
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
}

#[test]
fn public_handle_metrics_reuse_lifecycle_duration() {
    let source = read_benchmark("../moirai-executor/src/hybrid/mod.rs");

    for required in [
        ".schedule::<BlockingTask, _>(Priority::Normal, None, move |worker_id|",
        "record_task_completed(running.complete())",
        "let execution_time = running.complete()",
        "send_task_result(result, result_sender, metrics.get(), execution_time)",
        "fn send_task_result<T>",
        "record_task_completed(execution_time)",
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
        "send_task_result(result, result_sender, metrics.get(), execution_time)",
        "sender.send(Err(TaskError::Panicked));",
        "fn spawn_blocking_reports_panicked_result()",
        "assert_eq!(handle.join(), Some(Err(moirai_core::TaskError::Panicked)))",
    ] {
        assert!(
            source.contains(required),
            "public handle panic containment must retain {required}"
        );
    }
}
