const DIAGNOSTIC_ASYNC_IDLE: u8 = 0;
const DIAGNOSTIC_ASYNC_QUEUED: u8 = 1;
const DIAGNOSTIC_ASYNC_POLLING: u8 = 2;
const DIAGNOSTIC_ASYNC_NOTIFIED: u8 = 3;
const DIAGNOSTIC_ASYNC_COMPLETED: u8 = 4;

struct DiagnosticAsyncWaker {
    state: AtomicU8,
}

impl DiagnosticAsyncWaker {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(DIAGNOSTIC_ASYNC_POLLING),
        }
    }

    fn notify_polling(&self) {
        let _ = self.state.compare_exchange(
            DIAGNOSTIC_ASYNC_POLLING,
            DIAGNOSTIC_ASYNC_NOTIFIED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

impl Wake for DiagnosticAsyncWaker {
    fn wake(self: Arc<Self>) {
        self.notify_polling();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.notify_polling();
    }
}

fn verify_async_state(actual: u8, expected: u8) -> usize {
    assert_eq!(actual, expected, "async diagnostic state transition mismatch");
    black_box(usize::from(actual))
}

fn direct_async_idle_to_queued_state_claim(state: &AtomicU8) -> usize {
    state.store(DIAGNOSTIC_ASYNC_IDLE, Ordering::Release);
    state
        .compare_exchange(
            DIAGNOSTIC_ASYNC_IDLE,
            DIAGNOSTIC_ASYNC_QUEUED,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .expect("idle async state must transition to queued");

    verify_async_state(state.load(Ordering::Acquire), DIAGNOSTIC_ASYNC_QUEUED)
}

fn direct_async_polling_to_notified_state_claim(state: &AtomicU8) -> usize {
    state.store(DIAGNOSTIC_ASYNC_POLLING, Ordering::Release);
    state
        .compare_exchange(
            DIAGNOSTIC_ASYNC_POLLING,
            DIAGNOSTIC_ASYNC_NOTIFIED,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .expect("polling async state must transition to notified");

    verify_async_state(state.load(Ordering::Acquire), DIAGNOSTIC_ASYNC_NOTIFIED)
}

fn direct_async_notified_to_polling_state_claim(state: &AtomicU8) -> usize {
    state.store(DIAGNOSTIC_ASYNC_NOTIFIED, Ordering::Release);
    state
        .compare_exchange(
            DIAGNOSTIC_ASYNC_NOTIFIED,
            DIAGNOSTIC_ASYNC_POLLING,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .expect("notified async state must transition to polling");

    verify_async_state(state.load(Ordering::Acquire), DIAGNOSTIC_ASYNC_POLLING)
}

fn direct_async_polling_to_idle_state_release(state: &AtomicU8) -> usize {
    state.store(DIAGNOSTIC_ASYNC_POLLING, Ordering::Release);
    state
        .compare_exchange(
            DIAGNOSTIC_ASYNC_POLLING,
            DIAGNOSTIC_ASYNC_IDLE,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .expect("pending async state must transition from polling to idle");

    verify_async_state(state.load(Ordering::Acquire), DIAGNOSTIC_ASYNC_IDLE)
}

fn direct_async_waker_from_arc(state: &Arc<DiagnosticAsyncWaker>) -> usize {
    let waker = Waker::from(Arc::clone(state));
    black_box(core::mem::size_of_val(&waker))
}

fn direct_async_wake_by_ref_polling_notification(
    state: &Arc<DiagnosticAsyncWaker>,
    waker: &Waker,
) -> usize {
    state
        .state
        .store(DIAGNOSTIC_ASYNC_POLLING, Ordering::Release);
    waker.wake_by_ref();

    verify_async_state(
        state.state.load(Ordering::Acquire),
        DIAGNOSTIC_ASYNC_NOTIFIED,
    )
}

fn direct_async_completed_state_store(state: &AtomicU8) -> usize {
    state.store(DIAGNOSTIC_ASYNC_POLLING, Ordering::Release);
    state.store(DIAGNOSTIC_ASYNC_COMPLETED, Ordering::Release);

    verify_async_state(state.load(Ordering::Acquire), DIAGNOSTIC_ASYNC_COMPLETED)
}

fn direct_async_future_present_drop_flag(future_present: &UnsafeCell<bool>) -> usize {
    // Safety: this diagnostic mirrors `AsyncFutureState`: one poll owner mutates
    // the flag while drop has exclusive access after the final `Arc` release.
    let was_present = unsafe {
        *future_present.get() = true;
        let was_present = *future_present.get();
        *future_present.get() = false;
        was_present
    };
    assert!(was_present, "async future-present flag must be set before drop");
    black_box(usize::from(was_present))
}

fn direct_async_lifecycle_complete(lifecycle: &DiagnosticLifecycle) -> usize {
    let started_after_ns = lifecycle.start(BLOCKING_NORMAL_WORKER);
    black_box(lifecycle.complete_since(started_after_ns))
}

fn direct_async_sender_cell_take_send_join() -> usize {
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    let sender_cell = UnsafeCell::new(Some(sender));

    // Safety: this diagnostic mirrors `AsyncFutureState`: a single completion
    // owner takes the one-shot sender exactly once.
    let sender = unsafe { (&mut *sender_cell.get()).take() }
        .expect("async sender-cell diagnostic must contain sender");
    sender.send(Ok(black_box(READY_VALUE)));

    let result = handle
        .join()
        .expect("async sender-cell diagnostic handle must be attached")
        .expect("async sender-cell diagnostic must contain a value");

    verify_ready_value(result)
}

fn direct_async_ready_completion_components(metrics: &ExecutorMetrics) -> usize {
    let state = AtomicU8::new(DIAGNOSTIC_ASYNC_POLLING);
    let future_present = UnsafeCell::new(true);
    let lifecycle = DiagnosticLifecycle::new();
    let started_after_ns = lifecycle.start(BLOCKING_NORMAL_WORKER);
    let (handle, sender) = TaskHandle::new_pending(TASK_ID);
    let sender_cell = UnsafeCell::new(Some(sender));

    // Safety: this diagnostic mirrors `AsyncFutureState::drop_future`, which is
    // called by the single poll owner.
    let was_present = unsafe {
        let present = &mut *future_present.get();
        let was_present = *present;
        *present = false;
        was_present
    };
    assert!(was_present, "async ready completion must drop one initialized future");
    state.store(DIAGNOSTIC_ASYNC_COMPLETED, Ordering::Release);
    let execution_time_ns = lifecycle.complete_since(started_after_ns);

    // Safety: this diagnostic has the same single completion-owner invariant as
    // `AsyncFutureState::take_result_sender`.
    let sender = unsafe { (&mut *sender_cell.get()).take() }
        .expect("async ready completion must contain sender");
    sender.send(Ok(black_box(READY_VALUE)));
    metrics.record_task_completed(Duration::from_nanos(execution_time_ns as u64));

    let result = handle
        .join()
        .expect("async ready completion handle must be attached")
        .expect("async ready completion must contain a value");

    black_box(verify_async_state(
        state.load(Ordering::Acquire),
        DIAGNOSTIC_ASYNC_COMPLETED,
    ));
    verify_ready_value(result)
}

fn benchmark_async_state_diagnostics(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    moirai: &Moirai,
) {
    group.bench_function("moirai_spawn_async_ready", |bench| {
        bench.iter(|| moirai_spawn_async_ready(moirai));
    });

    group.bench_function("moirai_spawn_async_wake_once", |bench| {
        bench.iter(|| moirai_spawn_async_wake_once(moirai));
    });

    group.bench_function("direct_async_idle_to_queued_state_claim", |bench| {
        let state = AtomicU8::new(DIAGNOSTIC_ASYNC_IDLE);
        bench.iter(|| direct_async_idle_to_queued_state_claim(&state));
    });

    group.bench_function("direct_async_polling_to_notified_state_claim", |bench| {
        let state = AtomicU8::new(DIAGNOSTIC_ASYNC_POLLING);
        bench.iter(|| direct_async_polling_to_notified_state_claim(&state));
    });

    group.bench_function("direct_async_notified_to_polling_state_claim", |bench| {
        let state = AtomicU8::new(DIAGNOSTIC_ASYNC_NOTIFIED);
        bench.iter(|| direct_async_notified_to_polling_state_claim(&state));
    });

    group.bench_function("direct_async_polling_to_idle_state_release", |bench| {
        let state = AtomicU8::new(DIAGNOSTIC_ASYNC_POLLING);
        bench.iter(|| direct_async_polling_to_idle_state_release(&state));
    });

    group.bench_function("direct_async_waker_from_arc", |bench| {
        let state = Arc::new(DiagnosticAsyncWaker::new());
        bench.iter(|| direct_async_waker_from_arc(&state));
    });

    group.bench_function("direct_async_wake_by_ref_polling_notification", |bench| {
        let state = Arc::new(DiagnosticAsyncWaker::new());
        let waker = Waker::from(Arc::clone(&state));
        bench.iter(|| direct_async_wake_by_ref_polling_notification(&state, &waker));
    });

    group.bench_function("direct_async_completed_state_store", |bench| {
        let state = AtomicU8::new(DIAGNOSTIC_ASYNC_POLLING);
        bench.iter(|| direct_async_completed_state_store(&state));
    });

    group.bench_function("direct_async_future_present_drop_flag", |bench| {
        let future_present = UnsafeCell::new(true);
        bench.iter(|| direct_async_future_present_drop_flag(&future_present));
    });

    group.bench_function("direct_async_lifecycle_complete", |bench| {
        let lifecycle = DiagnosticLifecycle::new();
        bench.iter(|| direct_async_lifecycle_complete(&lifecycle));
    });

    group.bench_function("direct_async_sender_cell_take_send_join", |bench| {
        bench.iter(direct_async_sender_cell_take_send_join);
    });

    group.bench_function("direct_async_ready_completion_components", |bench| {
        let metrics = ExecutorMetrics::new();
        bench.iter(|| direct_async_ready_completion_components(&metrics));
    });
}
