#[test]
fn timeout_combinator_stores_future_inline() {
    let source = read_benchmark("../moirai-async/src/timer.rs");

    for required in [
        "pub struct Timeout<F>",
        "future: F",
        "future,",
        "Pin::new_unchecked(&mut this.future)",
        "preserves support for",
    ] {
        assert!(
            source.contains(required),
            "timeout combinator must retain inline generic future storage through {required}"
        );
    }

    for prohibited in ["future: Pin<Box<F>>", "Box::pin(future)"] {
        assert!(
            !source.contains(prohibited),
            "timeout combinator must not reintroduce heap-pinned generic future storage through {prohibited}"
        );
    }
}

#[test]
fn async_executor_uses_monomorphized_erased_future_queue() {
    let source = read_benchmark("../moirai-async/src/executor.rs");

    for required in [
        "future: ErasedTaskFuture",
        "struct ErasedTaskFuture",
        "poll: unsafe fn(NonNull<()>, &mut Context<'_>) -> Poll<()>",
        "poll_erased_future::<F>",
        "drop_erased_future::<F>",
        "let ptr = Box::into_raw(Box::new(future)).cast::<()>();",
        "Pin::new_unchecked(&mut *ptr.cast::<F>().as_ptr())",
        "next_task_id: AtomicU64",
    ] {
        assert!(
            source.contains(required),
            "async executor must retain monomorphized erased future queue through {required}"
        );
    }

    for prohibited in [
        "Pin<Box<dyn Future<Output = ()>",
        "future: Box::pin(wrapped_future)",
        "AtomicU64::new(0).fetch_add",
    ] {
        assert!(
            !source.contains(prohibited),
            "async executor must not reintroduce {prohibited}"
        );
    }
}

#[test]
fn async_executor_handle_uses_inline_result_slot() {
    let source = format!(
        "{}\n{}",
        read_benchmark("../moirai-async/src/executor.rs"),
        read_benchmark("../moirai-async/src/executor/result_slot.rs")
    );

    for required in [
        "struct AsyncResultSlot<T>",
        "result: UnsafeCell<MaybeUninit<T>>",
        "state: AtomicU8",
        "waiter: UnsafeCell<MaybeUninit<Waker>>",
        "const ASYNC_RESULT_WAITING",
        "const ASYNC_RESULT_UPDATING_WAKER",
        "fn complete(&self, result: T)",
        "fn register_waker(&self, waker: &Waker)",
        "fn begin_completion(&self) -> Option<bool>",
        "test_ready_task_completion_wakes_registered_handle",
    ] {
        assert!(
            source.contains(required),
            "async handle result path must retain inline atomic result/waker slot through {required}"
        );
    }

    for prohibited in [
        "result_receiver: Arc<Mutex<Option<T>>>",
        "struct WakerRegistry",
        "HashMap<TaskId, Waker>",
        "register_waker(&self, task_id",
        "waker_registry",
    ] {
        assert!(
            !source.contains(prohibited),
            "async handle result path must not reintroduce mutex/hashmap waker storage through {prohibited}"
        );
    }
}

#[test]
fn parallel_join_uses_static_policy_and_scoped_scheduler() {
    let source = format!(
        "{}\n{}",
        read_benchmark("../moirai-parallel/src/ops.rs"),
        read_benchmark("../moirai-parallel/src/policy.rs")
    );

    for required in [
        "pub fn join_with<P, A, B, RA, RB>",
        "pub fn join<A, B, RA, RB>",
        "fn parallelize_pair() -> bool",
        "impl ExecutionPolicy for Parallel",
        "scope::<SyncTask",
        "scope.spawn(|_|",
        "scope.flush()?",
        "left_result = Some(left())",
        "right_result = Some(right())",
        "join_with::<crate::Adaptive",
    ] {
        assert!(
            source.contains(required),
            "parallel join must retain static scoped implementation through {required}"
        );
    }

    for prohibited in ["dyn ExecutionPolicy", "Box<dyn", "spawn_fn(", "TaskHandle<"] {
        assert!(
            !source.contains(prohibited),
            "parallel join must not route through dynamic dispatch or per-task handles via {prohibited}"
        );
    }
}

#[test]
fn parallel_join_benchmark_compares_value_checked_rayon_row() {
    let source = read_benchmark("../moirai-parallel/benches/par_benchmarks.rs");

    for required in [
        "fn bench_join(c: &mut Criterion)",
        "join_sum_pair",
        "closed_form_sum",
        "rayon::join",
        "join_with::<Parallel",
        "assert_eq!(sequential, expected)",
        "assert_eq!(rayon, expected)",
        "assert_eq!(moirai, expected)",
        "BenchmarkId::new(\"moirai\", n)",
    ] {
        assert!(
            source.contains(required),
            "parallel join benchmark must retain value-checked Rayon comparison through {required}"
        );
    }
}

#[test]
fn indexed_reduce_uses_worker_plus_caller_lane() {
    let source = read_benchmark("../moirai-executor/src/schedule/runtime/mod.rs");

    for required in [
        "count.min(worker_count.max(1).saturating_add(1))",
        "let max_chunks = count.min(worker_count.saturating_add(1));",
        "assert_eq!(indexed_reduce_chunk_count::<usize>(1024, 4), 5);",
    ] {
        assert!(
            source.contains(required),
            "indexed scheduling must retain the worker-plus-caller chunk cap through {required}"
        );
    }

    assert!(
        !source.contains("let max_chunks = count.min(worker_count);"),
        "indexed reduction must not cap chunks at worker-only lanes while the caller computes one chunk"
    );
}

#[test]
fn rayon_tokio_dependencies_stay_out_of_runtime_dependency_sections() {
    for relative in [
        "../moirai/Cargo.toml",
        "../moirai-async/Cargo.toml",
        "../moirai-core/Cargo.toml",
        "../moirai-executor/Cargo.toml",
        "../moirai-gpu/Cargo.toml",
        "../moirai-iter/Cargo.toml",
        "../moirai-metrics/Cargo.toml",
        "../moirai-pal/Cargo.toml",
        "../moirai-scheduler/Cargo.toml",
        "../moirai-sync/Cargo.toml",
        "../moirai-transport/Cargo.toml",
        "../moirai-utils/Cargo.toml",
    ] {
        let manifest = read_benchmark(relative);
        let dependencies = manifest_section(&manifest, "[dependencies]");
        for dependency in ["rayon", "tokio"] {
            assert!(
                !manifest_section_declares_dependency(dependencies, dependency),
                "{relative} must not use {dependency} as a runtime dependency"
            );
        }
    }

    let benchmark_manifest = read_benchmark("Cargo.toml");
    let benchmark_dependencies = manifest_section(&benchmark_manifest, "[dependencies]");
    for dependency in ["rayon", "tokio"] {
        assert!(
            manifest_section_declares_dependency(benchmark_dependencies, dependency),
            "benchmark crate must retain {dependency} as a comparison dependency"
        );
    }

    let public_manifest = read_benchmark("../moirai/Cargo.toml");
    let public_dev_dependencies = manifest_section(&public_manifest, "[dev-dependencies]");
    for dependency in ["rayon", "tokio"] {
        assert!(
            manifest_section_declares_dependency(public_dev_dependencies, dependency),
            "public crate examples/tests must retain {dependency} only as a dev comparison dependency"
        );
    }
}

#[test]
fn metrics_crate_uses_real_storage_and_export() {
    let lib = read_benchmark("../moirai-metrics/src/lib.rs");
    let collector = read_benchmark("../moirai-metrics/src/collector.rs");
    let counter = read_benchmark("../moirai-metrics/src/counter.rs");
    let gauge = read_benchmark("../moirai-metrics/src/gauge.rs");
    let histogram = read_benchmark("../moirai-metrics/src/histogram.rs");
    let exporter = read_benchmark("../moirai-metrics/src/exporter.rs");
    let tests = read_benchmark("../moirai-metrics/src/tests.rs");
    let benchmark = read_benchmark("benches/metrics_collector_comparison.rs");
    let combined = [
        lib.as_str(),
        collector.as_str(),
        counter.as_str(),
        gauge.as_str(),
        histogram.as_str(),
        exporter.as_str(),
        tests.as_str(),
        benchmark.as_str(),
    ]
    .join("\n");

    for required in [
        "mod collector;",
        "mod counter;",
        "mod exporter;",
        "mod gauge;",
        "mod histogram;",
        "mod snapshot;",
        "Arc<AtomicU64>",
        "Arc<AtomicI64>",
        "Arc<Mutex<HistogramState>>",
        "pub fn try_record(&self, value: f64) -> Result<(), HistogramError>",
        "pub fn collect(&self) -> MetricsSnapshot",
        "pub fn export(&self, snapshot: &MetricsSnapshot) -> String",
        "metrics_handles_share_named_storage",
        "prometheus_exporter_emits_deterministic_values",
        "metrics_collector_comparison",
        "counter_handle_add_get",
        "collector_snapshot_32_each",
        "prometheus_export_32_each",
    ] {
        assert!(
            combined.contains(required),
            "metrics crate must retain real storage/export marker {required}"
        );
    }

    for prohibited in [
        "Placeholder",
        "placeholder",
        "#[allow(dead_code)]",
        "pub fn export(&self, _snapshot: &MetricsSnapshot) -> String",
        "MetricsSnapshot::default()",
        "pub struct Metrics {\n}",
        "pub struct Histogram {\n}",
    ] {
        assert!(
            !combined.contains(prohibited),
            "metrics crate must not reintroduce placeholder marker {prohibited}"
        );
    }
}

#[test]
fn utility_simd_surface_uses_generic_scalar_contract() {
    let root = read_benchmark("../moirai-utils/src/simd/mod.rs");
    let scalar = read_benchmark("../moirai-utils/src/simd/scalar.rs");
    let arch = read_benchmark("../moirai-utils/src/simd/arch/mod.rs");
    let lib = read_benchmark("../moirai-utils/src/lib.rs");
    let simd_tests = read_benchmark("../moirai-utils/src/simd/tests.rs");
    let simd_benchmark = read_benchmark("benches/simd_benchmarks.rs");
    let moirai_benchmark = read_benchmark("benches/moirai_benchmarks.rs");
    let performance_benchmark = read_benchmark("benches/performance_benchmarks.rs");

    let implementation = format!("{root}\n{scalar}\n{arch}\n{lib}\n{simd_tests}");
    let benchmarks = format!("{simd_benchmark}\n{moirai_benchmark}\n{performance_benchmark}");

    for required in [
        "mod arch;",
        "mod scalar;",
        "pub use scalar::{SimdReal, SimdScalar};",
        "pub fn has_native_vector_path<T: SimdScalar>() -> bool",
        "pub fn add<T: SimdScalar>",
        "pub fn mul<T: SimdScalar>",
        "pub fn dot<T: SimdScalar>",
        "pub fn sum<T: SimdScalar>",
        "pub fn mean<T: SimdReal>",
        "pub fn variance<T: SimdReal>",
        "pub fn matrix_mul_square<T: SimdScalar, const N: usize>",
        "pub trait SimdScalar",
        "pub trait SimdReal",
        "sealed::Sealed",
        "impl SimdScalar for f32",
        "impl SimdScalar for f64",
        "impl SimdScalar for u64",
        "fn native_wide_vector_available() -> bool",
        "fn uses_native_wide_vector_path(len: usize) -> bool",
        "fn uses_native_vector_path(len: usize) -> bool",
        "fn native_vector_chunk_len(len: usize) -> Option<usize>",
        ".then_some((len / arch::LANES) * arch::LANES)",
        "fn matrix_mul_square<const N: usize>",
        "unaligned_lengths_preserve_values",
        "unaligned_vector_prefix_records_vector_dispatch_when_available",
        "wide_unaligned_vector_prefix_records_vector_dispatch_when_available",
    ] {
        assert!(
            implementation.contains(required),
            "utility SIMD implementation must retain generic scalar marker {required}"
        );
    }

    for required in [
        "add(black_box(&a), black_box(&b), black_box(&mut result))",
        "mul(black_box(&a), black_box(&b), black_box(&mut result))",
        "black_box(dot(black_box(&a), black_box(&b)))",
        "matrix_mul_square::<f32, 4>",
        "black_box(sum(black_box(&data)))",
        "black_box(mean(black_box(&data)))",
        "black_box(variance(black_box(&data)))",
        "vector_prefix_tail_addition",
        "vector_addition_wide",
        "wide_vectorized",
        "let expected: Vec<f64>",
        "generic_prefix_tail",
        "simd::add(&a, &b, &mut result)",
        "use moirai_utils::simd::{add, mul};",
    ] {
        assert!(
            benchmarks.contains(required),
            "SIMD benchmarks must consume generic utility API marker {required}"
        );
    }

    for prohibited in [
        "safe_vectorized_add_f32",
        "safe_vectorized_mul_f32",
        "safe_vectorized_dot_product_f32",
        "safe_vectorized_matrix_mul_4x4_f32",
        "safe_vectorized_sum_f32",
        "safe_vectorized_mean_f32",
        "safe_vectorized_variance_f32",
        "pub unsafe fn vectorized_add_f32",
        "pub unsafe fn vectorized_mul_f32",
        "pub unsafe fn vectorized_dot_product_f32",
        "pub unsafe fn vectorized_matrix_mul_4x4_f32",
        "SIMD_F32_WIDTH",
        "neon_vectorized_add_f32",
        "add_f64",
        "mul_f64",
        "dot_f64",
        "sum_f64",
        "squared_diff_sum_f64",
        "horizontal_sum_f64",
        "generate_test_data_f64",
        "bench_vector_addition_f64",
        "vector_addition_f64",
    ] {
        assert!(
            !implementation.contains(prohibited),
            "utility SIMD public surface must not retain type-suffixed marker {prohibited}"
        );
        assert!(
            !benchmarks.contains(prohibited),
            "SIMD benchmarks must not consume removed type-suffixed marker {prohibited}"
        );
    }
}

#[test]
fn async_executor_erases_futures_with_monomorphized_poll_drop() {
    let source = read_benchmark("../moirai-async/src/executor.rs");

    for required in [
        "struct ErasedTaskFuture",
        "poll: unsafe fn(NonNull<()>, &mut Context<'_>) -> Poll<()>",
        "drop: unsafe fn(NonNull<()>)",
        "fn new<F>(future: F) -> Self",
        "Box::into_raw(Box::new(future))",
        "poll_erased_future::<F>",
        "drop_erased_future::<F>",
        "Pin::new_unchecked",
    ] {
        assert!(
            source.contains(required),
            "async executor future erasure must retain monomorphized storage through {required}"
        );
    }

    for prohibited in ["future: Pin<Box", "dyn Future<Output"] {
        assert!(
            !source.contains(prohibited),
            "async executor must not reintroduce heap-pinned dynamic future storage through {prohibited}"
        );
    }
}

#[test]
fn rayon_tokio_gap_audit_tracks_executable_coverage() {
    let audit = read_benchmark("../docs/rayon_tokio_gap_audit.md");

    for required in [
        "No active comparison gap remains",
        "Moirai::spawn_fn",
        "Moirai::spawn_async",
        "Moirai::scope",
        "Moirai::map_reduce_indexed",
        "tokio::spawn",
        "rayon::scope",
        "into_par_iter().map(...).sum()",
        "public_result_handle_comparison",
        "thread_schedule_comparison",
        "mixed_unified_schedule",
        "real_application_mixed_workload",
        "moirai_real_app_pipeline",
        "tokio_rayon_real_app_pipeline",
        "standalone_deque_reclaim_policy",
        "Tokio plus Rayon",
        "QuiescentReclaim",
        "SharedEpochReclaim",
        "industry_comparison",
        "result_handle_diagnostics",
        "transport_archive_comparison",
        "benchmark_contracts",
        "WorkClass",
        "worker plus caller",
        "boxed inline trampoline",
        "ArchiveView",
        "ErasedTaskFuture",
        "ErasedThreadJob",
        "ChannelSplitter<T, I, C>",
        "channel_fusion_uses_typed_channels_without_placeholder_pipeline",
        "StreamingIter<T, F>",
        "streaming_iter_uses_monomorphized_producer_and_fifo_buffer",
        "iterator_base_does_not_expose_boxed_future_execution_trait",
        "Timeout<F>",
        "TimerWheel",
        "timer_wheel_cancellation_is_real_and_lazy",
        "TokioCompat<T>",
        "MoiraiCompat<T>",
        "async_io_compat_comparison",
        "async_tcp_readiness_comparison",
        "async_tcp_cancel_safety_comparison",
        "channel_matrix",
        "bounded_channel_matrix",
        "tokio_mpsc",
        "moirai_mpmc",
        "Box<dyn FnOnce>",
        "comparison-example dependencies",
    ] {
        assert!(
            audit.contains(required),
            "Rayon/Tokio gap audit must track {required}"
        );
    }

    for prohibited in [
        "[ ]".to_owned(),
        ["simu", "lated"].concat(),
        ["esti", "mated"].concat(),
    ] {
        assert!(
            !audit.contains(&prohibited),
            "Rayon/Tokio gap audit must not contain unresolved or non-executable marker {prohibited}"
        );
    }
}

#[test]
fn rayon_tokio_comparison_report_tracks_bounded_channel_coverage() {
    let report = read_benchmark("../docs/moirai_rayon_tokio_comparison.md");

    for required in [
        "Bounded channel transfer",
        "channel_matrix",
        "bounded_channel_matrix",
        "tokio_mpsc",
        "moirai_mpmc",
        "tokio::sync::mpsc::channel",
        "moirai_core::channel::mpmc",
    ] {
        assert!(
            report.contains(required),
            "Rayon/Tokio comparison report must track bounded channel marker {required}"
        );
    }
}

#[test]
fn public_facade_does_not_expose_placeholder_distributed_execution() {
    let source = read_benchmark("../moirai/src/lib.rs");
    let routed = read_benchmark("../moirai/src/routed.rs");
    let manifest = read_benchmark("../moirai/Cargo.toml");
    let source_all = format!("{source}\n{routed}\n{manifest}");

    for required in [
        "mod routed;",
        "pub use routed::{FixedRemoteTask, RoutedProcessTarget, RoutedServerTarget};",
        "moirai-transport/scheduler-routes",
        "pub struct FixedRemoteTask<C: RemoteCapability, Payload>",
        "pub struct RoutedServerTarget",
        "pub struct RoutedProcessTarget",
        "pub fn execute_routed_server_task<W, P, C, Payload>",
        "pub fn execute_routed_process_task<W, P, C, Payload>",
        "RemoteCapabilityToken<C>",
        "Payload: IntoRemoteOperation<C>",
        "build_remote_operation(task.payload, task.token)",
        "RoutedRemoteTaskClient::<P>::new",
        "RoutedProcessTaskClient::<P>::new",
        "public_facade_executes_fixed_capability_server_route",
        "public_facade_executes_fixed_capability_process_route",
        "distributed_feature_does_not_add_facade_remote_closure_execution",
    ] {
        assert!(
            source_all.contains(required),
            "public facade must document the distributed execution boundary through {required}"
        );
    }

    for prohibited in [
        "pub fn spawn_remote",
        "pub fn get_nodes",
        "pub fn register_node",
        "pub fn enable_distributed",
        "pub fn node_id",
        "remote-task-",
        "DISTRIBUTED:",
        "Simulate remote execution",
        "simulated locally",
        "worker-node-1",
        "gpu-cluster",
        "Box<dyn RemoteCapability",
        "dyn RemoteCapability",
        "dyn RemoteTask",
    ] {
        assert!(
            !source_all.contains(prohibited),
            "public facade must not reintroduce placeholder distributed execution marker {prohibited}"
        );
    }
}

#[test]
fn core_zero_copy_primitives_use_vertical_leaf_modules() {
    let communication = read_benchmark("../moirai-core/src/communication.rs");
    let module = read_benchmark("../moirai-core/src/communication/zero_copy/mod.rs");
    let error = read_benchmark("../moirai-core/src/communication/zero_copy/error.rs");
    let ring = read_benchmark("../moirai-core/src/communication/zero_copy/ring.rs");
    let channel = read_benchmark("../moirai-core/src/communication/zero_copy/channel.rs");
    let adaptive = read_benchmark("../moirai-core/src/communication/zero_copy/adaptive.rs");
    let router = read_benchmark("../moirai-core/src/communication/zero_copy/router.rs");

    for required in [
        "pub mod zero_copy;",
        "AdaptiveBatchChannel",
        "DomainId",
        "MemoryMappedRing",
        "ZeroCopyChannel",
        "ZeroCopyError",
        "ZeroCopyRouter",
    ] {
        assert!(
            communication.contains(required),
            "communication facade must retain zero-copy export {required}"
        );
    }

    for required in [
        "mod adaptive;",
        "mod channel;",
        "mod error;",
        "mod ring;",
        "mod router;",
        "pub use adaptive::{",
        "pub use channel::{ZeroCopyChannel, ZeroCopyReceiver, ZeroCopySender};",
        "pub use error::{ZeroCopyError, ZeroCopyResult};",
        "pub use ring::MemoryMappedRing;",
        "pub use router::{DomainId, ZeroCopyRouter};",
    ] {
        assert!(
            module.contains(required),
            "zero-copy module hierarchy must retain {required}"
        );
    }

    for (source, required) in [
        (error.as_str(), "pub enum ZeroCopyError"),
        (ring.as_str(), "pub struct MemoryMappedRing<T>"),
        (channel.as_str(), "pub struct ZeroCopyChannel<T>"),
        (adaptive.as_str(), "pub struct AdaptiveBatchChannel<T>"),
        (router.as_str(), "pub struct ZeroCopyRouter<T>"),
    ] {
        assert!(
            source.contains(required),
            "zero-copy leaf must retain {required}"
        );
    }
}

#[test]
fn scheduler_join_keeps_fast_quiescent_path_before_condvar_wait() {
    let source = read_benchmark("../moirai-executor/src/schedule/runtime/mod.rs");

    for required in [
        "const JOIN_FAST_SPIN_ATTEMPTS",
        "for _ in 0..JOIN_FAST_SPIN_ATTEMPTS",
        "core::hint::spin_loop()",
        "join_waiters.fetch_add(1, Ordering::AcqRel)",
        "join_waiters.load(Ordering::Acquire) != 0",
        "wait_signal.notify_all()",
    ] {
        assert!(
            source.contains(required),
            "scheduler join must retain fast quiescent spin and gated condvar path through {required}"
        );
    }
}

#[test]
fn ready_task_comparison_paths_compute_the_same_sum() {
    let moirai = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("Moirai runtime must start");

    let moirai_sum = AtomicUsize::new(0);
    moirai
        .scope(|scope| {
            for value in 0..READY_COUNT {
                let moirai_sum = &moirai_sum;
                scope.spawn(move |_| {
                    moirai_sum.fetch_add(value.wrapping_add(1), Ordering::Relaxed);
                })?;
            }
            Ok(())
        })
        .expect("Moirai scope must complete");

    let moirai_indexed_reduce_sum = moirai
        .map_reduce_indexed(
            READY_COUNT,
            0usize,
            |value| value.wrapping_add(1),
            usize::wrapping_add,
        )
        .expect("Moirai indexed reduction must complete");

    let rayon_sum = AtomicUsize::new(0);
    let rayon = rayon::ThreadPoolBuilder::new()
        .num_threads(WORKER_THREADS)
        .build()
        .expect("Rayon pool must start");
    rayon.scope(|scope| {
        for value in 0..READY_COUNT {
            let rayon_sum = &rayon_sum;
            scope.spawn(move |_| {
                rayon_sum.fetch_add(value.wrapping_add(1), Ordering::Relaxed);
            });
        }
    });

    let tokio = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(WORKER_THREADS)
        .enable_all()
        .build()
        .expect("Tokio runtime must start");
    let tokio_sum = tokio.block_on(async {
        let handles = (0..READY_COUNT)
            .map(|value| tokio::spawn(async move { value.wrapping_add(1) }))
            .collect::<Vec<_>>();

        let mut sum = 0usize;
        for handle in handles {
            sum = sum.wrapping_add(handle.await.expect("Tokio task must complete"));
        }
        sum
    });

    let expected = expected_ready_sum(READY_COUNT);
    assert_eq!(moirai_sum.load(Ordering::Relaxed), expected);
    assert_eq!(moirai_indexed_reduce_sum, expected);
    assert_eq!(rayon_sum.load(Ordering::Relaxed), expected);
    assert_eq!(tokio_sum, expected);

    moirai.shutdown();
}

#[test]
fn mixed_unified_comparison_paths_compute_the_same_sum() {
    const MIXED_COUNT: usize = 17;

    let moirai = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("Moirai runtime must start");

    let moirai_async_handles = (0..MIXED_COUNT)
        .map(|value| moirai.spawn_async(async move { value.wrapping_add(1) }))
        .collect::<Vec<_>>();
    let moirai_scope_sum = AtomicUsize::new(0);
    moirai
        .scope(|scope| {
            for value in 0..MIXED_COUNT {
                let moirai_scope_sum = &moirai_scope_sum;
                scope.spawn(move |_| {
                    moirai_scope_sum.fetch_add(value.wrapping_add(1), Ordering::Relaxed);
                })?;
            }
            Ok(())
        })
        .expect("Moirai mixed scope must complete");
    let mut moirai_sum = moirai_scope_sum.load(Ordering::Relaxed).wrapping_add(
        moirai
            .map_reduce_indexed(
                MIXED_COUNT,
                0usize,
                |value| value.wrapping_add(1),
                usize::wrapping_add,
            )
            .expect("Moirai mixed indexed reduction must complete"),
    );
    for handle in moirai_async_handles {
        moirai_sum = moirai_sum.wrapping_add(
            handle
                .join()
                .expect("Moirai mixed async handle must be attached")
                .expect("Moirai mixed async task must complete"),
        );
    }

    let rayon = rayon::ThreadPoolBuilder::new()
        .num_threads(WORKER_THREADS)
        .build()
        .expect("Rayon pool must start");
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(WORKER_THREADS)
        .enable_all()
        .build()
        .expect("Tokio runtime must start");

    let tokio_rayon_sum = tokio.block_on(async {
        let async_handles = (0..MIXED_COUNT)
            .map(|value| tokio::spawn(async move { value.wrapping_add(1) }))
            .collect::<Vec<_>>();

        let rayon_scope_sum = AtomicUsize::new(0);
        rayon.scope(|scope| {
            for value in 0..MIXED_COUNT {
                let rayon_scope_sum = &rayon_scope_sum;
                scope.spawn(move |_| {
                    rayon_scope_sum.fetch_add(value.wrapping_add(1), Ordering::Relaxed);
                });
            }
        });
        let mut sum = rayon_scope_sum
            .load(Ordering::Relaxed)
            .wrapping_add(rayon.install(|| {
                (0..MIXED_COUNT)
                    .into_par_iter()
                    .map(|value| value.wrapping_add(1))
                    .sum::<usize>()
            }));
        for handle in async_handles {
            sum = sum.wrapping_add(handle.await.expect("Tokio mixed async task must complete"));
        }
        sum
    });

    let expected = expected_ready_sum(MIXED_COUNT).wrapping_mul(3);
    assert_eq!(moirai_sum, expected);
    assert_eq!(tokio_rayon_sum, expected);

    moirai.shutdown();
}

#[test]
fn real_application_mixed_workload_contract_uses_closed_form_checksum() {
    const REQUEST_RECORDS: usize = 13;
    const CHANNEL_RECORDS: usize = 5;
    const ANALYTICS_RECORDS: usize = 97;

    let async_scope_component = expected_ready_sum(REQUEST_RECORDS).wrapping_mul(8);
    let channel_component = expected_ready_sum(CHANNEL_RECORDS).wrapping_mul(5);
    let analytics_component = expected_ready_sum(ANALYTICS_RECORDS).wrapping_mul(3);
    let expected = async_scope_component
        .wrapping_add(channel_component)
        .wrapping_add(analytics_component);

    let (tx, rx) = moirai_core::channel::spsc::<usize>(CHANNEL_RECORDS.next_power_of_two());
    for value in 0..CHANNEL_RECORDS {
        tx.send(value.wrapping_add(1).wrapping_mul(5))
            .expect("Moirai SPSC channel must accept checksum record");
    }
    let mut moirai_channel_sum = 0usize;
    for _ in 0..CHANNEL_RECORDS {
        moirai_channel_sum = moirai_channel_sum.wrapping_add(
            rx.recv()
                .expect("Moirai SPSC channel must receive checksum record"),
        );
    }

    let moirai = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("Moirai runtime must start");
    let moirai_analytics_sum = moirai
        .map_reduce_indexed(
            ANALYTICS_RECORDS,
            0usize,
            |value| value.wrapping_add(1).wrapping_mul(3),
            usize::wrapping_add,
        )
        .expect("Moirai analytics reduction must complete");
    let moirai_sum = expected_ready_sum(REQUEST_RECORDS)
        .wrapping_mul(8)
        .wrapping_add(moirai_channel_sum)
        .wrapping_add(moirai_analytics_sum);

    assert_eq!(moirai_channel_sum, channel_component);
    assert_eq!(moirai_sum, expected);
    moirai.shutdown();
}

#[test]
fn benchmark_spawn_smoke_path_returns_values() {
    let moirai = Moirai::builder()
        .worker_threads(2)
        .build()
        .expect("Moirai runtime must start");

    let handles = (0..10)
        .map(|value| {
            let task = TaskBuilder::new().build(move || value * 2);
            moirai.spawn(task)
        })
        .collect::<Vec<_>>();

    let results = handles
        .into_iter()
        .map(|handle| handle.join().expect("task handle must be attached"))
        .collect::<Result<Vec<_>, _>>()
        .expect("benchmark smoke tasks must not fail");

    assert_eq!(results, (0..10).map(|value| value * 2).collect::<Vec<_>>());
    moirai.shutdown();
}

#[test]
fn simd_benchmark_setup_computes_expected_values() {
    let a = vec![1.0; 64];
    let b = vec![2.0; 64];
    let mut result = vec![0.0; 64];

    simd::add(&a, &b, &mut result);

    assert_eq!(result, vec![3.0; 64]);
}

#[test]
fn rayon_map_reduce_reference_matches_closed_form_sum() {
    let rayon = rayon::ThreadPoolBuilder::new()
        .num_threads(WORKER_THREADS)
        .build()
        .expect("Rayon pool must start");

    let sum: u64 = rayon.install(|| (0..MAP_REDUCE_COUNT).into_par_iter().map(cpu_work).sum());

    assert_eq!(sum, expected_cpu_work_sum(MAP_REDUCE_COUNT));
}
