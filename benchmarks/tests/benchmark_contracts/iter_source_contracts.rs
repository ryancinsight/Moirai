#[test]
fn iter_thread_pool_uses_monomorphized_erased_jobs() {
    let source = read_benchmark("../moirai-iter/src/base.rs");

    for required in [
        "sender: Option<std::sync::mpsc::Sender<ErasedThreadJob>>",
        "struct ErasedThreadJob",
        "run: unsafe fn(NonNull<()>)",
        "drop: unsafe fn(NonNull<()>)",
        "run_thread_job::<F>",
        "drop_thread_job::<F>",
        "std::sync::mpsc::channel::<ErasedThreadJob>()",
        "job.run()",
        "ErasedThreadJob::new(job)",
        "test_erased_thread_job_runs_once",
        "test_erased_thread_job_drops_unrun_capture",
    ] {
        assert!(
            source.contains(required),
            "iterator thread pool must retain monomorphized erased jobs through {required}"
        );
    }

    for prohibited in [
        "Sender<Box<dyn FnOnce() + Send>>",
        "channel::<Box<dyn FnOnce() + Send",
        "s.send(Box::new(job))",
    ] {
        assert!(
            !source.contains(prohibited),
            "iterator thread pool must not reintroduce dynamic job dispatch through {prohibited}"
        );
    }
}

#[test]
fn iterator_base_does_not_expose_boxed_future_execution_trait() {
    let source = read_benchmark("../moirai-iter/src/base.rs");

    for prohibited in [
        "pub trait ExecutionBase: Send + Sync + 'static",
        "Pin<Box<dyn Future",
        "Box::pin(async move",
        "execute_each<T, F>",
        "execute_map<T, R, F>",
        "execute_filter<T, F>",
    ] {
        assert!(
            !source.contains(prohibited),
            "iterator base must not reintroduce unused boxed-future execution trait shape through {prohibited}"
        );
    }
}

#[test]
fn channel_fusion_uses_typed_channels_without_placeholder_pipeline() {
    let source = read_benchmark("../moirai-iter/src/channel_fusion.rs");

    for required in [
        "pub struct ChannelSplitter<T, I, C>",
        "channels: Vec<C>",
        "C: FusableChannel<T>",
        "pub fn add_channel(mut self, channel: C) -> Self",
        "pub struct ChannelMerger<T, C>",
        "buffer: VecDeque<T>",
        "pop_front()",
        "fn split_channels<C>(",
        "test_channel_merger_fair_merge_uses_fifo_order",
        "test_channel_splitter_broadcast_clones_to_every_channel",
    ] {
        assert!(
            source.contains(required),
            "channel fusion must retain typed zero-cost channel structure through {required}"
        );
    }

    for prohibited in [
        "Vec<Box<dyn FusableChannel<T>>>",
        "Box<dyn FusableChannel<T>>",
        "SplitStrategy::Hash",
        "pub struct Pipeline",
        "PipelineStage",
        "let hash = 0",
        "remove(0)",
        ".add_channel(Box::new",
    ] {
        assert!(
            !source.contains(prohibited),
            "channel fusion must not reintroduce dynamic or placeholder structure through {prohibited}"
        );
    }
}

#[test]
fn streaming_iter_uses_monomorphized_producer_and_fifo_buffer() {
    let source = format!(
        "{}\n{}\n{}",
        read_benchmark("../moirai-iter/src/iter_ops.rs"),
        read_benchmark("../moirai-iter/src/iter_ops/streaming.rs"),
        read_benchmark("../moirai-iter/src/iter_ops/tests.rs")
    );

    for required in [
        "mod streaming;",
        "streaming::StreamingIter",
        "pub struct StreamingIter<T, F>",
        "buffer: VecDeque<T>",
        "producer: F",
        "F: FnMut() -> Option<T>",
        "VecDeque::with_capacity(capacity)",
        "capacity: capacity.max(1)",
        "push_back(item)",
        "pop_front()",
        "streaming_iter_preserves_fifo_order",
    ] {
        assert!(
            source.contains(required),
            "streaming iterator must retain monomorphized producer/FIFO shape through {required}"
        );
    }

    for prohibited in [
        "producer: Box<dyn FnMut() -> Option<T>>",
        "producer: Box::new(producer)",
        "Box<dyn FnMut",
        "self.buffer.remove(0)",
    ] {
        assert!(
            !source.contains(prohibited),
            "streaming iterator must not reintroduce boxed producer or shifting FIFO through {prohibited}"
        );
    }
}

#[test]
fn rayon_adapter_surface_audit_tracks_current_iterator_scope() {
    let audit = read_benchmark("../docs/rayon_adapter_surface_audit.md");
    let adapter_benchmark = read_benchmark("benches/iterator_adapter_comparison.rs");
    let benchmark_manifest = read_benchmark("Cargo.toml");
    let adapter_source = format!(
        "{}\n{}\n{}\n{}\n{}",
        read_benchmark("../moirai-iter/src/parallel.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/chunks.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/side_effect.rs"),
        read_benchmark("../moirai-iter/src/parallel/sources.rs")
    );
    let source = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
        read_benchmark("../moirai-iter/src/parallel.rs"),
        read_benchmark("../moirai-iter/src/parallel/traits.rs"),
        read_benchmark("../moirai-iter/src/parallel/sources.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/chunks.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/side_effect.rs"),
        read_benchmark("../moirai-iter/src/parallel/consumers.rs"),
        read_benchmark("../moirai-iter/src/parallel/tests.rs"),
        read_benchmark("../moirai-iter/src/lib.rs")
    );

    for required in [
        "Moirai does not currently provide full Rayon adapter parity",
        "Covered subset",
        "Sequential by contract",
        "Slice extension boundary",
        "IndexedParallelIterator",
        "Moirai::for_each_indexed",
        "Moirai::map_reduce_indexed",
        "sole indexed public scheduler paths",
        "Rayon-style non-indexed adapter subset",
        "ISSUE-092",
        "ISSUE-093",
        "ISSUE-094",
        "ISSUE-101",
        "ISSUE-102",
        "ISSUE-103",
        "ISSUE-104",
        "ISSUE-105",
        "Competitive Rayon performance claims must continue using value-checked benchmark paths",
    ] {
        assert!(
            audit.contains(required),
            "Rayon adapter audit must classify {required}"
        );
    }

    for required in [
        "pub trait ParallelIterator",
        "mod traits;",
        "mod sources;",
        "mod consumers;",
        "fn map<F, R>",
        "fn map_with<T, F, R>",
        "fn map_init<Init, T, F, R>",
        "fn update<F>(self, update_fn: F) -> Update<Self, F>",
        "fn filter<F>",
        "fn inspect<F>(self, inspect_fn: F) -> Inspect<Self, F>",
        "fn panic_fuse(self) -> PanicFuse<Self>",
        "fn filter_map<F, R>",
        "fn while_some<T>(self) -> WhileSome<Self>",
        "fn flat_map<F, U>",
        "fn enumerate(self)",
        "fn zip<J>",
        "fn take(self, count: usize) -> Take<Self>",
        "fn skip(self, count: usize) -> Skip<Self>",
        "fn chain<J>(self, other: J) -> Chain<Self, J>",
        "fn intersperse(self, separator: Self::Item) -> Intersperse<Self>",
        "fn rev(self) -> Rev<Self>",
        "fn chunks(self, chunk_size: usize) -> Chunks<Self>",
        "fn copied<'data, T>(self) -> Copied<Self>",
        "fn cloned<'data, T>(self) -> Cloned<Self>",
        "fn seq_items_window(self, skip: usize, take: Option<usize>)",
        "fn seq_items_reversed(self)",
        "fn seq_items_reversed_prefix(self, count: usize)",
        "fn reduce<F>",
        "fn reduce_with<F>",
        "fn try_reduce<Identity, F, T, E>",
        "fn sum<S>(self) -> S",
        "fn product<P>(self) -> P",
        "fn min(self) -> Option<Self::Item>",
        "fn max(self) -> Option<Self::Item>",
        "fn min_by<F>(self, compare: F) -> Option<Self::Item>",
        "fn max_by<F>(self, compare: F) -> Option<Self::Item>",
        "fn min_by_key<K, F>(self, key_fn: F) -> Option<Self::Item>",
        "fn max_by_key<K, F>(self, key_fn: F) -> Option<Self::Item>",
        "fn fold<T, F>",
        "fn collect<C>",
        "fn partition<C, F>(self, predicate: F) -> (C, C)",
        "fn unzip<A, B, FromA, FromB>(self) -> (FromA, FromB)",
        "fn count(self)",
        "fn find_last<F>(self, predicate: F) -> Option<Self::Item>",
        "fn position_first<F>(self, predicate: F) -> Option<usize>",
        "fn position_any<F>(self, predicate: F) -> Option<usize>",
        "fn position_last<F>(self, predicate: F) -> Option<usize>",
        "fn find_map_first<F, R>(self, map_fn: F) -> Option<R>",
        "fn find_map_any<F, R>(self, map_fn: F) -> Option<R>",
        "fn find_map_last<F, R>(self, map_fn: F) -> Option<R>",
        "fn any<F>",
        "fn all<F>",
        "fn for_each_with<T, F>(self, init: T, op: F)",
        "fn for_each_init<Init, T, F>(self, init: Init, op: F)",
        "fn try_for_each<F, E>(self, op: F) -> Result<(), E>",
        "fn try_for_each_with<T, F, E>(self, init: T, op: F) -> Result<(), E>",
        "fn try_for_each_init<Init, T, F, E>(self, init: Init, op: F) -> Result<(), E>",
        "fn find_any<F>",
        "pub trait IntoParallelIterator",
        "pub trait IntoParallelRefIterator",
        "focused Rayon-style non-indexed adapter subset",
        "Core parallel iterator trait for Moirai's Rayon-style non-indexed subset",
        "pub struct Inspect<I, F>",
        "for item in &items",
        "inspect_fn(item);",
        "pub struct PanicFuse<I>",
        "struct PanicFusePolicy",
        "std::mem::size_of::<PanicFusePolicy>()",
        "pub struct FilterMap<I, F>",
        "pub struct MapWith<I, T, F>",
        "pub struct MapInit<I, Init, F>",
        "pub struct Update<I, F>",
        "filter_map(self.filter_map_fn)",
        "pub struct WhileSome<I>",
        "map_while(|item| item)",
        "pub struct FlatMap<I, F>",
        "flat_map(self.flat_map_fn)",
        "pub struct Enumerate<I>",
        "pub struct Zip<I, J>",
        "pub struct Copied<I>",
        "pub struct Cloned<I>",
        "pub struct Take<I>",
        "pub struct Skip<I>",
        "pub struct Chain<I, J>",
        "pub struct Intersperse<I>",
        "pub struct Rev<I>",
        "pub struct Chunks<I>",
        "struct ChunkSize(usize)",
        "assert!(value != 0, \"chunk size must be non-zero\");",
        "pub use adapters::{",
        "Inspect",
        "PanicFuse",
        "Chunks",
        "Copied",
        "Cloned",
        "Parallel range iterator for Moirai's Rayon-style non-indexed subset",
        "Moirai::for_each_indexed",
        "Moirai::map_reduce_indexed",
        "impl<T: Send + Sync + Clone + 'static> IntoParallelIterator for Vec<T>",
        "impl IntoParallelIterator for std::ops::Range<usize>",
        "pub struct Reduction<T, F>",
        "let reduction: Reduction<Self::Item, F> = self.drive(ReduceConsumer::new(reduce_fn));",
        "let reduction: Reduction<Self::Item, F> = self.drive(ReduceWithConsumer::new(reduce_fn));",
        "Some(reduce_fn(left, right))",
        "self.data.is_empty()",
        "self.data.len() <= 1",
        "Preserve sequential value semantics for this API",
        "test_parallel_filter_map_retains_present_values",
        "test_parallel_map_with_uses_cloned_state",
        "test_parallel_map_init_uses_initialized_state",
        "test_parallel_update_mutates_items_before_yielding",
        "test_parallel_while_some_unwraps_present_prefix",
        "test_parallel_while_some_empty_when_first_is_none",
        "test_parallel_try_for_each_returns_ok_after_processing_all_items",
        "test_parallel_try_for_each_returns_first_error",
        "test_parallel_flat_map_preserves_flattened_order",
        "test_parallel_enumerate_pairs_logical_indices",
        "test_parallel_zip_stops_at_shorter_input",
        "test_parallel_copied_materializes_borrowed_copy_values",
        "test_parallel_cloned_materializes_borrowed_clone_values",
        "test_parallel_take_keeps_prefix",
        "test_parallel_skip_discards_prefix",
        "test_parallel_take_and_skip_saturate_at_bounds",
        "test_parallel_chain_preserves_left_then_right_order",
        "test_parallel_intersperse_inserts_separator_between_items",
        "test_parallel_intersperse_preserves_empty_and_singleton_streams",
        "test_parallel_rev_reverses_logical_order",
        "test_parallel_inspect_observes_items_without_changing_output",
        "test_parallel_panic_fuse_preserves_values",
        "test_parallel_panic_fuse_propagates_panic",
        "test_parallel_chunks_groups_full_chunks_and_tail",
        "test_parallel_chunks_rejects_zero_size",
        "test_parallel_partition_preserves_relative_order",
        "test_parallel_unzip_splits_pair_streams",
        "test_parallel_reduce_empty_returns_none",
        "test_parallel_try_reduce_returns_reduced_value",
        "test_parallel_try_reduce_returns_first_error",
        "test_parallel_sum_and_product_match_standard_values",
        "test_parallel_min_and_max_match_standard_values",
        "test_parallel_min_max_by_use_comparator",
        "test_parallel_min_max_by_key_use_key_function",
        "test_parallel_find_last_returns_last_matching_value",
        "test_parallel_position_terminals_return_logical_indices",
        "test_parallel_find_map_first_maps_first_present_value",
        "test_parallel_find_map_any_maps_present_value",
        "test_parallel_find_map_last_maps_last_present_value",
        "test_parallel_for_each_with_uses_cloned_state",
        "test_parallel_for_each_init_uses_initialized_state",
        "test_parallel_try_for_each_with_uses_cloned_state_and_propagates_error",
        "test_parallel_try_for_each_init_uses_initialized_state_and_propagates_error",
    ] {
        assert!(
            source.contains(required),
            "parallel iterator source must retain audited surface marker {required}"
        );
    }

    for prohibited in [
        "known limitation of the current consumer",
        "Simplified - should use reduce_fn",
        "Should use reduce_fn",
        "pub struct FoldConsumer",
        "Rayon-compatible API",
        "matches Rayon's API",
        "Rayon compatibility",
    ] {
        assert!(
            !source.contains(prohibited),
            "parallel iterator source must not reintroduce prototype reduction marker {prohibited}"
        );
    }

    for prohibited in ["context: ParallelContext", "ParallelContext::new()"] {
        assert!(
            !adapter_source.contains(prohibited),
            "pure parallel adapter source must not allocate execution context through {prohibited}"
        );
    }

    for required in [
        "name = \"iterator_adapter_comparison\"",
        "moirai_indexed_pipeline",
        "rayon_indexed_pipeline",
        "moirai_filter_flat_pipeline",
        "rayon_filter_flat_pipeline",
        "moirai_map_state_pipeline",
        "rayon_map_state_pipeline",
        "moirai_update_pipeline",
        "rayon_update_pipeline",
        "moirai_while_some_pipeline",
        "rayon_while_some_pipeline",
        "moirai_try_for_each_pipeline",
        "rayon_try_for_each_pipeline",
        "moirai_for_each_state_pipeline",
        "rayon_for_each_state_pipeline",
        "moirai_try_for_each_state_pipeline",
        "rayon_try_for_each_state_pipeline",
        "moirai_try_reduce_pipeline",
        "rayon_try_reduce_pipeline",
        "moirai_chain_rev_pipeline",
        "rayon_chain_rev_pipeline",
        "moirai_intersperse_pipeline",
        "rayon_intersperse_pipeline",
        "moirai_inspect_chunks_pipeline",
        "rayon_inspect_chunks_pipeline",
        "moirai_partition_pipeline",
        "rayon_partition_pipeline",
        "moirai_terminal_reducer_pipeline",
        "rayon_terminal_reducer_pipeline",
        "moirai_ordered_reducer_pipeline",
        "rayon_ordered_reducer_pipeline",
        "moirai_find_map_pipeline",
        "rayon_find_map_pipeline",
        "moirai_position_pipeline",
        "rayon_position_pipeline",
        "moirai_ref_copied_cloned_pipeline",
        "rayon_ref_copied_cloned_pipeline",
        "moirai_unzip_pipeline",
        "rayon_unzip_pipeline",
        "assert_eq!(moirai_expected, rayon_expected)",
        "iterator_adapter_indexed_pipeline",
        "iterator_adapter_filter_flat_pipeline",
        "iterator_adapter_map_state",
        "iterator_adapter_update",
        "iterator_adapter_while_some",
        "iterator_adapter_try_for_each",
        "iterator_adapter_for_each_state",
        "iterator_adapter_try_for_each_state",
        "iterator_adapter_try_reduce",
        "iterator_adapter_chain_rev_pipeline",
        "iterator_adapter_intersperse",
        "iterator_adapter_inspect_chunks_pipeline",
        "iterator_adapter_partition_pipeline",
        "iterator_adapter_terminal_reducers",
        "iterator_adapter_ordered_reducers",
        "iterator_adapter_find_map",
        "iterator_adapter_position",
        "iterator_adapter_ref_copy_clone",
        "iterator_adapter_unzip",
    ] {
        assert!(
            adapter_benchmark.contains(required) || benchmark_manifest.contains(required),
            "iterator adapter benchmark must retain comparison marker {required}"
        );
    }
}

#[test]
fn sorting_slice_extension_is_value_semantic_and_benchmarked() {
    let audit = read_benchmark("../docs/rayon_adapter_surface_audit.md");
    let comparison_report = read_benchmark("../docs/moirai_rayon_tokio_comparison.md");
    let sorting_source = read_benchmark("../moirai-iter/src/parallel/sorting.rs");
    let parallel_root = read_benchmark("../moirai-iter/src/parallel.rs");
    let lib_root = read_benchmark("../moirai-iter/src/lib.rs");
    let sorting_benchmark = read_benchmark("benches/sorting_comparison.rs");
    let benchmark_manifest = read_benchmark("Cargo.toml");

    for required in [
        "mod sorting;",
        "pub use sorting::ParallelSliceMut;",
        "ParallelSliceMut",
    ] {
        assert!(
            parallel_root.contains(required) || lib_root.contains(required),
            "parallel sorting boundary must retain exported marker {required}"
        );
    }

    for required in [
        "pub trait ParallelSliceMut<T: Send>",
        "fn par_sort(&mut self)",
        "fn par_sort_by<F>(&mut self, compare: F)",
        "fn par_sort_by_key<K, F>(&mut self, f: F)",
        "fn par_sort_unstable(&mut self)",
        "fn par_sort_unstable_by<F>(&mut self, compare: F)",
        "fn par_sort_unstable_by_key<K, F>(&mut self, f: F)",
        "const STABLE_SEQUENTIAL_THRESHOLD: usize = 2048;",
        "const UNSTABLE_SEQUENTIAL_THRESHOLD: usize = 16_384;",
        "par_merge_sort_impl(self, &compare, &pool)",
        "par_sort_unstable_by_impl(self, &compare, &pool)",
        "test_sorting_empty_and_single",
        "test_sorting_large_random",
        "test_sorting_stability",
        "test_panic_safety_no_double_drop",
    ] {
        assert!(
            sorting_source.contains(required),
            "parallel sorting source must retain marker {required}"
        );
    }

    for prohibited in [
        "sorting adapters | no parallel sorting adapters",
        "sorting adapters remain unsupported",
    ] {
        assert!(
            !audit.contains(prohibited),
            "Rayon adapter audit must not retain obsolete sorting unsupported marker {prohibited}"
        );
        assert!(
            !comparison_report.contains(prohibited),
            "Rayon/Tokio comparison report must not retain obsolete sorting unsupported marker {prohibited}"
        );
    }

    for required in [
        "Slice extension boundary",
        "ParallelSliceMut",
        "sorting_comparison",
        "ParallelSliceMut` benchmark",
    ] {
        assert!(
            audit.contains(required),
            "Rayon adapter audit must retain sorting boundary marker {required}"
        );
    }

    for required in [
        "Rayon Adapter Surface Boundary",
        "inspect",
        "panic_fuse",
        "chunks",
        "partition",
        "ParallelSliceMut` for stable and unstable slice sorting",
        "Stable slice sort",
        "Unstable slice sort",
        "Full Rayon ecosystem parity is incomplete",
    ] {
        assert!(
            comparison_report.contains(required),
            "Rayon/Tokio comparison report must retain current adapter marker {required}"
        );
    }

    for required in [
        "name = \"sorting_comparison\"",
        "MoiraiParallelSliceMut::par_sort",
        "RayonParallelSliceMut::par_sort",
        "MoiraiParallelSliceMut::par_sort_unstable",
        "RayonParallelSliceMut::par_sort_unstable",
        "assert_eq!(moirai_stable, rayon_stable)",
        "assert_eq!(moirai_unstable, rayon_unstable)",
        "parallel_sorting_stable",
        "parallel_sorting_unstable",
    ] {
        assert!(
            sorting_benchmark.contains(required) || benchmark_manifest.contains(required),
            "sorting benchmark must retain comparison marker {required}"
        );
    }
}

#[test]
fn async_iterator_terminal_futures_are_value_semantic_and_benchmarked() {
    let async_source = read_benchmark("../moirai-iter/src/async_iter.rs");
    let async_benchmark = read_benchmark("benches/async_iterator_comparison.rs");
    let benchmark_manifest = read_benchmark("Cargo.toml");

    for required in [
        "fn into_vec(self) -> Vec<Self::Item>",
        "pub struct AsyncCollect<I, C>",
        "iter: Option<I>",
        "collection.extend(iter.into_vec())",
        "pub struct AsyncFold<I, T, F>",
        "accumulator: Option<T>",
        "async fold polled after completion",
        "pub struct AsyncReduce<I, F>",
        "Poll::Ready(Some(accumulator))",
        "test_async_vec_iter",
        "test_async_map",
        "test_parallel_async_map",
        "test_async_filter_fold_reduce_values",
        "assert_eq!(result, vec![1, 2, 3, 4, 5])",
        "assert_eq!(result, vec![2, 4, 6, 8, 10])",
        "assert_eq!(filtered, vec![2, 4, 6])",
        "assert_eq!(folded, 4)",
        "assert_eq!(reduced, Some(10))",
        "use futures::stream::{self, StreamExt};",
        "let concurrency = self.concurrency.max(1);",
        ".buffered(concurrency)",
        "test_parallel_async_map_uses_bounded_in_flight_work",
        "test_parallel_async_filter_uses_bounded_in_flight_work",
        "test_parallel_async_for_each_uses_bounded_in_flight_work",
        "max_active.fetch_max(now, Ordering::SeqCst)",
        "assert_eq!(max_active.load(Ordering::SeqCst), 3)",
        "assert_eq!(max_active.load(Ordering::SeqCst), 4)",
        "assert_eq!(max_active.load(Ordering::SeqCst), 2)",
    ] {
        assert!(
            async_source.contains(required),
            "async iterator implementation must retain value-semantic marker {required}"
        );
    }

    for prohibited in [
        "Poll::Ready(C::default())",
        "unsafe { std::ptr::read(acc) }",
        "Test would verify",
        "Simplified implementation",
        "let _result",
    ] {
        assert!(
            !async_source.contains(prohibited),
            "async iterator implementation must not reintroduce placeholder marker {prohibited}"
        );
    }

    for required in [
        "name = \"async_iterator_comparison\"",
        "moirai_ready_pipeline",
        "tokio_joinset_ready_pipeline",
        "moirai_bounded_yield_pipeline",
        "tokio_bounded_yield_pipeline",
        "BOUNDED_CONCURRENCY",
        "pending_once",
        "JoinSet::new()",
        "assert_eq!(moirai_expected, tokio_expected)",
        "async_iterator_ready_pipeline",
        "async_iterator_bounded_yield_pipeline",
        "tokio_joinset",
    ] {
        assert!(
            async_benchmark.contains(required) || benchmark_manifest.contains(required),
            "async iterator benchmark must retain comparison marker {required}"
        );
    }
}
