#[test]
fn iterator_base_does_not_expose_boxed_future_execution_trait() {
    let source = format!(
        "{}\n{}",
        read_benchmark("../moirai-iter/src/base.rs"),
        read_benchmark("../moirai-iter/src/base/tests.rs")
    );

    for required in [
        "pub const fn inner(&self) -> &I",
        "pub fn context(&self) -> &Arc<C>",
        "pub fn into_parts(self) -> (I, Arc<C>)",
        "pub const fn function(&self) -> &F",
        "pub const fn predicate(&self) -> &F",
        "pub const fn size(&self) -> usize",
        "#[path = \"base/tests.rs\"]",
        "base_adapters_expose_components_without_dead_fields",
    ] {
        assert!(
            source.contains(required),
            "iterator base adapters must expose live fields through {required}"
        );
    }

    for prohibited in [
        "pub trait ExecutionBase: Send + Sync + 'static",
        "Pin<Box<dyn Future",
        "Box::pin(async move",
        "execute_each<T, F>",
        "execute_map<T, R, F>",
        "execute_filter<T, F>",
        "#[allow(dead_code)]",
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
fn iter_ops_parallel_iter_uses_scoped_borrowed_chunks() {
    let source = format!(
        "{}\n{}\n{}",
        read_benchmark("../moirai-iter/src/iter_ops.rs"),
        read_benchmark("../moirai-iter/src/iter_ops/parallel.rs"),
        read_benchmark("../moirai-iter/src/iter_ops/tests.rs")
    );
    let benchmark = read_benchmark("benches/iter_ops_parallel_comparison.rs");
    let manifest = read_benchmark("Cargo.toml");

    for required in [
        "mod parallel;",
        "parallel::ParallelIter",
        "pub struct ParallelIter<T>",
        "data: Vec<T>",
        "DEFAULT_RING_BUFFER_CAPACITY",
        "should_execute_scoped",
        "chunk_size > DEFAULT_RING_BUFFER_CAPACITY",
        // The fan-out lends borrowed chunks to scheduler lanes and joins before
        // returning; the fallback re-runs the same closure on the caller. Both
        // are what keep the closure non-`'static`, so both are the marker.
        "for_each_indexed",
        "sequential_fallback_permitted",
        ".chunks(chunk_size)",
        "F: Fn(&T) -> U + Send + Sync",
        "parallel_iter_map_borrows_data_without_static_closure",
        "parallel_iter_reduce_matches_sequential_sum",
        "parallel_iter_reduce_empty_returns_identity",
    ] {
        assert!(
            source.contains(required),
            "iter_ops ParallelIter must retain scoped borrowed chunk marker {required}"
        );
    }

    for required in [
        "name = \"iter_ops_parallel_comparison\"",
        "iter_ops_parallel_map",
        "iter_ops_parallel_reduce",
        "moirai_parallel_map",
        "rayon_parallel_map",
        "moirai_parallel_reduce",
        "rayon_parallel_reduce",
        "assert_eq!",
    ] {
        assert!(
            benchmark.contains(required) || manifest.contains(required),
            "iter_ops ParallelIter benchmark contract must retain marker {required}"
        );
    }

    for prohibited in [
        "data: Arc<Vec<T>>",
        "Arc::new(data)",
        "self.data.clone()",
        "F: Fn(&T) -> U + Send + Sync + 'static",
        "F: Fn(T, &T) -> T + Send + Sync + 'static",
        "thread::spawn(move ||",
    ] {
        assert!(
            !source.contains(prohibited),
            "iter_ops ParallelIter must not reintroduce owned refcount or unscoped worker marker {prohibited}"
        );
    }
}

#[test]
fn cache_zero_copy_parallel_iter_borrows_scoped_map_inputs() {
    let source = read_benchmark("../moirai-iter/src/cache.rs");
    let benchmark = read_benchmark("benches/cache_iterator_comparison.rs");
    let manifest = read_benchmark("Cargo.toml");

    for required in [
        "pub struct ZeroCopyParallelIter<'a, T>",
        "data: &'a [T]",
        "fn should_execute_scoped_cache<T>(len: usize, chunk_size: usize) -> bool",
        "DEFAULT_RING_BUFFER_CAPACITY",
        "should_execute_scoped_cache::<T>(self.data.len(), self.chunk_size)",
        "return self.data.iter().map(&func).collect();",
        ".chunks(chunk_size).enumerate()",
        "let func_ref = &func",
        "func_ref(item)",
        "zero_copy_parallel_map_borrows_data_and_closure",
        "zero_copy_parallel_map_matches_sequential_values",
        "fn reduce_owned_pairs<T, F>(items: Vec<T>, func: &F) -> Vec<T>",
        "current_results = reduce_owned_pairs(current_results, &func);",
        "reduce_owned_pairs_moves_non_clone_odd_value",
        "cache_scoped_execution_gate_uses_batch_capacity_floor",
        "zero_copy_parallel_reduce_accepts_non_clone_reducer",
    ] {
        assert!(
            source.contains(required),
            "cache zero-copy parallel iterator must retain borrowed scoped map marker {required}"
        );
    }

    for required in [
        "name = \"cache_iterator_comparison\"",
        "cache_iterator_zero_copy_map",
        "cache_iterator_zero_copy_reduce",
        "moirai_zero_copy_map",
        "rayon_borrowed_map",
        "moirai_zero_copy_reduce",
        "rayon_borrowed_reduce",
        "cache_iterator_zero_copy_large_reduce",
        "moirai_zero_copy_large_reduce",
        "rayon_borrowed_large_reduce",
        "assert_eq!",
    ] {
        assert!(
            benchmark.contains(required) || manifest.contains(required),
            "cache iterator benchmark contract must retain marker {required}"
        );
    }

    for prohibited in [
        "use std::sync::Arc",
        "let func = Arc::new(func)",
        "let data = Arc::new(self.data)",
        "Arc::clone(&func)",
        "Arc::clone(&data)",
        "F: Fn(&T, &T) -> T + Send + Sync + Clone",
        "let chunk = current_results[chunk_start..chunk_end].to_vec();",
        "let func = func.clone();",
    ] {
        assert!(
            !source.contains(prohibited),
            "cache zero-copy parallel iterator must not reintroduce refcounted map or cloned reduce routing through {prohibited}"
        );
    }
}

#[test]
fn execution_context_iter_consumes_owned_chunks_without_clone() {
    let source = read_benchmark("../moirai-iter/src/execution/mod.rs");
    let benchmark = read_benchmark("benches/execution_context_comparison.rs");
    let manifest = read_benchmark("Cargo.toml");

    for required in [
        "fn owned_chunks<T>(items: Vec<T>, chunk_size: usize) -> Vec<Vec<T>>",
        "let mut iter = items.into_iter();",
        "chunks.push(iter.by_ref().take(take).collect());",
        "if items.len() <= chunk_size",
        "return Ok(items.into_iter().map(func).collect());",
        "non_clone_parallel_context_execute_iter_consumes_items",
        "non_clone_async_context_execute_iter_consumes_items",
        "owned_chunks_move_values_without_clone_bound",
    ] {
        assert!(
            source.contains(required),
            "execution context iterator must retain owned move marker {required}"
        );
    }

    for required in [
        "name = \"execution_context_comparison\"",
        "execution_context_owned_map",
        "moirai_parallel_context_map",
        "rayon_owned_map",
        "assert_eq!",
    ] {
        assert!(
            benchmark.contains(required) || manifest.contains(required),
            "execution context benchmark contract must retain marker {required}"
        );
    }

    for prohibited in [
        ".map(|chunk| chunk.to_vec())",
        "let result = func(item.clone());",
        "T: Send + Clone + 'static",
        "Simplified async execution",
    ] {
        assert!(
            !source.contains(prohibited),
            "execution context iterator must not reintroduce clone-bound owned chunking through {prohibited}"
        );
    }
}

#[test]
fn iterator_simd_surface_uses_generic_scalar_contract() {
    let source = read_benchmark("../moirai-iter/src/simd_iter.rs");
    let benchmark = read_benchmark("benches/iter_simd_comparison.rs");
    let manifest = read_benchmark("Cargo.toml");
    let audit = read_benchmark("../docs/rayon_tokio_gap_audit.md");
    let backup = benchmark_path("../moirai-iter/src/simd_iter_backup.rs");

    for required in [
        "pub trait SimdScalar",
        "mod sealed",
        "pub struct SimdSliceIter<'a, T>",
        "impl<'a, T: SimdScalar> SimdSliceIter<'a, T>",
        "pub fn add_slice(self, other: &'a [T]) -> Vec<T>",
        "pub fn scale(self, scalar: T) -> Vec<T>",
        "pub fn dot(self, other: &'a [T]) -> T",
        "pub struct CacheFriendlyIterator<T>",
        "(CACHE_LINE_SIZE / scalar_size).max(1)",
        "pub fn reduce<T, F, R>(data: &[T], identity: R, op: F) -> R",
        "pub fn filter<T, P>(data: Vec<T>, predicate: P) -> Vec<T>",
        "generic_slice_addition_preserves_values",
        "generic_slice_scale_preserves_native_precision_values",
        "generic_slice_dot_preserves_values",
        "cache_friendly_iterator_processes_large_elements",
        "simd_ops_reduce_and_filter_are_value_semantic",
    ] {
        assert!(
            source.contains(required),
            "SIMD iterator source must retain generic scalar marker {required}"
        );
    }

    for prohibited in [
        "SimdF32Iterator",
        "simd_add",
        "simd_multiply",
        "simd_dot_product",
        "simd_parallel_reduce",
        "pub const AVX2_F32_WIDTH",
        "pub const SSE2_F32_WIDTH",
        "CACHE_FRIENDLY_CHUNK_SIZE: usize = CACHE_LINE_SIZE / std::mem::size_of::<f32>()",
        "For now",
        "placeholder",
        "Real implementation",
    ] {
        assert!(
            !source.contains(prohibited),
            "SIMD iterator source must not retain non-generic or placeholder marker {prohibited}"
        );
    }

    assert!(
        !backup.exists(),
        "stale SIMD backup source must not remain in the repository"
    );

    for required in [
        "name = \"iter_simd_comparison\"",
        "SimdSliceIter::new(left).add_slice(right)",
        "SimdSliceIter::new(left).dot(right)",
        "assert_eq!(generic_add(&left, &right), scalar_add(&left, &right))",
        "assert_eq!(generic_dot(&left, &right), scalar_dot(&left, &right))",
        "iter_simd_generic_add",
        "iter_simd_generic_dot",
    ] {
        assert!(
            benchmark.contains(required) || manifest.contains(required),
            "SIMD iterator benchmark must retain executable marker {required}"
        );
    }

    for required in [
        "Iterator SIMD surface is generic",
        "iter_simd_comparison",
        "SimdSliceIter<T>",
    ] {
        assert!(
            audit.contains(required),
            "Rayon/Tokio audit must retain SIMD cleanup marker {required}"
        );
    }
}

#[test]
fn rayon_adapter_surface_audit_tracks_current_iterator_scope() {
    let audit = read_benchmark("../docs/rayon_adapter_surface_audit.md");
    let adapter_benchmark = read_benchmark("benches/iterator_adapter_comparison.rs");
    let regression_benchmark = read_benchmark("benches/parallel_iterator_regression.rs");
    let benchmark_manifest = read_benchmark("Cargo.toml");
    let adapter_source = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
        read_benchmark("../moirai-iter/src/parallel.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/blocks.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/chunks.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/pair.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/position.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/side_effect.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/stride.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/window.rs"),
        read_benchmark("../moirai-iter/src/parallel/sources.rs")
    );
    let source = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
        read_benchmark("../moirai-iter/src/parallel.rs"),
        read_benchmark("../moirai-iter/src/parallel/fallible.rs"),
        read_benchmark("../moirai-iter/src/parallel/indexed.rs"),
        read_benchmark("../moirai-iter/src/parallel/split.rs"),
        read_benchmark("../moirai-iter/src/parallel/traits.rs"),
        read_benchmark("../moirai-iter/src/parallel/sources.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/blocks.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/chunks.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/pair.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/position.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/side_effect.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/stride.rs"),
        read_benchmark("../moirai-iter/src/parallel/adapters/window.rs"),
        read_benchmark("../moirai-iter/src/parallel/consumers.rs"),
        read_benchmark("../moirai-iter/src/parallel/tests.rs"),
        read_benchmark("../moirai-iter/src/facade/mod.rs"),
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
        "bounded indexed source boundary",
        "full Rayon indexed producer/consumer adapter model",
        "ISSUE-092",
        "ISSUE-093",
        "ISSUE-094",
        "ISSUE-101",
        "ISSUE-102",
        "ISSUE-103",
        "ISSUE-104",
        "ISSUE-105",
        "ISSUE-178",
        "ISSUE-179",
        "ISSUE-180",
        "ISSUE-181",
        "ISSUE-182",
        "Competitive Rayon performance claims must continue using value-checked benchmark paths",
    ] {
        assert!(
            audit.contains(required),
            "Rayon adapter audit must classify {required}"
        );
    }

    for required in [
        "pub trait ParallelIterator",
        "mod indexed;",
        "mod fallible;",
        "mod split;",
        "mod position;",
        "mod traits;",
        "mod sources;",
        "mod consumers;",
        "mod blocks;",
        "mod window;",
        "pub trait IndexedParallelIterator",
        "pub use blocks::{ExponentialBlocks, UniformBlocks};",
        "pub use pair::{Interleave, InterleaveShortest, Zip, ZipEq};",
        "pub use position::{MapPositions, Positions};",
        "pub use window::{SkipAnyWhile, TakeAnyWhile};",
        "pub use fallible::TryStreamItem;",
        "pub use split::Either;",
        "fn len(&self) -> usize",
        "fn is_empty(&self) -> bool",
        "fn interleave<J>(self, other: J) -> Interleave<Self, J::Iter>",
        "fn interleave_shortest<J>(self, other: J) -> InterleaveShortest<Self, J::Iter>",
        "fn by_exponential_blocks(self) -> ExponentialBlocks<Self>",
        "fn by_uniform_blocks(self, block_size: usize) -> UniformBlocks<Self>",
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
        "fn flat_map_iter<F, U>",
        "fn flatten(self) -> Flatten<Self>",
        "fn flatten_iter(self) -> Flatten<Self>",
        "fn enumerate(self)",
        "fn zip<J>",
        "fn zip_eq<J>",
        "fn take(self, count: usize) -> Take<Self>",
        "fn take_any(self, count: usize) -> Take<Self>",
        "fn skip(self, count: usize) -> Skip<Self>",
        "fn skip_any(self, count: usize) -> Skip<Self>",
        "fn take_any_while<F>(self, predicate: F) -> TakeAnyWhile<Self, F>",
        "fn skip_any_while<F>(self, predicate: F) -> SkipAnyWhile<Self, F>",
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
        "fn try_reduce_with<F>(self, reduce_fn: F) -> Option<Self::Item>",
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
        "fn collect_vec_list(self) -> std::collections::LinkedList<Vec<Self::Item>>",
        "fn partition<C, F>(self, predicate: F) -> (C, C)",
        "fn partition_map<A, B, P, L, R>(self, predicate: P) -> (A, B)",
        "fn unzip<A, B, FromA, FromB>(self) -> (FromA, FromB)",
        "fn count(self)",
        "fn find_last<F>(self, predicate: F) -> Option<Self::Item>",
        "fn position_first<F>(self, predicate: F) -> Option<usize>",
        "fn position_any<F>(self, predicate: F) -> Option<usize>",
        "fn position_last<F>(self, predicate: F) -> Option<usize>",
        "fn positions<F>(self, predicate: F) -> Positions<Self, F>",
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
        "focused Rayon-style adapter subset",
        "Core parallel iterator trait for Moirai's Rayon-style non-indexed subset",
        "impl<T: Send + Sync + 'static> IndexedParallelIterator for VecParIter<T>",
        "impl IndexedParallelIterator for RangeParIter<usize>",
        "impl<'data, T: Send + Sync + 'data> IndexedParallelIterator for VecRefParIter<'data, T>",
        "impl<'a, T: Send + Sync> IndexedParallelIterator for RefVecParIter<'a, T>",
        "fn collect_into_vec(self, target: &mut Vec<Self::Item>)",
        "fn unzip_into_vecs<A, B>(self, left: &mut Vec<A>, right: &mut Vec<B>)",
        "target.clear();",
        "target.extend(self.seq_items());",
        "left.reserve_exact(expected_len);",
        "right.reserve_exact(expected_len);",
        "left.push(left_item);",
        "right.push(right_item);",
        "fn move_vec_items_into<T>(source: Vec<T>, target: &mut Vec<T>)",
        "target.extend(source);",
        "let mut left = left.into_iter();",
        "let mut right = right.into_iter();",
        "let mut output = Vec::with_capacity(len);",
        "pub struct Inspect<I, F>",
        "for item in &items",
        "inspect_fn(item);",
        "pub struct PanicFuse<I>",
        "struct PanicFusePolicy",
        "std::mem::size_of::<PanicFusePolicy>()",
        "pub struct FilterMap<I, F>",
        "pub struct MapWith<I, T, F>",
        "pub fn try_reduce_with<ReduceFn, R>(self, reduce_fn: ReduceFn) -> Option<R>",
        "fallible::try_reduce_with_items(",
        "pub fn positions<Predicate, R>(",
        "position::MapPositions::new(self.base, self.map_fn, predicate)",
        "pub struct MapInit<I, Init, F>",
        "pub struct Update<I, F>",
        "filter_map(self.filter_map_fn)",
        "pub struct WhileSome<I>",
        "map_while(|item| item)",
        "pub struct FlatMap<I, F>",
        "flat_map(self.flat_map_fn)",
        "pub struct Flatten<I>",
        "self.base.seq_items().into_iter().flatten().collect()",
        "pub struct Enumerate<I>",
        "pub struct Zip<I, J>",
        "pub struct ZipEq<I, J>",
        "zip_eq requires equal input lengths",
        "pub struct Interleave<I, J>",
        "interleave_all(self.left.seq_items(), self.right.seq_items())",
        "pub struct InterleaveShortest<I, J>",
        "interleave_shortest(self.left.seq_items(), self.right.seq_items())",
        "pub struct StepBy<I>",
        "assert!(value != 0, \"step size must be non-zero\");",
        ".step_by(self.step.get())",
        "pub struct ExponentialBlocks<I>",
        "struct ExponentialBlockPolicy",
        "std::mem::size_of::<ExponentialBlockPolicy>()",
        "pub struct UniformBlocks<I>",
        "assert!(value != 0, \"block size must be non-zero\");",
        "pub struct Positions<I, F>",
        ".filter_map(|(index, item)| (self.predicate)(item).then_some(index))",
        "pub struct MapPositions<I, MapFn, Predicate>",
        "predicate(map_fn(item)).then_some(index)",
        "pub enum Either<L, R>",
        "Either::Left(value) => left.extend(std::iter::once(value))",
        "Either::Right(value) => right.extend(std::iter::once(value))",
        "pub struct FoldConsumer<Acc, InitFn, FoldFn, CombineFn>",
        "pub struct ShortCircuitConsumer<Acc, InitFn, FoldFn, CombineFn>",
        "for FoldConsumer<Acc, InitFn, FoldFn, CombineFn>",
        "for ShortCircuitConsumer<Acc, InitFn, FoldFn, CombineFn>",
        "fn seq_try_fold<T, B, F>(self, init: T, fold_fn: F) -> std::ops::ControlFlow<B, T>",
        "fn seq_fold<T, F>(self, init: T, mut fold_fn: F) -> T",
        "pub trait TryStreamItem: private::Sealed + Send",
        "impl<T> TryStreamItem for Option<T>",
        "impl<T, E> TryStreamItem for Result<T, E>",
        "try_reduce_with_items(iterator.seq_items(), reduce_fn)",
        "ControlFlow::Break(residual) => return Some(residual)",
        "Some(Item::from_output(accumulator))",
        "pub struct Copied<I>",
        "pub struct Cloned<I>",
        "pub struct Take<I>",
        "pub struct Skip<I>",
        "pub struct TakeAnyWhile<I, F>",
        "pub struct SkipAnyWhile<I, F>",
        "retained.push(item);",
        "retained.extend(items);",
        "pub struct Chain<I, J>",
        "pub struct Intersperse<I>",
        "pub struct Rev<I>",
        "pub struct Chunks<I>",
        "struct ChunkSize(usize)",
        "assert!(value != 0, \"chunk size must be non-zero\");",
        "pub(in crate::parallel) fn into_parts(self) -> (I, usize)",
        "pub(in crate::parallel) fn into_vec(self) -> Vec<T>",
        "pub(in crate::parallel) fn into_slice(self) -> &'data [T]",
        "pub use adapters::{",
        "Inspect",
        "PanicFuse",
        "Chunks",
        "Copied",
        "Cloned",
        "Parallel range iterator for Moirai's Rayon-style non-indexed subset",
        "Moirai::for_each_indexed",
        "Moirai::map_reduce_indexed",
        "impl<T: Send + Sync + 'static> IntoParallelIterator for Vec<T>",
        "impl<'data, T: Send + Sync + 'data> IntoParallelRefIterator<'data> for Vec<T>",
        "impl IntoParallelIterator for std::ops::Range<usize>",
        "data: Vec<T>",
        // The borrowed drive splits a slice by index range and copies nothing.
        // The owned drive still splits by move, because owned elements have no
        // safe zero-copy split, but only down to the dispatch threshold.
        "struct SliceParIter<'data, T>",
        "self.data.split_at(mid)",
        "let right_data = data.split_off(mid);",
        "pub struct Reduction<T, F>",
        "let reduction: Reduction<Self::Item, F> = self.drive(ReduceConsumer::new(reduce_fn));",
        "Some(reduce_fn(left, right))",
        // A shard at or below the dispatch threshold is consumed in one
        // sequential pass; recursing to single-element shards bought no
        // parallelism and cost a consumer split and combine per element.
        "self.data.len() <= PARALLEL_DRIVE_THRESHOLD",
        "Preserve sequential value semantics for this API",
        "Segment count is not part of the semantic contract",
        "list.push_back(items)",
        "test_parallel_filter_map_retains_present_values",
        "test_parallel_collect_vec_list_moves_non_clone_values",
        "test_parallel_map_with_uses_cloned_state",
        "test_parallel_map_init_uses_initialized_state",
        "test_parallel_update_mutates_items_before_yielding",
        "test_parallel_while_some_unwraps_present_prefix",
        "test_parallel_while_some_empty_when_first_is_none",
        "test_parallel_try_for_each_returns_ok_after_processing_all_items",
        "test_parallel_try_for_each_returns_first_error",
        "test_parallel_flat_map_preserves_flattened_order",
        "test_parallel_flat_map_iter_accepts_serial_inner_iterators",
        "test_parallel_flatten_preserves_nested_order",
        "test_parallel_flatten_iter_preserves_serial_inner_order",
        "test_parallel_enumerate_pairs_logical_indices",
        "test_parallel_zip_stops_at_shorter_input",
        "test_parallel_zip_eq_preserves_equal_length_pairs",
        "test_parallel_zip_eq_rejects_length_mismatch",
        "test_indexed_interleave_moves_non_clone_values_without_clone_bound",
        "test_indexed_interleave_shortest_drops_truncated_tail_once",
        "test_indexed_step_by_moves_non_clone_values_without_clone_bound",
        "test_indexed_step_by_reports_exact_length",
        "test_indexed_step_by_rejects_zero_step",
        "test_indexed_step_by_drops_skipped_values_once",
        "test_indexed_block_adapters_preserve_values_without_clone_bound",
        "test_indexed_by_uniform_blocks_rejects_zero_size",
        "test_indexed_parallel_iterator_reports_source_lengths",
        "test_indexed_collect_into_vec_moves_non_clone_values",
        "test_indexed_unzip_into_vecs_moves_non_clone_pairs_into_existing_storage",
        "test_parallel_copied_materializes_borrowed_copy_values",
        "test_parallel_cloned_materializes_borrowed_clone_values",
        "test_non_clone_parallel_ref_iterator_maps_borrowed_values",
        "test_parallel_take_keeps_prefix",
        "test_parallel_skip_discards_prefix",
        "test_parallel_take_and_skip_saturate_at_bounds",
        "test_parallel_take_any_and_skip_any_use_bounded_window_semantics",
        "test_parallel_take_any_while_and_skip_any_while_use_deterministic_prefix_semantics",
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
        "test_parallel_partition_map_splits_either_streams",
        "test_parallel_unzip_splits_pair_streams",
        "test_parallel_reduce_empty_returns_none",
        "test_parallel_try_reduce_returns_reduced_value",
        "test_parallel_try_reduce_returns_first_error",
        "test_parallel_try_reduce_with_result_streams",
        "test_parallel_try_reduce_with_option_streams",
        "test_parallel_sum_and_product_match_standard_values",
        "test_parallel_min_and_max_match_standard_values",
        "test_parallel_min_max_by_use_comparator",
        "test_parallel_min_max_by_key_use_key_function",
        "test_parallel_find_last_returns_last_matching_value",
        "test_parallel_position_terminals_return_logical_indices",
        "test_parallel_positions_yields_all_matching_logical_indices",
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
        // The banned shape is the two-parameter placeholder introduced in
        // 8cd4286, which carried no `Consumer` implementation and was handed to
        // `drive` as scaffolding. The folding consumer that replaced it is
        // required below, and its marker pins the trait implementation rather
        // than the struct so a placeholder cannot satisfy it.
        "pub struct FoldConsumer<T, F>",
        "Rayon-compatible API",
        "matches Rayon's API",
        "Rayon compatibility",
        "Arc<Vec<T>>",
        "VecNonCloneParIter",
        "std::mem::ManuallyDrop::new",
        // The borrowed drive must never rebuild a reference vector before
        // splitting; that cost one pointer per element ahead of any work.
        "let refs: Vec<&'data T> = self.data.iter().collect();",
        // Inherent `sum` on an adapter shadows the trait terminal, because an
        // inherent method wins method resolution against a trait one. Four
        // such specializations existed to dodge the intermediate vectors the
        // trait path used to build; now that the terminal folds shards through
        // `Consumer`, a shadowing `sum` silently returns the chain to one
        // thread — `copied().map().filter().sum()` measured 35.55us against
        // the trait path's 11.18us at 131072 elements. The doc lines of the
        // four removed specializations are banned so they cannot return under
        // their original wording, and the marker below pins the terminal that
        // replaced them.
        "Sum mapped chunk outputs without materializing the chunk-output stream",
        "Sum mapped vector-backed interleaved index/value pairs without building pair streams",
        "Sum a borrowed copied-map-filter stream without materializing references",
        "Sum a flattened, mapped, filtered nested stream without intermediate vectors",
        "impl<T: Send + Sync + Clone + 'static> IntoParallelIterator for Vec<T>",
        "impl<'data, T: Send + Sync + Clone + 'static> IntoParallelRefIterator<'data> for Vec<T>",
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
        "name = \"parallel_iterator_regression\"",
        "moirai_indexed_boundary",
        "rayon_indexed_boundary",
        "moirai_collect_into_vec_pipeline",
        "rayon_collect_into_vec_pipeline",
        "moirai_unzip_into_vecs_pipeline",
        "rayon_unzip_into_vecs_pipeline",
        "moirai_indexed_pipeline",
        "rayon_indexed_pipeline",
        "moirai_filter_flat_pipeline",
        "rayon_filter_flat_pipeline",
        "moirai_flatten_pipeline",
        "rayon_flatten_pipeline",
        ".flat_map_iter(|value|",
        ".flatten_iter()",
        "moirai_take_skip_any_pipeline",
        "rayon_take_skip_any_pipeline",
        "moirai_take_skip_any_while_pipeline",
        "rayon_take_skip_any_while_pipeline",
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
        "moirai_try_reduce_with_pipeline",
        "rayon_try_reduce_with_pipeline",
        "moirai_chain_rev_pipeline",
        "rayon_chain_rev_pipeline",
        "moirai_zip_eq_pipeline",
        "rayon_zip_eq_pipeline",
        "moirai_interleave_pipeline",
        "rayon_interleave_pipeline",
        "moirai_step_by_pipeline",
        "rayon_step_by_pipeline",
        "moirai_blocks_pipeline",
        "rayon_blocks_pipeline",
        "moirai_collect_vec_list_pipeline",
        "rayon_collect_vec_list_pipeline",
        "summarize_vec_list",
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
        "moirai_positions_pipeline",
        "rayon_positions_pipeline",
        "moirai_ref_copied_cloned_pipeline",
        "rayon_ref_copied_cloned_pipeline",
        "moirai_non_clone_ref_map",
        "rayon_non_clone_ref_map",
        "moirai_unzip_pipeline",
        "rayon_unzip_pipeline",
        "moirai_partition_map_pipeline",
        "rayon_partition_map_pipeline",
        "assert_eq!(moirai_expected, rayon_expected)",
        "iterator_indexed_boundary",
        "iterator_indexed_collect_into_vec",
        "iterator_indexed_unzip_into_vecs",
        "iterator_adapter_indexed_pipeline",
        "iterator_adapter_filter_flat_pipeline",
        "iterator_adapter_flatten",
        "iterator_adapter_take_skip_any",
        "iterator_adapter_take_skip_any_while",
        "iterator_adapter_map_state",
        "iterator_adapter_update",
        "iterator_adapter_while_some",
        "iterator_adapter_try_for_each",
        "iterator_adapter_for_each_state",
        "iterator_adapter_try_for_each_state",
        "iterator_adapter_try_reduce",
        "iterator_adapter_try_reduce_with",
        "iterator_adapter_chain_rev_pipeline",
        "iterator_adapter_zip_eq",
        "iterator_indexed_interleave",
        "iterator_indexed_step_by",
        "iterator_indexed_blocks",
        "iterator_adapter_collect_vec_list",
        "iterator_adapter_intersperse",
        "iterator_adapter_inspect_chunks_pipeline",
        "iterator_adapter_partition_pipeline",
        "iterator_adapter_terminal_reducers",
        "iterator_adapter_ordered_reducers",
        "iterator_adapter_find_map",
        "iterator_adapter_position",
        "iterator_adapter_positions",
        "iterator_adapter_ref_copy_clone",
        "iterator_adapter_non_clone_ref_map",
        "iterator_adapter_unzip",
        "iterator_adapter_partition_map",
    ] {
        assert!(
            adapter_benchmark.contains(required)
                || regression_benchmark.contains(required)
                || benchmark_manifest.contains(required),
            "iterator adapter benchmark must retain comparison marker {required}"
        );
    }

    for required in [
        "parallel_iterator_map_reduce_sizes",
        "parallel_iterator_zip_filter_collect_sizes",
        "parallel_iterator_borrowed_positions_sizes",
        "parallel_iterator_borrowed_copied_reduce_sizes",
        "parallel_iterator_collect_into_existing_sizes",
        "parallel_iterator_nested_flatten_reduce_sizes",
        "parallel_iterator_chunked_map_reduce_sizes",
        "parallel_iterator_indexed_step_interleave_sizes",
        "parallel_iterator_partition_unzip_sizes",
        "parallel_iterator_position_find_sizes",
        "moirai_map_reduce",
        "rayon_map_reduce",
        "moirai_zip_filter_collect",
        "rayon_zip_filter_collect",
        "moirai_borrowed_positions",
        "rayon_borrowed_positions",
        "moirai_borrowed_copied_reduce",
        "rayon_borrowed_copied_reduce",
        "MoiraiIntoParallelRefIterator::par_iter(data)",
        "rayon::iter::IntoParallelRefIterator::par_iter(data)",
        "moirai_collect_into_existing",
        "rayon_collect_into_existing",
        "MoiraiIndexedParallelIterator::collect_into_vec",
        "RayonIndexedParallelIterator::collect_into_vec",
        "moirai_nested_flatten_reduce",
        "rayon_nested_flatten_reduce",
        "moirai_chunked_map_reduce",
        "rayon_chunked_map_reduce",
        "moirai_indexed_step_interleave",
        "rayon_indexed_step_interleave",
        "moirai_partition_unzip",
        "rayon_partition_unzip",
        "moirai_position_find",
        "rayon_position_find",
        "assert_eq!(",
        "sample_size(SAMPLE_SIZE)",
        "measurement_time(Duration::from_millis(MEASUREMENT_MILLIS))",
        "warm_up_time(Duration::from_millis(WARM_UP_MILLIS))",
        "without_plots",
    ] {
        assert!(
            regression_benchmark.contains(required),
            "parallel iterator regression benchmark must retain marker {required}"
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
        // The fork-join runs on the scheduler scope, not a crate-owned pool
        // (ADR-022), and splits only while a sub-slice is worth another lane.
        "par_merge_sort_impl(executor, self, &compare, grain)",
        "par_sort_unstable_by_impl(executor, self, &compare, grain)",
        "fn fork_join_halves",
        "executor.scope::<SyncTask, _>",
        "fn fork_grain",
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
    let async_source = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}",
        read_benchmark("../moirai-iter/src/async_iter/traits.rs"),
        read_benchmark("../moirai-iter/src/async_iter/sources.rs"),
        read_benchmark("../moirai-iter/src/async_iter/adapters.rs"),
        read_benchmark("../moirai-iter/src/async_iter/consumers.rs"),
        read_benchmark("../moirai-iter/src/async_iter/parallel.rs"),
        read_benchmark("../moirai-iter/src/async_iter/mod.rs"),
        read_benchmark("../moirai-iter/src/async_iter_tests.rs")
    );
    let async_benchmark = read_benchmark("benches/async_iterator_comparison.rs");
    let benchmark_manifest = read_benchmark("Cargo.toml");

    for required in [
        "fn into_vec(self) -> Vec<Self::Item>",
        "mod adapters;",
        "mod consumers;",
        "mod parallel;",
        "mod sources;",
        "mod traits;",
        "pub struct AsyncCollect<I, C>",
        "iter: Option<I>",
        "collection.extend(iter.into_vec())",
        "pub struct AsyncVecIter<T> {\n    items: Vec<T>,\n}",
        "pub struct AsyncRangeIter {\n    start: usize,\n    end: usize,\n}",
        "async_source_iterators_do_not_store_unused_cursors",
        "pub struct AsyncFold<I, T, F>",
        "accumulator: Option<T>",
        "async fold polled after completion",
        "pub struct AsyncReduce<I, F>",
        // The terminal returns the real accumulated value (Some(acc), or None on
        // empty) — not a placeholder default. Since the cooperative rewrite this
        // is `Poll::Ready(self.accumulator.take())`; the prohibited-marker list
        // below still guards against `Poll::Ready(C::default())`.
        "Poll::Ready(this.accumulator.take())",
        "test_async_vec_iter",
        "test_async_map",
        "pub struct AsyncTake<I>",
        "pub struct AsyncSkip<I>",
        "pub struct AsyncEnumerate<I>",
        "pub struct AsyncZip<I, J>",
        "fn take(self, count: usize) -> AsyncTake<Self>",
        "fn skip(self, count: usize) -> AsyncSkip<Self>",
        "fn enumerate(self) -> AsyncEnumerate<Self>",
        "fn zip<J>(self, other: J) -> AsyncZip<Self, J>",
        "test_async_take_skip_window_values",
        "test_async_enumerate_zip_values",
        "test_parallel_async_map",
        "test_async_filter_fold_reduce_values",
        "assert_eq!(result, vec![1, 2, 3, 4, 5])",
        "assert_eq!(result, vec![2, 4, 6, 8, 10])",
        "assert_eq!(result, vec![3, 4, 5])",
        "assert_eq!(result, vec![(0, (1, 10)), (1, (2, 20)), (2, (3, 30))])",
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
        "#![allow(dead_code)]",
        "index: usize",
        "current: usize",
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
        "moirai_take_skip_pipeline",
        "tokio_joinset_take_skip_pipeline",
        "moirai_enumerate_zip_pipeline",
        "tokio_joinset_enumerate_zip_pipeline",
        "moirai_bounded_yield_pipeline",
        "tokio_bounded_yield_pipeline",
        "BOUNDED_CONCURRENCY",
        "pending_once",
        "JoinSet::new()",
        "assert_eq!(moirai_expected, tokio_expected)",
        "async_iterator_ready_pipeline",
        "async_iterator_take_skip_pipeline",
        "async_iterator_enumerate_zip_pipeline",
        "async_iterator_bounded_yield_pipeline",
        "tokio_joinset",
    ] {
        assert!(
            async_benchmark.contains(required) || benchmark_manifest.contains(required),
            "async iterator benchmark must retain comparison marker {required}"
        );
    }
}
