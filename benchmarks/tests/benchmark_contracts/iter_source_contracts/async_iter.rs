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
        "use crate::stream::{retained_buffered, retained_unordered};",
        "let concurrency = self.concurrency.max(1);",
        "retained_buffered(futures, concurrency)",
        "retained_unordered(stream::iter(items).map(func), concurrency)",
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
        ".buffered(concurrency)",
        ".buffer_unordered(concurrency)",
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
