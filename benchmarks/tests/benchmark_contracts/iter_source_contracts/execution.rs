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
        "iter_ops_parallel_map_output",
        "DIRECT_OUTPUT_WORK_ITEMS",
        "BatchSize::LargeInput",
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
    let output = read_benchmark("../moirai-iter/src/parallel/output.rs");
    let benchmark = read_benchmark("benches/cache_iterator_comparison.rs");
    let manifest = read_benchmark("Cargo.toml");

    for required in [
        "pub struct ZeroCopyParallelIter<'a, T>",
        "data: &'a [T]",
        "fn should_execute_scoped_cache<T>(len: usize, chunk_size: usize) -> bool",
        "DEFAULT_RING_BUFFER_CAPACITY",
        "should_execute_scoped_cache::<T>(self.data.len(), self.chunk_size)",
        "return self.data.iter().map(&func).collect();",
        "let chunk_start = chunk_index * self.chunk_size",
        "ChunkWriter::new(output_ptr.as_ptr().cast(), chunk_start..chunk_end)",
        "for item in chunk",
        "writer.push(func_ref(item))",
        ".write(writer.finish())",
        "func_ref(item)",
        "zero_copy_map_borrows_data_and_closure",
        "zero_copy_map_matches_sequential_values",
        "fn reduce_owned_pairs<T, F>(items: Vec<T>, func: &F) -> Vec<T>",
        "current_results = reduce_owned_pairs(current_results, &func);",
        "reduce_owned_pairs_moves_non_clone_odd_value",
        "scoped_execution_gate_uses_batch_capacity_floor",
        "zero_copy_reduce_accepts_non_clone_reducer",
    ] {
        assert!(
            source.contains(required),
            "cache zero-copy parallel iterator must retain borrowed scoped map marker {required}"
        );
    }

    for required in [
        "pub(crate) struct MapOutput<T>",
        "pub(crate) struct ChunkWriter<T>",
        "impl<T> Drop for ChunkWriter<T>",
        "ptr::drop_in_place",
        "Vec::from_raw_parts",
        "unfinished_writer_drops_only_its_initialized_prefix",
        "zero_sized_outputs_retain_their_logical_length",
    ] {
        assert!(
            output.contains(required),
            "shared parallel map output must retain ownership marker {required}"
        );
    }

    for required in [
        "name = \"cache_iterator_comparison\"",
        "cache_iterator_zero_copy_map",
        "cache_iterator_zero_copy_large_map",
        "MAP_FAN_OUT_WORK_ITEMS",
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
        "ExecutionContext::Parallel(_) => crate::base::process_parallelism()",
        ".map(func)",
        ".for_each(|()| async {})",
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
        "execution_context_parallel_async_map",
        "moirai_parallel_context_async_map",
        "execution_context_parallel_pending_async_map",
        "moirai_parallel_context_pending_async_map",
        "execution_context_sparse_pending_map",
        "moirai_sparse_pending_map",
        "incumbent_sparse_pending_map",
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
        "let func = Arc::new(func)",
        "let predicate = Arc::new(predicate)",
    ] {
        assert!(
            !source.contains(prohibited),
            "execution context iterator must not reintroduce clone-bound owned chunking through {prohibited}"
        );
    }
}
