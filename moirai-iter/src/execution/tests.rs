//! Unit tests for execution contexts.

use super::async_ctx::AsyncContext;
use super::base::ExecutionContext;
use super::hybrid::owned_chunks;
use super::parallel::ParallelContext;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, PartialEq)]
struct NonClone(u64);

#[test]
fn owned_chunks_move_values_without_clone_bound() {
    let chunks = owned_chunks((0..5).map(NonClone).collect(), 2);
    let values = chunks
        .into_iter()
        .flatten()
        .map(|item| item.0)
        .collect::<Vec<_>>();

    assert_eq!(values, vec![0, 1, 2, 3, 4]);
}

#[test]
fn non_clone_parallel_context_execute_iter_consumes_items() {
    let data = (0..6).map(NonClone).collect::<Vec<_>>();

    let mapped = ParallelContext::with_chunk_size(2)
        .execute_iter(data, |item| item.0.wrapping_mul(3))
        .expect("parallel context map should consume non-clone items");

    assert_eq!(mapped, vec![0, 3, 6, 9, 12, 15]);
}

#[test]
fn non_clone_async_context_execute_iter_consumes_items() {
    let data = (0..4).map(NonClone).collect::<Vec<_>>();

    let mapped = AsyncContext::with_batch_size(2)
        .execute_iter(data, |item| item.0.wrapping_add(1))
        .expect("async context map should consume non-clone items");

    assert_eq!(mapped, vec![1, 2, 3, 4]);
}

#[tokio::test]
async fn async_context_map_runs_bounded_concurrently_and_preserves_order() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let context = ExecutionContext::Async(AsyncContext::new().with_max_concurrent(4));
    let active_count = Arc::new(AtomicUsize::new(0));
    let max_observed = Arc::new(AtomicUsize::new(0));

    let active_count_clone = active_count.clone();
    let max_observed_clone = max_observed.clone();

    let mapped = context
        .execute_async_iter((0..8).collect::<Vec<_>>(), move |value| {
            let active_count = active_count_clone.clone();
            let max_observed = max_observed_clone.clone();
            async move {
                let active = active_count.fetch_add(1, Ordering::SeqCst) + 1;
                let mut current = max_observed.load(Ordering::SeqCst);
                while active > current {
                    match max_observed.compare_exchange_weak(
                        current,
                        active,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => break,
                        Err(actual) => current = actual,
                    }
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
                active_count.fetch_sub(1, Ordering::SeqCst);
                value * 2
            }
        })
        .await
        .expect("async map should complete");

    assert_eq!(mapped, vec![0, 2, 4, 6, 8, 10, 12, 14]);
    assert!(
        max_observed.load(Ordering::SeqCst) > 1,
        "bounded concurrent map should execute tasks concurrently"
    );
}

#[tokio::test]
async fn async_context_filter_runs_bounded_concurrently_and_preserves_order() {
    let context = ExecutionContext::Async(AsyncContext::new().with_max_concurrent(4));

    let filtered = context
        .execute_async_filter((0..8).collect::<Vec<_>>(), |value| {
            let value = *value;
            async move {
                tokio::time::sleep(Duration::from_millis(10)).await;
                value % 2 == 0
            }
        })
        .await
        .expect("async filter should complete");

    assert_eq!(filtered, vec![0, 2, 4, 6]);
}
