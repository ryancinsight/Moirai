use super::*;

#[test]
fn test_tree_reduce() {
    let items = vec![1, 2, 3, 4, 5];
    let result = tree_reduce(items, |a, b| a + b);
    assert_eq!(result, Some(15));

    let empty: Vec<i32> = vec![];
    let result = tree_reduce(empty, |a, b| a + b);
    assert_eq!(result, None);
}

#[test]
fn test_process_in_batches() {
    let items = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let result = process_in_batches(items, 3, |chunk| vec![chunk.iter().sum::<i32>()]);
    assert_eq!(result, vec![6, 15, 15]);
}

#[test]
fn base_adapters_expose_components_without_dead_fields() {
    let base = BaseIterator::new(vec![1_u64, 2, 3], "context");
    assert_eq!(base.inner(), &vec![1, 2, 3]);
    assert_eq!(**base.context(), "context");
    let (inner, context) = base.into_parts();
    assert_eq!(inner, vec![1, 2, 3]);
    assert_eq!(*context, "context");

    let map = MapAdapter::<_, _, u64, u64>::new(vec![1_u64, 2, 3], |value| value + 1);
    assert_eq!(map.inner(), &vec![1, 2, 3]);
    assert_eq!((map.function())(4), 5);
    let (inner, map_fn) = map.into_parts();
    assert_eq!(inner, vec![1, 2, 3]);
    assert_eq!(map_fn(5), 6);

    let filter = FilterAdapter::<_, _, u64>::new(vec![1_u64, 2, 3], |value: &u64| *value > 1);
    assert_eq!(filter.inner(), &vec![1, 2, 3]);
    assert!(filter.predicate()(&2));
    let (inner, predicate) = filter.into_parts();
    assert_eq!(inner, vec![1, 2, 3]);
    assert!(predicate(&3));

    let batch = BatchAdapter::new(vec![1_u64, 2, 3], 0);
    assert_eq!(batch.inner(), &vec![1, 2, 3]);
    assert_eq!(batch.size(), 1);
    let (inner, size) = batch.into_parts();
    assert_eq!(inner, vec![1, 2, 3]);
    assert_eq!(size, 1);
}

#[test]
fn sequential_fallback_only_on_pre_execution_shutdown() {
    use moirai_core::error::ExecutorError;
    assert!(!sequential_fallback_permitted(&Ok(())));
    assert!(sequential_fallback_permitted(&Err(
        ExecutorError::ShuttingDown
    )));
}

#[test]
#[should_panic(expected = "partial execution")]
fn sequential_fallback_rejects_partial_execution_errors() {
    use moirai_core::error::ExecutorError;
    let _ = sequential_fallback_permitted(&Err(ExecutorError::SpawnFailed(
        moirai_core::error::TaskError::Panicked,
    )));
}

#[test]
fn test_tree_reduce_parallel() {
    let items: Vec<i32> = (1..=1000).collect();
    let result = tree_reduce(items, |a, b| a + b);
    assert_eq!(result, Some(500500));
}
