use super::*;

#[test]
fn zero_copy_iter_yields_borrowed_items() {
    let data = vec![1, 2, 3, 4, 5];
    let collected: Vec<_> = ZeroCopyIter::new(&data).collect();

    assert_eq!(collected, vec![&1, &2, &3, &4, &5]);
}

#[test]
fn chunked_iter_keeps_trailing_partial_chunk() {
    let chunks: Vec<_> = vec![1, 2, 3, 4, 5, 6, 7].into_iter().chunked(3).collect();

    assert_eq!(chunks, vec![vec![1, 2, 3], vec![4, 5, 6], vec![7]]);
}

#[test]
fn fused_map_filter_preserves_value_semantics() {
    let result: Vec<_> = vec![1, 2, 3, 4, 5]
        .into_iter()
        .map_filter(|x| x * 2, |x| x > &5)
        .collect();

    assert_eq!(result, vec![6, 8, 10]);
}

#[test]
fn window_iter_emits_overlapping_slices() {
    let data = vec![1, 2, 3, 4, 5];
    let windows: Vec<_> = WindowIter::new(&data, 3).collect();

    assert_eq!(
        windows,
        vec![&[1, 2, 3][..], &[2, 3, 4][..], &[3, 4, 5][..]]
    );
}

#[test]
fn interleave_alternates_until_inputs_are_empty() {
    let result: Vec<_> = vec![1, 3, 5]
        .into_iter()
        .interleave(vec![2, 4, 6])
        .collect();

    assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn batch_iter_emits_transformed_batches() {
    let result: Vec<_> = vec![1, 2, 3, 4, 5, 6, 7]
        .into_iter()
        .batch(3, |batch| batch.iter().sum::<i32>())
        .collect();

    assert_eq!(result, vec![6, 15, 7]);
}

#[test]
fn streaming_iter_preserves_fifo_order() {
    let mut next = 0usize;
    let values: Vec<_> = StreamingIter::new(2, move || {
        next += 1;
        (next <= 4).then_some(next)
    })
    .collect();

    assert_eq!(values, vec![1, 2, 3, 4]);
}

#[test]
fn parallel_iter_map_borrows_data_without_static_closure() {
    let factor = 3_i32;

    let values = ParallelIter::new(vec![1, 2, 3, 4, 5]).map(|value| *value * factor);

    assert_eq!(values, vec![3, 6, 9, 12, 15]);
}

#[test]
fn parallel_iter_reduce_matches_sequential_sum() {
    let data = (1_u64..=1024).collect::<Vec<_>>();
    let expected = data.iter().copied().sum::<u64>();

    let reduced = ParallelIter::new(data).reduce(0_u64, |accumulator, value| accumulator + *value);

    assert_eq!(reduced, expected);
}

#[test]
fn parallel_iter_reduce_empty_returns_identity() {
    let reduced = ParallelIter::<u64>::new(Vec::new())
        .reduce(17_u64, |accumulator, value| accumulator + *value);

    assert_eq!(reduced, 17);
}

#[test]
fn scan_ref_threads_state_and_borrows_a_non_static_local() {
    // `scan_ref` accepts a non-`'static` `FnMut`, so the closure can both thread
    // the scan state (`state`) and borrow a stack local (`running`) by mutable
    // reference — the property `scan` cannot offer.
    let data = [1, 2, 3, 4, 5];
    let mut running = 0;
    let prefix_sums: Vec<i32> = data
        .iter()
        .scan_ref(0, |state, &value| {
            *state += value;
            running = *state;
            Some(*state)
        })
        .collect();

    assert_eq!(prefix_sums, vec![1, 3, 6, 10, 15]);
    assert_eq!(running, 15);
}

#[test]
fn partition_ref_splits_by_predicate_into_two_collections() {
    let (evens, odds): (Vec<i32>, Vec<i32>) =
        (1..=10).partition_ref(|value| value % 2 == 0).partition();

    assert_eq!(evens, vec![2, 4, 6, 8, 10]);
    assert_eq!(odds, vec![1, 3, 5, 7, 9]);
}
