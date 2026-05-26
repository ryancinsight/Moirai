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
