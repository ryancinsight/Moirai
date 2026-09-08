//! Indexed adapters: logical positions, lengths, and bounded windows.

use super::*;

#[test]
fn test_parallel_enumerate_pairs_logical_indices() {
    let data = vec![4, 8, 15, 16];
    let result: Vec<(usize, i32)> = data.into_par_iter().enumerate().collect();
    assert_eq!(result, vec![(0, 4), (1, 8), (2, 15), (3, 16)]);
}

#[test]
fn test_parallel_zip_stops_at_shorter_input() {
    let left = vec![1, 2, 3, 4];
    let right = vec![10, 20];
    let result: Vec<(i32, i32)> = left.into_par_iter().zip(right.into_par_iter()).collect();
    assert_eq!(result, vec![(1, 10), (2, 20)]);
}

#[test]
fn test_parallel_zip_eq_preserves_equal_length_pairs() {
    let left = vec![1, 2, 3];
    let right = vec![10, 20, 30];
    let result: Vec<(i32, i32)> = left
        .into_par_iter()
        .zip_eq(right.into_par_iter())
        .map(|(left, right)| (left * 2, right + 1))
        .collect();
    assert_eq!(result, vec![(2, 11), (4, 21), (6, 31)]);
}

#[test]
#[should_panic(expected = "zip_eq requires equal input lengths")]
fn test_parallel_zip_eq_rejects_length_mismatch() {
    let left = vec![1, 2, 3];
    let right = vec![10, 20];
    let _: Vec<(i32, i32)> = left.into_par_iter().zip_eq(right.into_par_iter()).collect();
}

#[test]
fn test_indexed_interleave_moves_non_clone_values_without_clone_bound() {
    struct NonCloneValue {
        value: u64,
    }

    let left = vec![NonCloneValue { value: 1 }, NonCloneValue { value: 2 }];
    let right = vec![
        NonCloneValue { value: 10 },
        NonCloneValue { value: 20 },
        NonCloneValue { value: 30 },
    ];
    let interleaved: Vec<u64> = left
        .into_par_iter()
        .interleave(right)
        .map(|item| item.value)
        .collect();
    assert_eq!(interleaved, vec![1, 10, 2, 20, 30]);

    let longer_left = vec![
        NonCloneValue { value: 1 },
        NonCloneValue { value: 2 },
        NonCloneValue { value: 3 },
        NonCloneValue { value: 4 },
    ];
    let shorter_right = vec![NonCloneValue { value: 10 }, NonCloneValue { value: 20 }];
    let shortest_left_tail: Vec<u64> = longer_left
        .into_par_iter()
        .interleave_shortest(shorter_right)
        .map(|item| item.value)
        .collect();
    assert_eq!(shortest_left_tail, vec![1, 10, 2, 20, 3]);

    let shorter_left = vec![NonCloneValue { value: 1 }, NonCloneValue { value: 2 }];
    let longer_right = vec![
        NonCloneValue { value: 10 },
        NonCloneValue { value: 20 },
        NonCloneValue { value: 30 },
    ];
    let shortest_right_tail: Vec<u64> = shorter_left
        .into_par_iter()
        .interleave_shortest(longer_right)
        .map(|item| item.value)
        .collect();
    assert_eq!(shortest_right_tail, vec![1, 10, 2, 20]);
}

#[test]
fn test_indexed_interleave_shortest_drops_truncated_tail_once() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct DropProbe {
        value: u64,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn probes(start: u64, count: usize, drops: &Arc<AtomicUsize>) -> Vec<DropProbe> {
        (0..count)
            .map(|offset| DropProbe {
                value: start + offset as u64,
                drops: Arc::clone(drops),
            })
            .collect()
    }

    let left_drops = Arc::new(AtomicUsize::new(0));
    let right_drops = Arc::new(AtomicUsize::new(0));
    let left = probes(1, 5, &left_drops);
    let right = probes(10, 2, &right_drops);
    let values = left
        .into_par_iter()
        .interleave_shortest(right)
        .map(|item| item.value)
        .collect::<Vec<_>>();
    assert_eq!(values, vec![1, 10, 2, 11, 3]);
    assert_eq!(left_drops.load(Ordering::SeqCst), 5);
    assert_eq!(right_drops.load(Ordering::SeqCst), 2);

    let left_drops = Arc::new(AtomicUsize::new(0));
    let right_drops = Arc::new(AtomicUsize::new(0));
    let left = probes(1, 2, &left_drops);
    let right = probes(10, 5, &right_drops);
    let values = left
        .into_par_iter()
        .interleave_shortest(right)
        .map(|item| item.value)
        .collect::<Vec<_>>();
    assert_eq!(values, vec![1, 10, 2, 11]);
    assert_eq!(left_drops.load(Ordering::SeqCst), 2);
    assert_eq!(right_drops.load(Ordering::SeqCst), 5);
}

#[test]
fn test_indexed_step_by_moves_non_clone_values_without_clone_bound() {
    struct NonCloneValue {
        value: u64,
    }

    let data = vec![
        NonCloneValue { value: 1 },
        NonCloneValue { value: 2 },
        NonCloneValue { value: 3 },
        NonCloneValue { value: 4 },
        NonCloneValue { value: 5 },
        NonCloneValue { value: 6 },
    ];
    let stepped = data
        .into_par_iter()
        .step_by(2)
        .map(|item| item.value)
        .collect::<Vec<_>>();
    assert_eq!(stepped, vec![1, 3, 5]);

    let data = vec![
        NonCloneValue { value: 8 },
        NonCloneValue { value: 13 },
        NonCloneValue { value: 21 },
    ];
    let stepped = data
        .into_par_iter()
        .step_by(8)
        .map(|item| item.value)
        .collect::<Vec<_>>();
    assert_eq!(stepped, vec![8]);
}

#[test]
fn test_indexed_step_by_reports_exact_length() {
    let data = vec![1_u64, 2, 3, 4, 5, 6, 7].into_par_iter().step_by(3);
    assert_eq!(IndexedParallelIterator::len(&data), 3);

    let empty = Vec::<u64>::new().into_par_iter().step_by(3);
    assert_eq!(IndexedParallelIterator::len(&empty), 0);

    let single = vec![1_u64].into_par_iter().step_by(3);
    assert_eq!(IndexedParallelIterator::len(&single), 1);
}

#[test]
#[should_panic(expected = "step size must be non-zero")]
fn test_indexed_step_by_rejects_zero_step() {
    let _ = vec![1_u64, 2, 3].into_par_iter().step_by(0);
}

#[test]
fn test_indexed_step_by_drops_skipped_values_once() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct DropProbe {
        value: u64,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));
    let data = (0..7)
        .map(|value| DropProbe {
            value,
            drops: Arc::clone(&drops),
        })
        .collect::<Vec<_>>();
    let values = data
        .into_par_iter()
        .step_by(3)
        .map(|item| item.value)
        .collect::<Vec<_>>();
    assert_eq!(values, vec![0, 3, 6]);
    assert_eq!(drops.load(Ordering::SeqCst), 7);
}

#[test]
fn test_indexed_block_adapters_preserve_values_without_clone_bound() {
    struct NonCloneValue {
        value: u64,
    }

    let exponential = vec![
        NonCloneValue { value: 1 },
        NonCloneValue { value: 2 },
        NonCloneValue { value: 3 },
    ]
    .into_par_iter()
    .by_exponential_blocks()
    .map(|item| item.value.wrapping_mul(5))
    .collect::<Vec<_>>();
    assert_eq!(exponential, vec![5, 10, 15]);

    let uniform = vec![
        NonCloneValue { value: 8 },
        NonCloneValue { value: 13 },
        NonCloneValue { value: 21 },
        NonCloneValue { value: 34 },
    ]
    .into_par_iter()
    .by_uniform_blocks(2)
    .map(|item| item.value.wrapping_add(1))
    .collect::<Vec<_>>();
    assert_eq!(uniform, vec![9, 14, 22, 35]);
}

#[test]
#[should_panic(expected = "block size must be non-zero")]
fn test_indexed_by_uniform_blocks_rejects_zero_size() {
    let _ = vec![1_u64, 2, 3].into_par_iter().by_uniform_blocks(0);
}

#[test]
fn test_indexed_parallel_iterator_reports_source_lengths() {
    let owned = vec![1_u64, 2, 3, 4].into_par_iter();
    assert_eq!(IndexedParallelIterator::len(&owned), 4);
    assert!(!IndexedParallelIterator::is_empty(&owned));

    let empty = Vec::<u64>::new().into_par_iter();
    assert_eq!(IndexedParallelIterator::len(&empty), 0);
    assert!(IndexedParallelIterator::is_empty(&empty));

    let range = (3..11).into_par_iter();
    assert_eq!(IndexedParallelIterator::len(&range), 8);

    let borrowed_data = vec![5_u64, 8, 13];
    let borrowed = borrowed_data.par_iter();
    assert_eq!(IndexedParallelIterator::len(&borrowed), borrowed_data.len());
}

#[test]
fn test_indexed_collect_into_vec_moves_non_clone_values() {
    struct NonCloneValue {
        value: u64,
    }

    let data = vec![
        NonCloneValue { value: 8 },
        NonCloneValue { value: 13 },
        NonCloneValue { value: 21 },
    ];
    let mut output = Vec::with_capacity(8);
    output.push(NonCloneValue { value: 999 });
    let capacity = output.capacity();

    data.into_par_iter().collect_into_vec(&mut output);

    assert_eq!(output.capacity(), capacity);
    assert_eq!(
        output.iter().map(|item| item.value).collect::<Vec<_>>(),
        vec![8, 13, 21]
    );
}

#[test]
fn test_indexed_unzip_into_vecs_moves_non_clone_pairs_into_existing_storage() {
    struct NonCloneValue {
        value: u64,
    }

    let data = vec![
        (NonCloneValue { value: 1 }, NonCloneValue { value: 10 }),
        (NonCloneValue { value: 2 }, NonCloneValue { value: 20 }),
        (NonCloneValue { value: 3 }, NonCloneValue { value: 30 }),
    ];
    let mut left = Vec::with_capacity(8);
    let mut right = Vec::with_capacity(8);
    left.push(NonCloneValue { value: 999 });
    right.push(NonCloneValue { value: 888 });
    let left_capacity = left.capacity();
    let right_capacity = right.capacity();

    data.into_par_iter().unzip_into_vecs(&mut left, &mut right);

    assert_eq!(left.capacity(), left_capacity);
    assert_eq!(right.capacity(), right_capacity);
    assert_eq!(
        left.iter().map(|item| item.value).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(
        right.iter().map(|item| item.value).collect::<Vec<_>>(),
        vec![10, 20, 30]
    );
}

#[test]
fn test_parallel_take_keeps_prefix() {
    let data = vec![3, 1, 4, 1, 5];
    let result: Vec<i32> = data.into_par_iter().take(3).collect();
    assert_eq!(result, vec![3, 1, 4]);
}

#[test]
fn test_parallel_skip_discards_prefix() {
    let data = vec![3, 1, 4, 1, 5];
    let result: Vec<i32> = data.into_par_iter().skip(2).collect();
    assert_eq!(result, vec![4, 1, 5]);
}

#[test]
fn test_parallel_take_and_skip_saturate_at_bounds() {
    let taken: Vec<i32> = vec![1, 2].into_par_iter().take(8).collect();
    let skipped: Vec<i32> = vec![1, 2].into_par_iter().skip(8).collect();
    assert_eq!(taken, vec![1, 2]);
    assert_eq!(skipped, Vec::<i32>::new());
}

#[test]
fn test_parallel_take_any_and_skip_any_use_bounded_window_semantics() {
    let data = vec![3, 1, 4, 1, 5, 9];
    let result: Vec<i32> = data.into_par_iter().take_any(5).skip_any(2).collect();
    assert_eq!(result, vec![4, 1, 5]);
}

#[test]
fn test_parallel_take_any_while_and_skip_any_while_use_deterministic_prefix_semantics() {
    let data = vec![2_u64, 4, 6, 9, 12, 14];
    let taken: Vec<_> = data
        .clone()
        .into_par_iter()
        .take_any_while(|value| *value % 2 == 0)
        .collect();
    assert_eq!(taken, vec![2, 4, 6]);

    let skipped: Vec<_> = data
        .into_par_iter()
        .skip_any_while(|value| *value % 2 == 0)
        .collect();
    assert_eq!(skipped, vec![9, 12, 14]);
}

#[test]
fn test_parallel_chunks_groups_full_chunks_and_tail() {
    let data = vec![1, 2, 3, 4, 5];
    let result: Vec<Vec<i32>> = data.into_par_iter().chunks(2).collect();
    assert_eq!(result, vec![vec![1, 2], vec![3, 4], vec![5]]);
}

#[test]
#[should_panic(expected = "chunk size must be non-zero")]
fn test_parallel_chunks_rejects_zero_size() {
    let data = vec![1, 2, 3];
    let _: Vec<Vec<i32>> = data.into_par_iter().chunks(0).collect();
}

#[test]
fn test_parallel_chain_preserves_left_then_right_order() {
    let left = vec![1, 2, 3];
    let right = vec![4, 5];
    let result: Vec<i32> = left.into_par_iter().chain(right.into_par_iter()).collect();
    assert_eq!(result, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_parallel_intersperse_inserts_separator_between_items() {
    let data = vec![1, 2, 3];
    let result: Vec<i32> = data.into_par_iter().intersperse(0).collect();
    assert_eq!(result, vec![1, 0, 2, 0, 3]);
}

#[test]
fn test_parallel_intersperse_preserves_empty_and_singleton_streams() {
    let empty: Vec<i32> = Vec::<i32>::new().into_par_iter().intersperse(0).collect();
    let singleton: Vec<i32> = vec![7].into_par_iter().intersperse(0).collect();
    assert_eq!(empty, Vec::<i32>::new());
    assert_eq!(singleton, vec![7]);
}

#[test]
fn test_parallel_rev_reverses_logical_order() {
    let data = vec![1, 2, 3, 4, 5];
    let result: Vec<i32> = data.into_par_iter().rev().collect();
    assert_eq!(result, vec![5, 4, 3, 2, 1]);
}

#[test]
fn test_parallel_unzip_splits_pair_streams() {
    let data = vec![1_u64, 2, 3, 4];
    let (left, right): (Vec<u64>, Vec<u64>) = data
        .into_par_iter()
        .map(|value| (value, value.wrapping_mul(10)))
        .unzip();
    assert_eq!(left, vec![1, 2, 3, 4]);
    assert_eq!(right, vec![10, 20, 30, 40]);
}

#[test]
fn test_parallel_position_terminals_return_logical_indices() {
    let data = vec![1_u64, 4, 7, 10, 13, 16];
    assert_eq!(
        data.clone()
            .into_par_iter()
            .position_first(|value| value % 6 == 4),
        Some(1)
    );
    assert_eq!(
        data.clone()
            .into_par_iter()
            .position_any(|value| value == 10),
        Some(3)
    );
    assert_eq!(
        data.clone()
            .into_par_iter()
            .position_last(|value| value % 6 == 4),
        Some(5)
    );
    assert_eq!(
        data.into_par_iter().position_first(|value| value > 100),
        None
    );
}

#[test]
fn test_parallel_positions_yields_all_matching_logical_indices() {
    let data = vec![2_u64, 3, 5, 8, 11, 14, 17, 20];
    let positions: Vec<usize> = data
        .clone()
        .into_par_iter()
        .positions(|value| value % 3 == 2)
        .collect();
    assert_eq!(positions, vec![0, 2, 3, 4, 5, 6, 7]);

    let borrowed_positions: Vec<usize> =
        data.par_iter().positions(|value| *value % 4 == 0).collect();
    assert_eq!(borrowed_positions, vec![3, 7]);

    let mapped_positions: Vec<usize> = data
        .into_par_iter()
        .map(|value| value + 1)
        .positions(|value| value % 5 == 0)
        .collect();
    assert_eq!(mapped_positions, vec![5]);
}
