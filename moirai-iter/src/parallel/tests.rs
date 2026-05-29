use super::*;

#[test]
fn test_parallel_map() {
    let data = vec![1, 2, 3, 4, 5];
    let result: Vec<i32> = data.into_par_iter().map(|x| x * 2).collect();
    assert_eq!(result, vec![2, 4, 6, 8, 10]);
}

#[test]
fn test_parallel_map_with_uses_cloned_state() {
    let data = vec![1_u64, 2, 3, 4];
    let result: Vec<u64> = data
        .into_par_iter()
        .map_with(10_u64, |state, value| {
            *state = state.wrapping_add(1);
            value.wrapping_mul(*state)
        })
        .collect();

    assert_eq!(result, vec![11, 24, 39, 56]);
}

#[test]
fn test_parallel_map_init_uses_initialized_state() {
    let data = vec![2_u64, 4, 6];
    let result: Vec<u64> = data
        .into_par_iter()
        .map_init(
            || 3_u64,
            |state, value| {
                let output = value.wrapping_add(*state);
                *state = state.wrapping_add(2);
                output
            },
        )
        .collect();

    assert_eq!(result, vec![5, 9, 13]);
}

#[test]
fn test_parallel_update_mutates_items_before_yielding() {
    let data = vec![1_u64, 2, 3, 4];
    let result: Vec<u64> = data
        .into_par_iter()
        .update(|value| {
            *value = value.wrapping_mul(3).wrapping_add(1);
        })
        .collect();

    assert_eq!(result, vec![4, 7, 10, 13]);
}

#[test]
fn test_parallel_filter() {
    let data = vec![1, 2, 3, 4, 5, 6];
    let result: Vec<i32> = data.into_par_iter().filter(|&x| x % 2 == 0).collect();
    assert_eq!(result, vec![2, 4, 6]);
}

#[test]
fn test_parallel_inspect_observes_items_without_changing_output() {
    let data = vec![1, 2, 3, 4];
    let observed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&observed);

    let result: Vec<i32> = data
        .clone()
        .into_par_iter()
        .inspect(move |value| sink.lock().expect("inspection lock").push(*value))
        .collect();

    assert_eq!(result, data);
    assert_eq!(*observed.lock().expect("inspection lock"), vec![1, 2, 3, 4]);
}

#[test]
fn test_parallel_panic_fuse_preserves_values() {
    let data = vec![1, 2, 3];
    let result: Vec<i32> = data
        .into_par_iter()
        .panic_fuse()
        .map(|value| value * 2)
        .collect();
    assert_eq!(result, vec![2, 4, 6]);
}

#[test]
#[should_panic(expected = "panic-fuse propagation")]
fn test_parallel_panic_fuse_propagates_panic() {
    let data = vec![1, 2, 3];
    let _: Vec<i32> = data
        .into_par_iter()
        .panic_fuse()
        .map(|value| {
            if value == 2 {
                panic!("panic-fuse propagation");
            }
            value
        })
        .collect();
}

#[test]
fn test_parallel_filter_map_retains_present_values() {
    let data = vec![1, 2, 3, 4, 5, 6];
    let result: Vec<i32> = data
        .into_par_iter()
        .filter_map(|value| (value % 2 == 0).then_some(value * 10))
        .collect();
    assert_eq!(result, vec![20, 40, 60]);
}

#[test]
fn test_parallel_while_some_unwraps_present_prefix() {
    let data = vec![Some(1_u64), Some(2), Some(3), None, Some(5)];
    let result: Vec<_> = data.into_par_iter().while_some().collect();
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_parallel_while_some_empty_when_first_is_none() {
    let data = vec![None, Some(2_u64), Some(3)];
    let result: Vec<_> = data.into_par_iter().while_some().collect();
    assert!(result.is_empty());
}

#[test]
fn test_parallel_flat_map_preserves_flattened_order() {
    let data = vec![1, 2, 3];
    let result: Vec<i32> = data.into_par_iter().flat_map(|value| 0..value).collect();
    assert_eq!(result, vec![0, 0, 1, 0, 1, 2]);
}

#[test]
fn test_parallel_flatten_preserves_nested_order() {
    let data = vec![vec![1, 2], Vec::new(), vec![3, 4, 5]];
    let result: Vec<i32> = data.into_par_iter().flatten().collect();
    assert_eq!(result, vec![1, 2, 3, 4, 5]);
}

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
fn test_parallel_copied_materializes_borrowed_copy_values() {
    let data = vec![1_u64, 2, 3, 4];
    let result: Vec<u64> = data.par_iter().copied().map(|value| value * 3).collect();
    assert_eq!(result, vec![3, 6, 9, 12]);
}

#[test]
fn test_parallel_cloned_materializes_borrowed_clone_values() {
    let data = vec!["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()];
    let result: Vec<String> = data
        .par_iter()
        .cloned()
        .filter(|value| value.contains('a'))
        .collect();
    assert_eq!(
        result,
        vec!["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()]
    );
}

#[test]
fn test_non_clone_parallel_ref_iterator_maps_borrowed_values() {
    struct NonCloneBorrowed {
        value: u64,
    }

    let data = vec![
        NonCloneBorrowed { value: 2 },
        NonCloneBorrowed { value: 3 },
        NonCloneBorrowed { value: 5 },
    ];

    let result = data
        .par_iter()
        .map(|item| item.value.wrapping_mul(7))
        .collect::<Vec<_>>();

    assert_eq!(result, vec![14, 21, 35]);
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
fn test_parallel_reduce() {
    let data = vec![1, 2, 3, 4, 5];
    let result = data.into_par_iter().reduce(|a, b| a + b);
    assert_eq!(result, Some(15));
}

#[test]
fn test_parallel_reduce_with_combines_split_halves() {
    let data = (1..=128).collect::<Vec<i32>>();
    let expected = data.iter().copied().sum::<i32>();
    let result = data.into_par_iter().reduce_with(|a, b| a + b);
    assert_eq!(result, Some(expected));
}

#[test]
fn test_parallel_try_reduce_returns_reduced_value() {
    let data = vec![Ok::<u64, u64>(1), Ok(2), Ok(3), Ok(4)];
    let result = data
        .into_par_iter()
        .try_reduce(|| 0_u64, |left, right| Ok::<u64, u64>(left + right));
    assert_eq!(result, Ok(10));
}

#[test]
fn test_parallel_try_reduce_returns_first_error() {
    let data = vec![Ok::<u64, u64>(1), Ok(2), Err(3), Ok(4)];
    let result = data
        .into_par_iter()
        .try_reduce(|| 0_u64, |left, right| Ok::<u64, u64>(left + right));
    assert_eq!(result, Err(3));
}

#[test]
fn test_parallel_reduce_empty_returns_none() {
    let data = Vec::<i32>::new();
    let result = data.into_par_iter().reduce(|a, b| a + b);
    assert_eq!(result, None);
}

#[test]
fn test_parallel_sum_and_product_match_standard_values() {
    let data = vec![1_u64, 2, 3, 4, 5];
    let sum = data.clone().into_par_iter().sum::<u64>();
    let product = data.into_par_iter().product::<u64>();
    assert_eq!(sum, 15);
    assert_eq!(product, 120);

    let empty_sum = Vec::<u64>::new().into_par_iter().sum::<u64>();
    let empty_product = Vec::<u64>::new().into_par_iter().product::<u64>();
    assert_eq!(empty_sum, 0);
    assert_eq!(empty_product, 1);
}

#[test]
fn test_parallel_min_and_max_match_standard_values() {
    let data = vec![8, 3, 13, 5, 2, 21];
    assert_eq!(data.clone().into_par_iter().min(), Some(2));
    assert_eq!(data.into_par_iter().max(), Some(21));

    let empty = Vec::<i32>::new();
    assert_eq!(empty.clone().into_par_iter().min(), None);
    assert_eq!(empty.into_par_iter().max(), None);
}

#[test]
fn test_parallel_min_max_by_use_comparator() {
    let data = vec![(8_u64, 40_u64), (3, 90), (13, 10), (5, 70)];
    assert_eq!(
        data.clone()
            .into_par_iter()
            .min_by(|left, right| left.1.cmp(&right.1)),
        Some((13, 10))
    );
    assert_eq!(
        data.into_par_iter()
            .max_by(|left, right| left.1.cmp(&right.1)),
        Some((3, 90))
    );
}

#[test]
fn test_parallel_min_max_by_key_use_key_function() {
    let data = vec![(8_u64, 40_u64), (3, 90), (13, 10), (5, 70)];
    assert_eq!(
        data.clone()
            .into_par_iter()
            .min_by_key(|(left, right)| left ^ right),
        Some((13, 10))
    );
    assert_eq!(
        data.into_par_iter()
            .max_by_key(|(left, right)| left ^ right),
        Some((3, 90))
    );

    let empty = Vec::<(u64, u64)>::new();
    assert_eq!(
        empty
            .clone()
            .into_par_iter()
            .min_by_key(|(left, right)| left ^ right),
        None
    );
    assert_eq!(
        empty
            .into_par_iter()
            .max_by_key(|(left, right)| left ^ right),
        None
    );
}

#[test]
fn test_parallel_fold_preserves_sequential_value_semantics() {
    let data = vec![1, 2, 3, 4, 5];
    let result = data.into_par_iter().fold(10, |acc, item| acc - item);
    assert_eq!(result, -5);
}

#[test]
fn test_parallel_partition_preserves_relative_order() {
    let data = vec![1, 2, 3, 4, 5, 6];
    let (even, odd): (Vec<i32>, Vec<i32>) = data.into_par_iter().partition(|value| value % 2 == 0);
    assert_eq!(even, vec![2, 4, 6]);
    assert_eq!(odd, vec![1, 3, 5]);
}

#[test]
fn test_parallel_partition_map_splits_either_streams() {
    let data = vec![1_u64, 2, 3, 4, 5, 6];
    let (multiples, residuals): (Vec<u64>, Vec<u64>) =
        data.into_par_iter().partition_map(|value| {
            if value % 3 == 0 {
                Either::Left(value.wrapping_mul(10))
            } else {
                Either::Right(value.wrapping_add(100))
            }
        });

    assert_eq!(multiples, vec![30, 60]);
    assert_eq!(residuals, vec![101, 102, 104, 105]);
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
fn test_range_parallel() {
    let result: Vec<usize> = (0..10).into_par_iter().map(|x| x * x).collect();
    let expected: Vec<usize> = (0..10).map(|x| x * x).collect();
    assert_eq!(result, expected);
}

#[test]
fn test_parallel_count() {
    let data = vec![1, 2, 3, 4, 5];
    let count = data.into_par_iter().count();
    assert_eq!(count, 5);
}

#[test]
fn test_parallel_any() {
    let data = vec![1, 2, 3, 4, 5];
    assert!(data.clone().into_par_iter().any(|x| *x == 3));
    assert!(!data.into_par_iter().any(|x| *x == 10));
}

#[test]
fn test_parallel_try_for_each_returns_ok_after_processing_all_items() {
    let data = vec![1_u64, 2, 3, 4];
    let total = std::sync::atomic::AtomicU64::new(0);
    let result = data.into_par_iter().try_for_each(|value| {
        total.fetch_add(value, std::sync::atomic::Ordering::Relaxed);
        Ok::<(), u64>(())
    });

    assert_eq!(result, Ok(()));
    assert_eq!(total.load(std::sync::atomic::Ordering::Relaxed), 10);
}

#[test]
fn test_parallel_try_for_each_returns_first_error() {
    let data = vec![1_u64, 2, 3, 4];
    let result = data
        .into_par_iter()
        .try_for_each(|value| if value == 3 { Err(value) } else { Ok(()) });

    assert_eq!(result, Err(3));
}

#[test]
fn test_parallel_find_last_returns_last_matching_value() {
    let data = vec![1_u64, 4, 7, 10, 13, 16];
    let result = data
        .clone()
        .into_par_iter()
        .find_last(|value| value % 3 == 1);
    assert_eq!(result, Some(16));

    let missing = data.into_par_iter().find_last(|value| *value > 100);
    assert_eq!(missing, None);
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
fn test_parallel_find_map_first_maps_first_present_value() {
    let data = vec![1_u64, 4, 7, 10, 13];
    let result = data
        .clone()
        .into_par_iter()
        .find_map_first(|value| (value % 5 == 0).then_some(value.wrapping_mul(11)));
    assert_eq!(result, Some(110));

    let missing = data
        .into_par_iter()
        .find_map_first(|value| (value > 100).then_some(value));
    assert_eq!(missing, None);
}

#[test]
fn test_parallel_find_map_any_maps_present_value() {
    let data = vec![1_u64, 4, 7, 10, 13];
    let result = data
        .into_par_iter()
        .find_map_any(|value| (value == 7).then_some(value.wrapping_mul(13)));
    assert_eq!(result, Some(91));
}

#[test]
fn test_parallel_find_map_last_maps_last_present_value() {
    let data = vec![1_u64, 4, 7, 10, 13, 16];
    let result = data
        .clone()
        .into_par_iter()
        .find_map_last(|value| (value % 3 == 1).then_some(value.wrapping_mul(17)));
    assert_eq!(result, Some(272));

    let missing = data
        .into_par_iter()
        .find_map_last(|value| (value > 100).then_some(value));
    assert_eq!(missing, None);
}

#[test]
fn test_parallel_for_each_with_uses_cloned_state() {
    let data = vec![1_u64, 2, 3, 4];
    let checksum = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

    data.into_par_iter()
        .map(|value| value.wrapping_mul(3))
        .for_each_with(std::sync::Arc::clone(&checksum), |state, value| {
            state.fetch_add(value, std::sync::atomic::Ordering::Relaxed);
        });

    assert_eq!(
        checksum.load(std::sync::atomic::Ordering::Relaxed),
        (1_u64 + 2 + 3 + 4) * 3
    );
}

#[test]
fn test_parallel_for_each_init_uses_initialized_state() {
    let data = vec![2_u64, 4, 6, 8];
    let checksum = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let sink = std::sync::Arc::clone(&checksum);

    data.into_par_iter()
        .map(|value| value.wrapping_add(1))
        .for_each_init(
            || std::sync::Arc::clone(&sink),
            |state, value| {
                state.fetch_add(value, std::sync::atomic::Ordering::Relaxed);
            },
        );

    assert_eq!(
        checksum.load(std::sync::atomic::Ordering::Relaxed),
        (2_u64 + 4 + 6 + 8) + 4
    );
}

#[test]
fn test_parallel_try_for_each_with_uses_cloned_state_and_propagates_error() {
    let data = vec![1_u64, 2, 3, 4];
    let checksum = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

    let result = data.clone().into_par_iter().try_for_each_with(
        std::sync::Arc::clone(&checksum),
        |state, value| {
            state.fetch_add(value.wrapping_mul(5), std::sync::atomic::Ordering::Relaxed);
            Ok::<(), u64>(())
        },
    );
    assert_eq!(result, Ok(()));
    assert_eq!(
        checksum.load(std::sync::atomic::Ordering::Relaxed),
        (1_u64 + 2 + 3 + 4) * 5
    );

    let error =
        data.into_par_iter().try_for_each_with(
            (),
            |_state, value| {
                if value == 3 {
                    Err(value)
                } else {
                    Ok(())
                }
            },
        );
    assert_eq!(error, Err(3));
}

#[test]
fn test_parallel_try_for_each_init_uses_initialized_state_and_propagates_error() {
    let data = vec![2_u64, 4, 6, 8];
    let checksum = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let sink = std::sync::Arc::clone(&checksum);

    let result = data.clone().into_par_iter().try_for_each_init(
        || std::sync::Arc::clone(&sink),
        |state, value| {
            state.fetch_add(value.wrapping_add(7), std::sync::atomic::Ordering::Relaxed);
            Ok::<(), u64>(())
        },
    );
    assert_eq!(result, Ok(()));
    assert_eq!(
        checksum.load(std::sync::atomic::Ordering::Relaxed),
        (2_u64 + 4 + 6 + 8) + (4 * 7)
    );

    let error = data.into_par_iter().try_for_each_init(
        || (),
        |_state, value| {
            if value == 6 {
                Err(value)
            } else {
                Ok(())
            }
        },
    );
    assert_eq!(error, Err(6));
}

#[test]
fn test_parallel_all() {
    let data = vec![2, 4, 6, 8];
    assert!(data.clone().into_par_iter().all(|x| *x % 2 == 0));
    assert!(!data.into_par_iter().all(|x| *x > 5));
}
