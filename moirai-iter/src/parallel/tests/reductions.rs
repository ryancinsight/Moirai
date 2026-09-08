//! Reducing terminals: folds, extrema, arithmetic, and stream splitting.

use super::*;

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
fn test_parallel_try_reduce_with_result_streams() {
    let data = vec![Ok::<u64, u64>(1), Ok(2), Ok(3), Ok(4)];
    let reduced = data
        .into_par_iter()
        .try_reduce_with(|left, right| Ok::<u64, u64>(left + right));
    assert_eq!(reduced, Some(Ok(10)));

    let error = vec![Ok::<u64, u64>(1), Ok(2), Err(7), Ok(4)]
        .into_par_iter()
        .try_reduce_with(|left, right| Ok::<u64, u64>(left + right));
    assert_eq!(error, Some(Err(7)));

    let empty = Vec::<Result<u64, u64>>::new()
        .into_par_iter()
        .try_reduce_with(|left, right| Ok::<u64, u64>(left + right));
    assert_eq!(empty, None);
}

#[test]
fn test_parallel_try_reduce_with_option_streams() {
    let reduced = vec![Some(2_u64), Some(4), Some(6)]
        .into_par_iter()
        .try_reduce_with(|left, right| Some(left + right));
    assert_eq!(reduced, Some(Some(12)));

    let stopped = vec![Some(2_u64), None, Some(6)]
        .into_par_iter()
        .try_reduce_with(|left, right| Some(left + right));
    assert_eq!(stopped, Some(None));
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
