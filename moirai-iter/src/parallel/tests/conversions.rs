//! Source adapters and the sequential conversions.

use super::*;

#[test]
fn nested_iteration_produces_correct_values() {
    // Regression guard for nested parallel iteration (an inner drive inside an
    // outer map). Both drives cross the scheduler-backed threshold, so this
    // test protects exact-once traversal and ordering while recursive branches
    // overlap on worker lanes. The former heap-corruption report is retained in
    // docs/concurrency_audit.md as the pre-ISSUE-208 failure mode.
    let outer_n = 1_025usize;
    let inner_n = 1_025u64;
    let expected_inner: u64 = (0..inner_n).sum();

    let results: Vec<u64> = (0..outer_n as u64)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(move |x| {
            let inner_sum: u64 = (0..inner_n)
                .collect::<Vec<_>>()
                .into_par_iter()
                .reduce(|a, b| a + b)
                .unwrap_or(0);
            inner_sum.wrapping_add(x)
        })
        .collect();

    assert_eq!(results.len(), outer_n);
    for (x, &r) in results.iter().enumerate() {
        assert_eq!(r, expected_inner.wrapping_add(x as u64));
    }
}

#[test]
fn test_range_parallel() {
    let result: Vec<usize> = (0..10).into_par_iter().map(|x| x * x).collect();
    let expected: Vec<usize> = (0..10).map(|x| x * x).collect();
    assert_eq!(result, expected);
}

#[test]
fn sequential_adapter_yields_the_parallel_items_in_order() {
    let doubled: Vec<i32> = vec![1, 2, 3, 4, 5]
        .into_par_iter()
        .map(|x| x * 2)
        .sequential()
        .into_iter()
        .collect();
    assert_eq!(doubled, vec![2, 4, 6, 8, 10]);
}

#[test]
fn sequential_iter_adapter_drives_a_sequential_source_through_the_consumers() {
    let squares: Vec<usize> = SequentialIterAdapter::new(1..=4).map(|x| x * x).collect();
    assert_eq!(squares, vec![1, 4, 9, 16]);
}
