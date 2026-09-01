use super::support::{allocations_of, source, ALLOCATION_BUDGET, LEN};
use moirai_iter::parallel::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

#[test]
fn borrowed_map_reassociated_sum_allocates_sublinearly() {
    let data = source();
    let expected: u64 = data.iter().map(|value| value % 7).sum();
    let (total, allocations) = allocations_of(|| {
        data.par_iter()
            .map(|value| value % 7)
            .sum_reassociated::<u64>()
    });
    assert_eq!(total, expected);
    assert!(
        allocations <= ALLOCATION_BUDGET,
        "borrowed map/sum made {allocations} allocations"
    );
}

#[test]
fn borrowed_copied_map_filter_standard_sum_allocates_nothing() {
    let data = source();
    let expected: u64 = data
        .iter()
        .copied()
        .map(|value| value.wrapping_mul(19).wrapping_add(23))
        .filter(|value| value & 7 != 0)
        .sum();
    let (total, allocations) = allocations_of(|| {
        data.par_iter()
            .copied()
            .map(|value| value.wrapping_mul(19).wrapping_add(23))
            .filter(|value| value & 7 != 0)
            .sum::<u64>()
    });
    assert_eq!(total, expected);
    assert_eq!(allocations, 0);
}

#[test]
fn owned_map_reassociated_sum_allocates_sublinearly() {
    let data = source();
    let expected: u64 = data.iter().map(|value| value % 7).sum();
    let (total, allocations) = allocations_of(|| {
        data.clone()
            .into_par_iter()
            .map(|value| value % 7)
            .sum_reassociated::<u64>()
    });
    assert_eq!(total, expected);
    assert!(allocations <= ALLOCATION_BUDGET);
}

#[test]
fn count_and_extrema_allocate_sublinearly() {
    let data = source();
    let (count, count_allocations) = allocations_of(|| data.par_iter().count());
    assert_eq!(count, LEN);
    assert!(count_allocations <= ALLOCATION_BUDGET);

    let (maximum, maximum_allocations) = allocations_of(|| data.par_iter().max());
    assert_eq!(maximum, data.iter().max());
    assert!(maximum_allocations <= ALLOCATION_BUDGET);
}

#[test]
fn find_any_allocates_sublinearly_and_short_circuits() {
    let mut data = source();
    let target = u64::MAX;
    data[LEN / 8] = target;
    let (found, allocations) =
        allocations_of(|| data.par_iter().find_any(|value| **value == target));
    assert_eq!(found, Some(&target));
    assert!(allocations <= ALLOCATION_BUDGET);
}
