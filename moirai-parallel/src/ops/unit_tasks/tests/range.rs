use super::super::{UNIT_TASK_BYTES, for_each_unit_task_range_with};
use super::{FOUR_UNITS, REPORTED, Reporting};
use crate::policy::{ExecutionPolicy, Parallel, Sequential};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Runs `(first_unit, count)` for `units` units under policy `P`, sorted,
/// after checking that every unit index in `0..units` was passed exactly once.
fn recorded_range_runs<P: ExecutionPolicy>(units: usize, unit_bytes: usize) -> Vec<(usize, usize)> {
    let seen: Vec<AtomicUsize> = (0..units).map(|_| AtomicUsize::new(0)).collect();
    let runs = Mutex::new(Vec::new());
    for_each_unit_task_range_with::<P, _, _, _>(
        units,
        unit_bytes,
        || (),
        |(), first, count| {
            for visits in seen.iter().skip(first).take(count) {
                visits.fetch_add(1, Ordering::Relaxed);
            }
            runs.lock()
                .expect("no task panicked while holding the record")
                .push((first, count));
        },
    );
    assert!(
        seen.iter().all(|count| count.load(Ordering::Relaxed) == 1),
        "every unit index is passed to exactly one call"
    );
    let mut runs = runs
        .into_inner()
        .expect("no task panicked while holding the record");
    runs.sort_unstable();
    runs
}

#[test]
fn range_tasks_carry_whole_units_with_their_first_index() {
    // Ten units, four to a task: runs of 4, 4 and a ragged 2, whichever
    // branch of the policy runs.
    let expected = vec![(0, 4), (4, 4), (8, 2)];
    assert_eq!(recorded_range_runs::<Parallel>(10, FOUR_UNITS), expected);
    assert_eq!(recorded_range_runs::<Sequential>(10, FOUR_UNITS), expected);
}

#[test]
fn a_range_unit_at_the_task_width_is_a_task_of_its_own() {
    assert_eq!(
        recorded_range_runs::<Parallel>(3, UNIT_TASK_BYTES),
        vec![(0, 1), (1, 1), (2, 1)]
    );
    // Past the width the split cannot go finer than one unit.
    assert_eq!(
        recorded_range_runs::<Parallel>(2, 4 * UNIT_TASK_BYTES),
        vec![(0, 1), (1, 1)]
    );
}

#[test]
fn a_range_of_no_units_runs_nothing() {
    let calls = AtomicUsize::new(0);
    for_each_unit_task_range_with::<Parallel, _, _, _>(
        0,
        FOUR_UNITS,
        || (),
        |(), _, _| {
            calls.fetch_add(1, Ordering::Relaxed);
        },
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn range_state_is_built_once_per_task_in_parallel_and_once_in_all_serially() {
    // Ten units at four to a task is three tasks.
    let built = AtomicUsize::new(0);
    for_each_unit_task_range_with::<Parallel, _, _, _>(
        10,
        FOUR_UNITS,
        || built.fetch_add(1, Ordering::Relaxed),
        |_, _, _| {},
    );
    assert_eq!(built.load(Ordering::Relaxed), 3);

    let built = AtomicUsize::new(0);
    for_each_unit_task_range_with::<Sequential, _, _, _>(
        10,
        FOUR_UNITS,
        || built.fetch_add(1, Ordering::Relaxed),
        |_, _, _| {},
    );
    assert_eq!(built.load(Ordering::Relaxed), 1);
}

#[test]
fn the_policy_sees_the_range_units_tasks_and_bytes_moved() {
    for_each_unit_task_range_with::<Reporting, _, _, _>(10, FOUR_UNITS, || (), |(), _, _| {});
    assert_eq!(
        *REPORTED.lock().expect("the reporting policy never panics"),
        Some((10, 3, 10 * FOUR_UNITS))
    );
}
