use super::{UNIT_TASK_BYTES, for_each_unit_task_mut_with, units_per_task};
use crate::policy::{
    Adaptive, AdaptiveWithThreshold, ExecutionPolicy, Parallel, Sequential, WorkBytes,
};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Bytes per unit that makes a task carry exactly four units.
const FOUR_UNITS: usize = UNIT_TASK_BYTES / 4;

/// Runs `(first_unit, run length)` for ten units of three under policy `P`,
/// sorted, after checking that every element saw its own unit index.
fn recorded_runs<P: ExecutionPolicy>() -> Vec<(usize, usize)> {
    let mut data = vec![usize::MAX; 30];
    let runs = Mutex::new(Vec::new());
    for_each_unit_task_mut_with::<P, _, _, _, _>(
        &mut data,
        3,
        FOUR_UNITS,
        || (),
        |(), first_unit, run| {
            for (offset, value) in run.iter_mut().enumerate() {
                *value = first_unit + offset / 3;
            }
            runs.lock()
                .expect("no task panicked while holding the record")
                .push((first_unit, run.len()));
        },
    );
    let expected: Vec<usize> = (0..30).map(|index| index / 3).collect();
    assert_eq!(data, expected);
    let mut runs = runs
        .into_inner()
        .expect("no task panicked while holding the record");
    runs.sort_unstable();
    runs
}

#[test]
fn tasks_carry_whole_units_with_their_first_index() {
    // Ten units, four to a task: runs of 4, 4 and a ragged 2.
    let expected = vec![(0, 12), (4, 12), (8, 6)];
    assert_eq!(recorded_runs::<Parallel>(), expected);
    assert_eq!(recorded_runs::<Sequential>(), expected);
}

#[test]
fn a_unit_wider_than_a_task_runs_alone() {
    assert_eq!(units_per_task(UNIT_TASK_BYTES + 1), 1);
    assert_eq!(units_per_task(UNIT_TASK_BYTES), 1);
    assert_eq!(units_per_task(0), UNIT_TASK_BYTES);
    assert_eq!(units_per_task(FOUR_UNITS), 4);
}

#[test]
fn state_is_built_once_per_task_in_parallel_and_once_in_all_serially() {
    for (expected, parallel) in [(3, true), (1, false)] {
        let inits = AtomicUsize::new(0);
        let mut data = vec![0_u8; 30];
        let init = || {
            inits.fetch_add(1, Ordering::Relaxed);
        };
        let body = |(): &mut (), _: usize, _: &mut [u8]| {};
        if parallel {
            for_each_unit_task_mut_with::<Parallel, _, _, _, _>(
                &mut data, 3, FOUR_UNITS, init, body,
            );
        } else {
            for_each_unit_task_mut_with::<Sequential, _, _, _, _>(
                &mut data, 3, FOUR_UNITS, init, body,
            );
        }
        assert_eq!(inits.load(Ordering::Relaxed), expected);
    }
}

/// Last `(len, chunks, bytes)` the operator reported to a policy.
static REPORTED: Mutex<Option<(usize, usize, usize)>> = Mutex::new(None);

struct Reporting;

impl ExecutionPolicy for Reporting {
    fn parallelize(_len: usize) -> bool {
        false
    }

    fn parallelize_work(len: usize, chunks: usize, bytes: usize) -> bool {
        *REPORTED.lock().expect("the reporting policy never panics") = Some((len, chunks, bytes));
        false
    }
}

#[test]
fn the_policy_sees_elements_tasks_and_bytes_moved() {
    let mut data = vec![0_u32; 30];
    for_each_unit_task_mut_with::<Reporting, _, _, _, _>(
        &mut data,
        3,
        FOUR_UNITS,
        || (),
        |(), _, _| {},
    );
    assert_eq!(
        *REPORTED.lock().expect("the reporting policy never panics"),
        Some((30, 3, 10 * FOUR_UNITS))
    );
}

#[test]
fn work_bytes_decides_by_bytes() {
    assert!(!WorkBytes::<100>::parallelize_work(1_000_000, 64, 99));
    assert!(WorkBytes::<100>::parallelize_work(1, 1, 100));
    // Entry points that report no bytes count one byte per element.
    assert!(!WorkBytes::<100>::parallelize(99));
    assert!(WorkBytes::<100>::parallelize(100));
}

#[test]
fn existing_policies_ignore_bytes() {
    for (len, chunks) in [
        (0, 0),
        (1, 1),
        (1023, 4),
        (1024, 1),
        (32_768, 5),
        (1 << 20, 64),
    ] {
        for bytes in [0, 1, usize::MAX] {
            assert_eq!(
                Adaptive::parallelize_work(len, chunks, bytes),
                Adaptive::parallelize_chunks(len, chunks)
            );
            assert_eq!(
                AdaptiveWithThreshold::<32_768>::parallelize_work(len, chunks, bytes),
                AdaptiveWithThreshold::<32_768>::parallelize_chunks(len, chunks)
            );
            assert!(!Sequential::parallelize_work(len, chunks, bytes));
        }
    }
}

#[test]
#[should_panic(expected = "not a multiple of unit length 4")]
fn a_partial_unit_is_rejected() {
    let mut data = vec![0_u8; 10];
    for_each_unit_task_mut_with::<Sequential, _, _, _, _>(&mut data, 4, 1, || (), |(), _, _| {});
}
