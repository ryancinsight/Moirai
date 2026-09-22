use super::super::for_each_unit_task_pair_mut_with;
use super::FOUR_UNITS;
use crate::policy::{ExecutionPolicy, Parallel, Sequential};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Runs `(first_unit, a length, b length)` for ten paired units of three under
/// policy `P`, sorted, after checking both buffers saw their own unit index.
fn recorded_pair_runs<P: ExecutionPolicy>() -> Vec<(usize, usize, usize)> {
    let mut a = vec![usize::MAX; 30];
    let mut b = vec![u32::MAX; 30];
    let runs = Mutex::new(Vec::new());
    for_each_unit_task_pair_mut_with::<P, _, _, _, _, _>(
        &mut a,
        &mut b,
        3,
        FOUR_UNITS,
        || (),
        |(), first_unit, run_a, run_b| {
            for (offset, (x, y)) in run_a.iter_mut().zip(run_b.iter_mut()).enumerate() {
                *x = first_unit + offset / 3;
                *y = u32::try_from(first_unit + offset / 3).expect("ten units fit");
            }
            runs.lock()
                .expect("no task panicked while holding the record")
                .push((first_unit, run_a.len(), run_b.len()));
        },
    );
    let expected: Vec<usize> = (0..30).map(|index| index / 3).collect();
    assert_eq!(a, expected);
    assert_eq!(
        b,
        expected
            .iter()
            .map(|&unit| u32::try_from(unit).expect("ten units fit"))
            .collect::<Vec<u32>>()
    );
    let mut runs = runs
        .into_inner()
        .expect("no task panicked while holding the record");
    runs.sort_unstable();
    runs
}

#[test]
fn paired_tasks_carry_aligned_whole_units() {
    let expected = vec![(0, 12, 12), (4, 12, 12), (8, 6, 6)];
    assert_eq!(recorded_pair_runs::<Parallel>(), expected);
    assert_eq!(recorded_pair_runs::<Sequential>(), expected);
}

#[test]
fn paired_state_is_built_once_per_task_in_parallel_and_once_in_all_serially() {
    for (expected, parallel) in [(3, true), (1, false)] {
        let inits = AtomicUsize::new(0);
        let (mut a, mut b) = (vec![0_u8; 30], vec![0_u16; 30]);
        let init = || {
            inits.fetch_add(1, Ordering::Relaxed);
        };
        let body = |(): &mut (), _: usize, _: &mut [u8], _: &mut [u16]| {};
        if parallel {
            for_each_unit_task_pair_mut_with::<Parallel, _, _, _, _, _>(
                &mut a, &mut b, 3, FOUR_UNITS, init, body,
            );
        } else {
            for_each_unit_task_pair_mut_with::<Sequential, _, _, _, _, _>(
                &mut a, &mut b, 3, FOUR_UNITS, init, body,
            );
        }
        assert_eq!(inits.load(Ordering::Relaxed), expected);
    }
}

#[test]
#[should_panic(expected = "paired unit tasks need buffers of one length")]
fn paired_buffers_of_different_lengths_are_rejected() {
    let (mut a, mut b) = (vec![0_u8; 12], vec![0_u8; 9]);
    for_each_unit_task_pair_mut_with::<Sequential, _, _, _, _, _>(
        &mut a,
        &mut b,
        3,
        1,
        || (),
        |(), _, _, _| {},
    );
}
