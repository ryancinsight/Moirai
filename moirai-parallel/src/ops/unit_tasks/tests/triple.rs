use super::super::for_each_unit_task_triple_mut_with;
use super::FOUR_UNITS;
use crate::policy::{ExecutionPolicy, Parallel, Sequential};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Runs `(first_unit, a length, b length, c length)` for ten aligned units of
/// three under policy `P`, sorted, after checking all three buffers saw their
/// own unit index.
fn recorded_triple_runs<P: ExecutionPolicy>() -> Vec<(usize, usize, usize, usize)> {
    let mut a = vec![usize::MAX; 30];
    let mut b = vec![u32::MAX; 30];
    let mut c = vec![u8::MAX; 30];
    let runs = Mutex::new(Vec::new());
    for_each_unit_task_triple_mut_with::<P, _, _, _, _, _, _>(
        &mut a,
        &mut b,
        &mut c,
        3,
        FOUR_UNITS,
        || (),
        |(), first_unit, run_a, run_b, run_c| {
            let aligned = run_a.iter_mut().zip(run_b.iter_mut()).zip(run_c.iter_mut());
            for (offset, ((x, y), z)) in aligned.enumerate() {
                let unit = first_unit + offset / 3;
                *x = unit;
                *y = u32::try_from(unit).expect("ten units fit");
                *z = u8::try_from(unit).expect("ten units fit");
            }
            runs.lock()
                .expect("no task panicked while holding the record")
                .push((first_unit, run_a.len(), run_b.len(), run_c.len()));
        },
    );
    let expected: Vec<usize> = (0..30).map(|index| index / 3).collect();
    assert_eq!(a, expected);
    assert!(
        b.iter()
            .zip(&c)
            .zip(&expected)
            .all(|((&y, &z), &unit)| y as usize == unit && z as usize == unit),
        "every element of b and c saw its own unit index"
    );
    let mut runs = runs
        .into_inner()
        .expect("no task panicked while holding the record");
    runs.sort_unstable();
    runs
}

#[test]
fn triple_tasks_carry_aligned_whole_units() {
    let expected = vec![(0, 12, 12, 12), (4, 12, 12, 12), (8, 6, 6, 6)];
    assert_eq!(recorded_triple_runs::<Parallel>(), expected);
    assert_eq!(recorded_triple_runs::<Sequential>(), expected);
}

#[test]
fn triple_state_is_built_once_per_task_in_parallel_and_once_in_all_serially() {
    for (expected, parallel) in [(3, true), (1, false)] {
        let inits = AtomicUsize::new(0);
        let (mut a, mut b, mut c) = (vec![0_u8; 30], vec![0_u16; 30], vec![0_u32; 30]);
        let init = || {
            inits.fetch_add(1, Ordering::Relaxed);
        };
        let body = |(): &mut (), _: usize, _: &mut [u8], _: &mut [u16], _: &mut [u32]| {};
        if parallel {
            for_each_unit_task_triple_mut_with::<Parallel, _, _, _, _, _, _>(
                &mut a, &mut b, &mut c, 3, FOUR_UNITS, init, body,
            );
        } else {
            for_each_unit_task_triple_mut_with::<Sequential, _, _, _, _, _, _>(
                &mut a, &mut b, &mut c, 3, FOUR_UNITS, init, body,
            );
        }
        assert_eq!(inits.load(Ordering::Relaxed), expected);
    }
}

#[test]
#[should_panic(expected = "triple unit tasks need buffers of one length")]
fn triple_second_buffer_of_a_different_length_is_rejected() {
    let (mut a, mut b, mut c) = (vec![0_u8; 12], vec![0_u8; 9], vec![0_u8; 12]);
    for_each_unit_task_triple_mut_with::<Sequential, _, _, _, _, _, _>(
        &mut a,
        &mut b,
        &mut c,
        3,
        1,
        || (),
        |(), _, _, _, _| {},
    );
}

#[test]
#[should_panic(expected = "triple unit tasks need buffers of one length")]
fn triple_third_buffer_of_a_different_length_is_rejected() {
    let (mut a, mut b, mut c) = (vec![0_u8; 12], vec![0_u8; 12], vec![0_u8; 15]);
    for_each_unit_task_triple_mut_with::<Sequential, _, _, _, _, _, _>(
        &mut a,
        &mut b,
        &mut c,
        3,
        1,
        || (),
        |(), _, _, _, _| {},
    );
}
