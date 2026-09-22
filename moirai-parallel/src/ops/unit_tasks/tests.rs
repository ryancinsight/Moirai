use super::{
    UNIT_TASK_BYTES, for_each_unit_task_many_mut_with, for_each_unit_task_mut_with,
    for_each_unit_task_pair_mut_with, for_each_unit_task_range_with,
    for_each_unit_task_triple_mut_with, units_per_task,
};
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

/// Runs `(first_unit, run length)` for ten aligned units of three across six
/// buffers under policy `P`, sorted, after checking every buffer saw its own
/// unit index in every element.
fn recorded_many_runs<P: ExecutionPolicy>() -> Vec<(usize, usize)> {
    let mut fields = [(); 6].map(|()| vec![usize::MAX; 30]);
    let runs = Mutex::new(Vec::new());
    {
        let mut borrowed = fields.each_mut().map(Vec::as_mut_slice);
        for_each_unit_task_many_mut_with::<P, _, _, _, _, 6>(
            borrowed.each_mut().map(|run| &mut **run),
            3,
            FOUR_UNITS,
            || (),
            |(), first_unit, mut runs_of_task| {
                let len = runs_of_task[0].len();
                for (field, run) in runs_of_task.iter_mut().enumerate() {
                    for (offset, cell) in run.iter_mut().enumerate() {
                        *cell = field * 100 + first_unit + offset / 3;
                    }
                }
                runs.lock()
                    .expect("no task panicked while holding the record")
                    .push((first_unit, len));
            },
        );
    }
    for (field, values) in fields.iter().enumerate() {
        let expected: Vec<usize> = (0..30).map(|index| field * 100 + index / 3).collect();
        assert_eq!(values, &expected, "buffer {field} saw its own unit indices");
    }
    let mut runs = runs
        .into_inner()
        .expect("no task panicked while holding the record");
    runs.sort_unstable();
    runs
}

#[test]
fn many_tasks_carry_aligned_whole_units_across_six_buffers() {
    let expected = vec![(0, 12), (4, 12), (8, 6)];
    assert_eq!(recorded_many_runs::<Parallel>(), expected);
    assert_eq!(recorded_many_runs::<Sequential>(), expected);
}

#[test]
fn many_state_is_built_once_per_task_in_parallel_and_once_in_all_serially() {
    for (expected, parallel) in [(3, true), (1, false)] {
        let inits = AtomicUsize::new(0);
        let mut fields = [(); 6].map(|()| vec![0_u8; 30]);
        let init = || {
            inits.fetch_add(1, Ordering::Relaxed);
        };
        let body = |(): &mut (), _: usize, _: [&mut [u8]; 6]| {};
        let mut borrowed = fields.each_mut().map(Vec::as_mut_slice);
        if parallel {
            for_each_unit_task_many_mut_with::<Parallel, _, _, _, _, 6>(
                borrowed.each_mut().map(|run| &mut **run),
                3,
                FOUR_UNITS,
                init,
                body,
            );
        } else {
            for_each_unit_task_many_mut_with::<Sequential, _, _, _, _, 6>(
                borrowed.each_mut().map(|run| &mut **run),
                3,
                FOUR_UNITS,
                init,
                body,
            );
        }
        assert_eq!(inits.load(Ordering::Relaxed), expected);
    }
}

#[test]
fn many_over_no_buffers_does_nothing() {
    for_each_unit_task_many_mut_with::<Sequential, u8, _, _, _, 0>(
        [],
        3,
        1,
        || (),
        |(), _, _| unreachable!("no buffer means no run"),
    );
}

#[test]
#[should_panic(expected = "many unit tasks need buffers of one length")]
fn many_buffer_of_a_different_length_is_rejected() {
    let (mut a, mut b, mut c) = (vec![0_u8; 12], vec![0_u8; 12], vec![0_u8; 9]);
    for_each_unit_task_many_mut_with::<Sequential, _, _, _, _, 3>(
        [&mut a, &mut b, &mut c],
        3,
        1,
        || (),
        |(), _, _| {},
    );
}
