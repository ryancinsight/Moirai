use super::super::for_each_unit_task_many_mut_with;
use super::FOUR_UNITS;
use crate::policy::{ExecutionPolicy, Parallel, Sequential};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

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
