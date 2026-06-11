//! Unit tests for the thread scheduler runtime.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc,
};

use super::types::ThreadScheduler;
use super::worker::indexed_reduce_chunk_count;
use crate::schedule::{AsyncTask, BlockingTask, SyncTask};
use moirai_core::{
    error::{ExecutorError, TaskError},
    Priority,
};

#[test]
fn scheduler_runs_all_work_classes_on_one_worker_set() {
    let scheduler = ThreadScheduler::new(2, "test-scheduler").unwrap();
    let completed = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel();

    {
        let completed = Arc::clone(&completed);
        let sender = sender.clone();
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                completed.fetch_add(1, Ordering::AcqRel);
                sender.send(()).unwrap();
            })
            .unwrap();
    }

    {
        let completed = Arc::clone(&completed);
        let sender = sender.clone();
        scheduler
            .schedule::<AsyncTask, _>(Priority::Normal, None, move |_| {
                completed.fetch_add(1, Ordering::AcqRel);
                sender.send(()).unwrap();
            })
            .unwrap();
    }

    {
        let completed = Arc::clone(&completed);
        scheduler
            .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
                completed.fetch_add(1, Ordering::AcqRel);
                sender.send(()).unwrap();
            })
            .unwrap();
    }

    for _ in 0..3 {
        receiver.recv().unwrap();
    }

    scheduler.shutdown();
    let metrics = scheduler.metrics();

    assert_eq!(completed.load(Ordering::Acquire), 3);
    assert_eq!(metrics.worker_count, 2);
    assert_eq!(metrics.pending_tasks, 0);
    assert_eq!(metrics.completed_tasks, 3);
    assert_eq!(metrics.failed_tasks, 0);
}

#[test]
fn quiescent_single_task_selection_reuses_work_class_worker() {
    let scheduler = ThreadScheduler::new(4, "test-quiescent-route").unwrap();
    let first = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);
    let second = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);

    scheduler.shutdown();

    assert_eq!(first, second);
    assert_eq!(first, 3);
}

#[test]
fn serial_handoff_selection_reuses_work_class_worker() {
    let scheduler = ThreadScheduler::new(4, "test-serial-handoff-route").unwrap();
    scheduler.inner.active_workers.store(1, Ordering::Release);

    let first = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);
    let second = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);

    scheduler.inner.active_workers.store(0, Ordering::Release);
    scheduler.shutdown();

    assert_eq!(first, second);
    assert_eq!(first, 3);
}

#[test]
fn queued_parallel_selection_rotates_workers() {
    let scheduler = ThreadScheduler::new(4, "test-parallel-route").unwrap();
    scheduler.inner.pending_tasks.store(1, Ordering::Release);

    let first = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);
    let second = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);

    scheduler.inner.pending_tasks.store(0, Ordering::Release);
    scheduler.shutdown();

    assert_ne!(first, second);
}

#[test]
fn scheduler_scope_runs_borrowing_jobs_before_return() {
    let scheduler = ThreadScheduler::new(2, "test-scope").unwrap();
    let sum = AtomicUsize::new(0);

    scheduler
        .scope::<SyncTask, _>(Priority::Normal, None, |scope| {
            for value in 1..=16 {
                let sum = &sum;
                scope.spawn(move |_| {
                    sum.fetch_add(value, Ordering::Relaxed);
                })?;
            }
            Ok(())
        })
        .unwrap();

    scheduler.shutdown();
    assert_eq!(sum.load(Ordering::Relaxed), 136);
}

#[test]
fn scheduler_scope_reports_panicked_job() {
    let scheduler = ThreadScheduler::new(1, "test-scope-panic").unwrap();
    let completed = AtomicUsize::new(0);

    let result = scheduler.scope::<SyncTask, _>(Priority::Normal, None, |scope| {
        scope.spawn(|_| panic!("scoped job panic"))?;
        let completed = &completed;
        scope.spawn(move |_| {
            completed.fetch_add(1, Ordering::Relaxed);
        })?;
        Ok(())
    });

    scheduler.shutdown();
    assert_eq!(result, Err(ExecutorError::SpawnFailed(TaskError::Panicked)));
    assert_eq!(completed.load(Ordering::Relaxed), 1);
}

#[test]
fn scheduler_join_waits_for_queued_and_active_work() {
    let scheduler = ThreadScheduler::new(2, "test-join").unwrap();
    let completed = Arc::new(AtomicUsize::new(0));

    for _ in 0..8 {
        let completed = Arc::clone(&completed);
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                completed.fetch_add(1, Ordering::AcqRel);
            })
            .unwrap();
    }

    assert!(scheduler.has_work());
    scheduler.join().unwrap();
    let metrics = scheduler.metrics();

    scheduler.shutdown();
    assert_eq!(completed.load(Ordering::Acquire), 8);
    assert_eq!(metrics.pending_tasks, 0);
    assert_eq!(metrics.active_workers, 0);
    assert_eq!(metrics.completed_tasks, 8);
    assert!(!scheduler.has_work());
}

#[test]
fn scheduler_join_waits_for_work_submitted_while_active() {
    let scheduler = ThreadScheduler::new(2, "test-join-transitive").unwrap();
    let completed = Arc::new(AtomicUsize::new(0));
    let (started_sender, started_receiver) = mpsc::channel();
    let (scheduled_sender, scheduled_receiver) = mpsc::channel();

    {
        let completed = Arc::clone(&completed);
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                started_sender.send(()).unwrap();
                scheduled_receiver.recv().unwrap();
                completed.fetch_add(1, Ordering::AcqRel);
            })
            .unwrap();
    }

    started_receiver.recv().unwrap();
    std::thread::scope(|scope| {
        let completed = Arc::clone(&completed);
        let scheduler_ref = &scheduler;
        scope.spawn(move || {
            scheduler_ref
                .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                    completed.fetch_add(1, Ordering::AcqRel);
                })
                .unwrap();
            scheduled_sender.send(()).unwrap();
        });

        scheduler.join().unwrap();
    });

    let metrics = scheduler.metrics();
    scheduler.shutdown();

    assert_eq!(completed.load(Ordering::Acquire), 2);
    assert_eq!(metrics.pending_tasks, 0);
    assert_eq!(metrics.active_workers, 0);
    assert_eq!(metrics.completed_tasks, 2);
}

#[test]
fn indexed_fan_out_runs_all_items() {
    let scheduler = ThreadScheduler::new(2, "test-indexed").unwrap();
    let sum = AtomicUsize::new(0);

    scheduler
        .for_each_indexed::<BlockingTask, _>(Priority::Normal, None, 32, |index| {
            sum.fetch_add(index + 1, Ordering::Relaxed);
        })
        .unwrap();

    scheduler.shutdown();
    assert_eq!(sum.load(Ordering::Relaxed), 528);
}

#[test]
fn indexed_map_reduce_returns_reduced_value() {
    let scheduler = ThreadScheduler::new(2, "test-indexed-reduce").unwrap();

    let sum = scheduler
        .map_reduce_indexed::<BlockingTask, _, _, _>(
            Priority::Normal,
            None,
            32,
            0usize,
            |index| index + 1,
            usize::wrapping_add,
        )
        .unwrap();

    scheduler.shutdown();
    assert_eq!(sum, 528);
}

#[test]
fn indexed_map_reduce_small_count_runs_inline() {
    let scheduler = ThreadScheduler::new(2, "test-indexed-reduce-inline").unwrap();

    let sum = scheduler
        .map_reduce_indexed::<BlockingTask, _, _, _>(
            Priority::Normal,
            None,
            32,
            0usize,
            |index| index + 1,
            usize::wrapping_add,
        )
        .unwrap();
    let metrics = scheduler.metrics();

    scheduler.shutdown();
    assert_eq!(sum, 528);
    assert_eq!(metrics.completed_tasks, 0);
}

#[test]
fn indexed_map_reduce_inline_reports_panicked_mapper() {
    let scheduler = ThreadScheduler::new(2, "test-indexed-reduce-inline-panic").unwrap();

    let result = scheduler.map_reduce_indexed::<BlockingTask, _, _, _>(
        Priority::Normal,
        None,
        4,
        0usize,
        |index| {
            if index == 2 {
                panic!("inline map panic");
            }
            index + 1
        },
        usize::wrapping_add,
    );
    let metrics = scheduler.metrics();

    scheduler.shutdown();
    assert_eq!(result, Err(ExecutorError::SpawnFailed(TaskError::Panicked)));
    assert_eq!(metrics.completed_tasks, 0);
}

#[test]
fn indexed_map_reduce_above_inline_limit_uses_scheduler_chunks() {
    let scheduler = ThreadScheduler::new(2, "test-indexed-reduce-parallel").unwrap();

    let sum = scheduler
        .map_reduce_indexed::<BlockingTask, _, _, _>(
            Priority::Normal,
            None,
            64,
            0usize,
            |index| index + 1,
            usize::wrapping_add,
        )
        .unwrap();
    let metrics = scheduler.metrics();

    scheduler.shutdown();
    assert_eq!(sum, 2080);
    assert_eq!(
        metrics.completed_tasks,
        indexed_reduce_chunk_count::<usize>(64, 2).saturating_sub(1) as u64
    );
}

#[test]
fn indexed_reduce_chunk_count_amortizes_scheduled_work() {
    assert_eq!(indexed_reduce_chunk_count::<usize>(64, 4), 1);
    assert_eq!(indexed_reduce_chunk_count::<usize>(256, 4), 2);
    assert_eq!(indexed_reduce_chunk_count::<usize>(1024, 4), 5);
}

#[test]
fn scheduler_scope_completes_registered_jobs_before_body_error_returns() {
    let scheduler = ThreadScheduler::new(2, "test-scope-body-error").unwrap();
    let completed = AtomicUsize::new(0);

    let result = scheduler.scope::<SyncTask, _>(Priority::Normal, None, |scope| {
        for _ in 0..8 {
            let completed = &completed;
            scope.spawn(move |_| {
                completed.fetch_add(1, Ordering::Relaxed);
            })?;
        }

        Err(ExecutorError::SpawnFailed(TaskError::InvalidOperation))
    });

    scheduler.shutdown();
    assert_eq!(
        result,
        Err(ExecutorError::SpawnFailed(TaskError::InvalidOperation))
    );
    assert_eq!(completed.load(Ordering::Relaxed), 8);
}

#[test]
fn scheduler_scope_completes_registered_jobs_before_resuming_body_panic() {
    let scheduler = ThreadScheduler::new(2, "test-scope-body-panic").unwrap();
    let completed = AtomicUsize::new(0);

    let result = catch_unwind(AssertUnwindSafe(|| {
        scheduler
            .scope::<SyncTask, _>(Priority::Normal, None, |scope| {
                for _ in 0..8 {
                    let completed = &completed;
                    scope.spawn(move |_| {
                        completed.fetch_add(1, Ordering::Relaxed);
                    })?;
                }

                panic!("scope body panic");
            })
            .unwrap();
    }));

    scheduler.shutdown();
    assert!(result.is_err());
    assert_eq!(completed.load(Ordering::Relaxed), 8);
}

#[test]
fn test_melinoe_partition_routing() {
    use melinoe::sync::partition_map;
    use melinoe::{brand_scope, MelinoeCell};

    let _exec = crate::global();

    brand_scope(|token| {
        let mut cells: Vec<MelinoeCell<'_, usize>> = (0..32).map(|_| MelinoeCell::new(0)).collect();

        let results = partition_map(&mut cells, 4, |start, mut shard| {
            for (i, cell) in shard.iter_mut().enumerate() {
                *cell = start + i;
            }
            shard.len()
        });

        assert_eq!(results.len(), 4);
        assert_eq!(results.iter().sum::<usize>(), 32);

        let snap = token.share();
        for (i, cell) in cells.iter().enumerate() {
            assert_eq!(*cell.borrow(snap), i);
        }
    });
}
