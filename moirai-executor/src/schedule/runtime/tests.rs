//! Unit tests for the thread scheduler runtime.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc, Barrier,
};

use super::types::ThreadScheduler;
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
fn saturated_admission_rolls_back_pending_and_recovers() {
    let scheduler = ThreadScheduler::<256>::new(1, "bounded-admission").unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    scheduler
        .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })
        .unwrap();
    started_rx.recv().unwrap();

    for _ in 0..1024 {
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, |_| {})
            .unwrap();
    }
    let rejection = scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, |_| {})
        .expect_err("capacity plus one admission must fail");
    assert!(matches!(rejection, ExecutorError::ResourceExhausted(_)));
    assert_eq!(scheduler.pending_tasks(), 1024);

    release_tx.send(()).unwrap();
    scheduler.join().unwrap();
    assert_eq!(scheduler.pending_tasks(), 0);

    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, |_| {})
        .unwrap();
    scheduler.join().unwrap();
    assert_eq!(scheduler.pending_tasks(), 0);
}

#[test]
fn large_pool_wakes_high_index_workers_across_idle_cycles() {
    // Regression for the single-AtomicU64 idle map: workers with id >= 64 were
    // never registered in the wake bitmap, so on a pool larger than 64 they
    // could not be targeted by the wake lottery. With a multi-word bitset every
    // worker is addressable. Drive several submit -> quiesce -> submit cycles so
    // the whole pool parks between rounds and must be re-woken each round; a
    // lost/unreachable wakeup would either drop a task (count mismatch) or hang
    // into the nextest timeout.
    const WORKERS: usize = 100;
    const ROUNDS: usize = 4;
    const TASKS_PER_ROUND: usize = 400;

    let scheduler = ThreadScheduler::new(WORKERS, "test-large-pool").unwrap();
    let completed = Arc::new(AtomicUsize::new(0));

    for _ in 0..ROUNDS {
        let (sender, receiver) = mpsc::channel();
        for _ in 0..TASKS_PER_ROUND {
            let completed = Arc::clone(&completed);
            let sender = sender.clone();
            scheduler
                .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                    completed.fetch_add(1, Ordering::AcqRel);
                    sender.send(()).unwrap();
                })
                .unwrap();
        }
        drop(sender);
        // Barrier: every task of this round must complete before the next round,
        // forcing the pool to fully quiesce (all workers park) in between.
        for _ in 0..TASKS_PER_ROUND {
            receiver.recv().unwrap();
        }
    }

    scheduler.shutdown();
    assert_eq!(completed.load(Ordering::Acquire), ROUNDS * TASKS_PER_ROUND);
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
fn scheduler_scope_nested_saturation_completes() {
    // Regression guard for ISSUE-208: a scoped job that itself opens a nested
    // scope must complete. Before help-while-waiting, `scope().wait()` parked the
    // caller without running scheduler work, so nested fork-join deadlocked
    // (provably with one worker) and corrupted the heap under concurrent nesting
    // (STATUS_HEAP_CORRUPTION). Value-semantic: every inner increment must land.
    for &workers in &[1usize, 2, 4] {
        let scheduler = ThreadScheduler::new(workers, "test-nested-scope").unwrap();
        let outer = 32usize;
        let inner = 16usize;
        let counter = AtomicUsize::new(0);

        scheduler
            .scope::<SyncTask, _>(Priority::Normal, None, |outer_scope| {
                for _ in 0..outer {
                    let scheduler = &scheduler;
                    let counter = &counter;
                    outer_scope.spawn(move |_| {
                        scheduler
                            .scope::<SyncTask, _>(Priority::Normal, None, |inner_scope| {
                                for _ in 0..inner {
                                    let counter = &counter;
                                    inner_scope.spawn(move |_| {
                                        counter.fetch_add(1, Ordering::Relaxed);
                                    })?;
                                }
                                Ok(())
                            })
                            .expect("nested scope must complete");
                    })?;
                }
                Ok(())
            })
            .unwrap();

        assert_eq!(
            counter.load(Ordering::Relaxed),
            outer * inner,
            "nested saturation lost increments at {workers} worker(s)"
        );
        scheduler.shutdown();
    }
}

fn recursive_scope_sum(scheduler: &ThreadScheduler, lo: u64, hi: u64) -> u64 {
    if hi.saturating_sub(lo) <= 1024 {
        return (lo..hi).sum();
    }
    let mid = lo + (hi - lo) / 2;
    let mut left = 0u64;
    let mut right = 0u64;
    {
        let left = &mut left;
        let right = &mut right;
        scheduler
            .scope::<SyncTask, _>(Priority::Normal, None, |scope| {
                scope.spawn(|_| {
                    *left = recursive_scope_sum(scheduler, lo, mid);
                })?;
                scope.spawn(|_| {
                    *right = recursive_scope_sum(scheduler, mid, hi);
                })?;
                Ok(())
            })
            .expect("recursive scope must complete");
    }
    left + right
}

#[test]
fn scheduler_scope_recursive_fork_join_is_sound() {
    // ISSUE-208 corruption guard: the recursive two-branch fork-join is the exact
    // shape of `moirai_iter` `drive` (log2-depth nested scopes, each branch stolen
    // by a peer worker that dereferences the parent scope's stack-owned state).
    // Before help-while-waiting this deadlocked (one worker) and corrupted the
    // heap (STATUS_HEAP_CORRUPTION) under concurrent nesting. Analytical oracle:
    // the arithmetic series sum, asserted value-semantically.
    const N: u64 = 200_000;
    let expected = N * (N - 1) / 2;
    for &workers in &[1usize, 2, 4] {
        let scheduler = ThreadScheduler::new(workers, "test-recursive-scope").unwrap();
        assert_eq!(
            recursive_scope_sum(&scheduler, 0, N),
            expected,
            "recursive fork-join sum diverged at {workers} worker(s)"
        );
        scheduler.shutdown();
    }
}

#[test]
fn scheduler_scope_nested_leaves_scheduler_quiescent() {
    // Accounting guard for help-while-waiting (ADR-019): a worker waiter runs
    // jobs via a re-entrant `execute_job`, which mutates the global
    // `pending_tasks`/`active_workers` counters. A leaked increment would leave
    // `join()` unable to observe quiescence (hang → nextest terminates). Assert
    // the scheduler returns to a consistent quiescent state with no spurious
    // failures after a nested workload.
    for &workers in &[1usize, 2, 4] {
        let scheduler = ThreadScheduler::new(workers, "test-nested-quiescent").unwrap();
        let outer = 16usize;
        let inner = 8usize;
        let counter = AtomicUsize::new(0);

        scheduler
            .scope::<SyncTask, _>(Priority::Normal, None, |outer_scope| {
                for _ in 0..outer {
                    let scheduler = &scheduler;
                    let counter = &counter;
                    outer_scope.spawn(move |_| {
                        scheduler
                            .scope::<SyncTask, _>(Priority::Normal, None, |inner_scope| {
                                for _ in 0..inner {
                                    let counter = &counter;
                                    inner_scope.spawn(move |_| {
                                        counter.fetch_add(1, Ordering::Relaxed);
                                    })?;
                                }
                                Ok(())
                            })
                            .expect("nested scope must complete");
                    })?;
                }
                Ok(())
            })
            .unwrap();

        // Terminates only if the help path leaked no pending/active count.
        scheduler.join().expect("scheduler must reach quiescence");
        let metrics = scheduler.metrics();

        assert_eq!(counter.load(Ordering::Relaxed), outer * inner);
        assert_eq!(
            metrics.pending_tasks, 0,
            "leaked pending count at {workers} worker(s)"
        );
        assert_eq!(
            metrics.active_workers, 0,
            "leaked active-worker count at {workers} worker(s)"
        );
        assert_eq!(
            metrics.failed_tasks, 0,
            "spurious job failure at {workers} worker(s)"
        );
        scheduler.shutdown();
    }
}

#[test]
fn scheduler_scope_nested_panic_propagates_and_pool_survives() {
    // Adversarial guard for the help-while-waiting scope (ADR-019): when a nested
    // scoped job panics, the nested scope must report SpawnFailed(Panicked), its
    // sibling job must still run, and the outer scope must complete without
    // deadlock or corruption — i.e. a panic on a help-stealing worker unwinds
    // only its own job, never the waiter's help loop.
    for &workers in &[1usize, 2, 4] {
        let scheduler = ThreadScheduler::new(workers, "test-nested-panic").unwrap();
        let outer = 8usize;
        let sibling_ran = AtomicUsize::new(0);
        let nested_panics_reported = AtomicUsize::new(0);

        scheduler
            .scope::<SyncTask, _>(Priority::Normal, None, |outer_scope| {
                for _ in 0..outer {
                    let scheduler = &scheduler;
                    let sibling_ran = &sibling_ran;
                    let nested_panics_reported = &nested_panics_reported;
                    outer_scope.spawn(move |_| {
                        let result =
                            scheduler.scope::<SyncTask, _>(Priority::Normal, None, |inner| {
                                inner.spawn(|_| panic!("nested scoped job panic"))?;
                                inner.spawn(move |_| {
                                    sibling_ran.fetch_add(1, Ordering::Relaxed);
                                })?;
                                Ok(())
                            });
                        if matches!(result, Err(ExecutorError::SpawnFailed(TaskError::Panicked))) {
                            nested_panics_reported.fetch_add(1, Ordering::Relaxed);
                        }
                    })?;
                }
                Ok(())
            })
            .unwrap();

        assert_eq!(
            nested_panics_reported.load(Ordering::Relaxed),
            outer,
            "each nested scope must report its panic at {workers} worker(s)"
        );
        assert_eq!(
            sibling_ran.load(Ordering::Relaxed),
            outer,
            "sibling of a panicking nested job must still run at {workers} worker(s)"
        );
        scheduler.shutdown();
    }
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
fn nested_indexed_saturation_completes() {
    // The outer jobs occupy every worker before entering indexed fan-out. A
    // parking indexed waiter therefore deadlocks with its inner chunks queued
    // and no runnable worker. The scheduler-owned drain path lets each waiter
    // execute queued work until its fan-out completes.
    const WORKERS: usize = 2;
    const INNER_ITEMS: usize = 1024;
    let scheduler = ThreadScheduler::new(WORKERS, "test-nested-indexed").unwrap();
    let barrier = Barrier::new(WORKERS);
    let sum = AtomicUsize::new(0);
    let reduced_sum = AtomicUsize::new(0);

    scheduler
        .scope::<SyncTask, _>(Priority::Normal, None, |scope| {
            for outer_index in 0..WORKERS {
                let scheduler = &scheduler;
                let barrier = &barrier;
                let sum = &sum;
                let reduced_sum = &reduced_sum;
                scope.spawn(move |_| {
                    barrier.wait();
                    scheduler
                        .for_each_indexed::<SyncTask, _>(
                            Priority::Normal,
                            None,
                            INNER_ITEMS,
                            |inner_index| {
                                sum.fetch_add(
                                    outer_index * INNER_ITEMS + inner_index + 1,
                                    Ordering::Relaxed,
                                );
                            },
                        )
                        .expect("nested indexed fan-out must complete");

                    barrier.wait();
                    let local_sum = scheduler
                        .map_reduce_indexed::<SyncTask, _, _, _>(
                            Priority::Normal,
                            None,
                            INNER_ITEMS,
                            0usize,
                            |inner_index| outer_index * INNER_ITEMS + inner_index + 1,
                            usize::wrapping_add,
                        )
                        .expect("nested indexed map/reduce must complete");
                    reduced_sum.fetch_add(local_sum, Ordering::Relaxed);
                })?;
            }
            Ok(())
        })
        .unwrap();

    let item_count = WORKERS * INNER_ITEMS;
    let expected = item_count * (item_count + 1) / 2;
    assert_eq!(sum.load(Ordering::Relaxed), expected);
    assert_eq!(reduced_sum.load(Ordering::Relaxed), expected);
    scheduler.join().unwrap();
    let metrics = scheduler.metrics();
    assert_eq!(metrics.pending_tasks, 0);
    assert_eq!(metrics.active_workers, 0);
    scheduler.shutdown();
}

#[test]
fn indexed_map_reduce_small_count_schedules_worker_lanes() {
    let scheduler = ThreadScheduler::new(2, "test-indexed-reduce-small").unwrap();

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
    scheduler.join().unwrap();
    let metrics = scheduler.metrics();

    scheduler.shutdown();
    assert_eq!(sum, 528);
    assert_eq!(metrics.completed_tasks, 2);
}

#[test]
fn indexed_map_reduce_reports_panicked_mapper() {
    let scheduler = ThreadScheduler::new(2, "test-indexed-reduce-panic").unwrap();

    let result = scheduler.map_reduce_indexed::<BlockingTask, _, _, _>(
        Priority::Normal,
        None,
        4,
        0usize,
        |index| {
            if index == 2 {
                panic!("map panic");
            }
            index + 1
        },
        usize::wrapping_add,
    );
    scheduler.join().unwrap();
    let metrics = scheduler.metrics();

    scheduler.shutdown();
    assert_eq!(result, Err(ExecutorError::SpawnFailed(TaskError::Panicked)));
    assert_eq!(metrics.completed_tasks, 1);
}

#[test]
fn indexed_map_reduce_caps_chunks_at_worker_plus_caller_lanes() {
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
    scheduler.join().unwrap();
    let metrics = scheduler.metrics();

    scheduler.shutdown();
    assert_eq!(sum, 2080);
    assert_eq!(metrics.completed_tasks, 2);
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
