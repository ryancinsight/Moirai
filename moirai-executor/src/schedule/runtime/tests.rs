//! Unit tests for the thread scheduler runtime.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc, Barrier,
};

use super::types::{get_current_worker_id, ThreadScheduler};
use crate::schedule::{AsyncTask, BlockingTask, SyncTask};
use moirai_core::{
    error::{ExecutorError, TaskError},
    Priority,
};

#[test]
fn scheduler_runs_all_work_classes_through_one_facade() {
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

    for _ in 0..256 {
        scheduler
            .schedule::<BlockingTask, _>(Priority::Normal, None, |_| {})
            .unwrap();
    }
    let rejection = scheduler
        .schedule::<BlockingTask, _>(Priority::Normal, None, |_| {})
        .expect_err("capacity plus one admission must fail");
    assert!(matches!(rejection, ExecutorError::ResourceExhausted(_)));
    assert_eq!(scheduler.pending_tasks(), 256);

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
fn saturated_indexed_admission_runs_rejected_chunks_on_caller() {
    const ADMISSION_CAPACITY: usize = crate::schedule::queue::INJECTOR_CAPACITY;
    let scheduler = ThreadScheduler::<256>::new(1, "indexed-caller-runs").unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })
        .unwrap();
    started_rx.recv().unwrap();

    for _ in 0..ADMISSION_CAPACITY {
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, |_| {})
            .unwrap();
    }

    let visits: [AtomicUsize; 2] = std::array::from_fn(|_| AtomicUsize::new(0));
    scheduler
        .for_each_indexed::<SyncTask, _>(Priority::Normal, None, visits.len(), |index| {
            visits[index].fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    assert_eq!(visits.map(|count| count.load(Ordering::Relaxed)), [1, 1]);

    let sum = scheduler
        .map_reduce_indexed::<SyncTask, _, _, _>(
            Priority::Normal,
            None,
            2,
            0usize,
            |index| index + 1,
            usize::wrapping_add,
        )
        .unwrap();
    assert_eq!(sum, 3);

    let panic_result =
        scheduler.for_each_indexed::<SyncTask, _>(Priority::Normal, None, 2, |index| {
            if index == 1 {
                panic!("caller-run chunk panic");
            }
        });
    assert_eq!(
        panic_result,
        Err(ExecutorError::SpawnFailed(TaskError::Panicked))
    );

    let reduction_panic = scheduler.map_reduce_indexed::<SyncTask, _, _, _>(
        Priority::Normal,
        None,
        2,
        0usize,
        |index| {
            if index == 1 {
                panic!("caller-run mapper panic");
            }
            index + 1
        },
        usize::wrapping_add,
    );
    assert_eq!(
        reduction_panic,
        Err(ExecutorError::SpawnFailed(TaskError::Panicked))
    );
    assert_eq!(scheduler.admission_caller_runs(), 4);

    release_tx.send(()).unwrap();
    scheduler.join().unwrap();
    assert_eq!(scheduler.pending_tasks(), 0);
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, |_| {})
        .unwrap();
    scheduler.join().unwrap();
    scheduler.shutdown();
}

#[test]
fn saturated_scope_admission_runs_rejected_jobs_on_caller() {
    // A scope owes its caller that every spawned job ran by the time it
    // returns. `flush` used to drop a job the admission queue rejected, so the
    // caller resumed as though borrowed work had happened when it never did —
    // silent, and invisible to the scope's own counters, which the dropped
    // job's completion token decrements either way.
    const ADMISSION_CAPACITY: usize = crate::schedule::queue::INJECTOR_CAPACITY;
    const SCOPED_JOBS: usize = 4;

    let scheduler = ThreadScheduler::<256>::new(1, "scope-caller-runs").unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })
        .unwrap();
    started_rx.recv().unwrap();

    for _ in 0..ADMISSION_CAPACITY {
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, |_| {})
            .unwrap();
    }

    let caller_runs_before = scheduler.admission_caller_runs();
    let visits: [AtomicUsize; SCOPED_JOBS] = std::array::from_fn(|_| AtomicUsize::new(0));
    let lanes: [AtomicUsize; SCOPED_JOBS] = std::array::from_fn(|_| AtomicUsize::new(usize::MAX));

    scheduler
        .scope::<SyncTask, _>(Priority::Normal, None, |scope| {
            for (index, (visit, lane)) in visits.iter().zip(lanes.iter()).enumerate() {
                scope.spawn(move |worker_id| {
                    visit.fetch_add(1, Ordering::Relaxed);
                    lane.store(worker_id, Ordering::Relaxed);
                    let _ = index;
                })?;
            }
            Ok(())
        })
        .expect("a saturated scope must still complete every spawned job");

    // Exactly once each: the refused job runs on the caller instead of being
    // dropped, and it must not also reach a worker.
    for visit in &visits {
        assert_eq!(visit.load(Ordering::Relaxed), 1);
    }
    // The caller's lane is the one past the last worker, never a worker index.
    for lane in &lanes {
        assert_eq!(lane.load(Ordering::Relaxed), scheduler.worker_count());
    }
    assert!(
        scheduler.admission_caller_runs() > caller_runs_before,
        "the caller-run backpressure event must be surfaced, not silent"
    );

    release_tx.send(()).unwrap();
    scheduler.join().unwrap();
    assert_eq!(scheduler.pending_tasks(), 0);
    scheduler.shutdown();
}

#[test]
fn saturated_scope_propagates_a_caller_run_job_panic() {
    // A job the caller runs keeps a worker's panic semantics: the scope reports
    // failure rather than unwinding through the scope body.
    const ADMISSION_CAPACITY: usize = crate::schedule::queue::INJECTOR_CAPACITY;

    let scheduler = ThreadScheduler::<256>::new(1, "scope-caller-panic").unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })
        .unwrap();
    started_rx.recv().unwrap();

    for _ in 0..ADMISSION_CAPACITY {
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, |_| {})
            .unwrap();
    }

    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let scoped = scheduler.scope::<SyncTask, _>(Priority::Normal, None, |scope| {
        scope.spawn(|_| panic!("caller-run scoped job panic"))?;
        Ok(())
    });
    std::panic::set_hook(previous_hook);

    assert_eq!(scoped, Err(ExecutorError::SpawnFailed(TaskError::Panicked)));

    release_tx.send(()).unwrap();
    scheduler.join().unwrap();
    scheduler.shutdown();
}

#[test]
fn blocking_lane_preserves_compute_progress_when_full() {
    let scheduler = ThreadScheduler::new(2, "blocking-lane-progress").unwrap();
    let blocking_started = Arc::new(Barrier::new(3));
    let blocking_release = Arc::new(Barrier::new(3));

    for _ in 0..2 {
        let blocking_started = Arc::clone(&blocking_started);
        let blocking_release = Arc::clone(&blocking_release);
        scheduler
            .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
                blocking_started.wait();
                blocking_release.wait();
            })
            .unwrap();
    }
    blocking_started.wait();

    let (compute_sender, compute_receiver) = mpsc::sync_channel(1);
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
            compute_sender.send(91usize).unwrap();
        })
        .unwrap();

    assert_eq!(
        compute_receiver
            .recv()
            .expect("compute work must not wait behind blocking work"),
        91
    );
    blocking_release.wait();
    scheduler.join().unwrap();
    scheduler.shutdown();
}

#[test]
fn blocking_lane_accepts_concurrent_producers() {
    const PRODUCERS: usize = 4;
    const JOBS_PER_PRODUCER: usize = 32;
    let scheduler = ThreadScheduler::new(PRODUCERS, "blocking-lane-producers").unwrap();
    let completed = Arc::new(AtomicUsize::new(0));

    std::thread::scope(|scope| {
        for _ in 0..PRODUCERS {
            let scheduler = scheduler.clone();
            let completed = Arc::clone(&completed);
            scope.spawn(move || {
                for _ in 0..JOBS_PER_PRODUCER {
                    let completed = Arc::clone(&completed);
                    scheduler
                        .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
                            completed.fetch_add(1, Ordering::Relaxed);
                        })
                        .unwrap();
                }
            });
        }
    });

    scheduler.join().unwrap();
    scheduler.shutdown();
    assert_eq!(
        completed.load(Ordering::Relaxed),
        PRODUCERS * JOBS_PER_PRODUCER
    );
}

#[test]
fn blocking_lane_preserves_priority_order() {
    let scheduler = ThreadScheduler::<8>::new_with_config(1, "blocking-lane-priority").unwrap();
    let blocking_started = Arc::new(Barrier::new(2));
    let blocking_release = Arc::new(Barrier::new(2));
    let (observed_sender, observed_receiver) = mpsc::channel();

    let started = Arc::clone(&blocking_started);
    let release = Arc::clone(&blocking_release);
    scheduler
        .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
            started.wait();
            release.wait();
        })
        .unwrap();
    blocking_started.wait();

    let low_sender = observed_sender.clone();
    scheduler
        .schedule::<BlockingTask, _>(Priority::Low, None, move |_| {
            low_sender.send(1usize).unwrap();
        })
        .unwrap();
    scheduler
        .schedule::<BlockingTask, _>(Priority::Critical, None, move |_| {
            observed_sender.send(2usize).unwrap();
        })
        .unwrap();

    blocking_release.wait();
    scheduler.join().unwrap();
    assert_eq!(
        [
            observed_receiver.recv().unwrap(),
            observed_receiver.recv().unwrap()
        ],
        [2, 1]
    );
    scheduler.shutdown();
}

#[test]
fn blocking_lane_rejects_admission_after_shutdown() {
    let scheduler = ThreadScheduler::new(1, "blocking-lane-shutdown").unwrap();
    scheduler.shutdown();

    let result = scheduler.schedule::<BlockingTask, _>(Priority::Normal, None, |_| {});
    assert_eq!(result, Err(ExecutorError::ShuttingDown));
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
    // and no runnable worker. Nested indexed regions flatten onto their current
    // worker lane, retaining outer parallelism without recursive job stealing.
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
                    let outer_worker = get_current_worker_id()
                        .expect("scoped outer task must execute on a scheduler worker");
                    barrier.wait();
                    scheduler
                        .for_each_indexed::<SyncTask, _>(
                            Priority::Normal,
                            None,
                            INNER_ITEMS,
                            |inner_index| {
                                assert_eq!(get_current_worker_id(), Some(outer_worker));
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
                            |inner_index| {
                                assert_eq!(get_current_worker_id(), Some(outer_worker));
                                outer_index * INNER_ITEMS + inner_index + 1
                            },
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
fn indexed_caller_flattens_nested_regions_onto_its_lane() {
    const WORKERS: usize = 2;
    const OUTER_ITEMS: usize = WORKERS + 1;
    const INNER_ITEMS: usize = 32;
    let scheduler = ThreadScheduler::new(WORKERS, "test-indexed-caller-nesting").unwrap();
    let visited = AtomicUsize::new(0);
    let reduced = AtomicUsize::new(0);

    scheduler
        .for_each_indexed::<SyncTask, _>(Priority::Normal, None, OUTER_ITEMS, |outer_index| {
            let outer_lane = get_current_worker_id();
            scheduler
                .for_each_indexed::<SyncTask, _>(
                    Priority::Normal,
                    None,
                    INNER_ITEMS,
                    |inner_index| {
                        assert_eq!(get_current_worker_id(), outer_lane);
                        visited.fetch_add(
                            outer_index * INNER_ITEMS + inner_index + 1,
                            Ordering::Relaxed,
                        );
                    },
                )
                .expect("nested indexed fan-out must remain on its outer lane");

            let local_sum = scheduler
                .map_reduce_indexed::<SyncTask, _, _, _>(
                    Priority::Normal,
                    None,
                    INNER_ITEMS,
                    0usize,
                    |inner_index| {
                        assert_eq!(get_current_worker_id(), outer_lane);
                        outer_index * INNER_ITEMS + inner_index + 1
                    },
                    usize::wrapping_add,
                )
                .expect("nested indexed reduction must remain on its outer lane");
            reduced.fetch_add(local_sum, Ordering::Relaxed);
        })
        .unwrap();

    let item_count = OUTER_ITEMS * INNER_ITEMS;
    let expected = item_count * (item_count + 1) / 2;
    assert_eq!(visited.load(Ordering::Relaxed), expected);
    assert_eq!(reduced.load(Ordering::Relaxed), expected);
    scheduler.join().unwrap();
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
    // Three lanes cover four items: the caller lane plus two scheduled worker
    // chunks. Scheduler completion metrics count both worker jobs because each
    // job contains its mapper panic and completes its scheduler lifecycle.
    assert_eq!(metrics.completed_tasks, 2);
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
fn indexed_operations_use_every_available_lane_above_cap() {
    const COUNT: usize = 10;
    const WORKERS: usize = 8;
    let scheduler = ThreadScheduler::new(WORKERS, "test-indexed-all-lanes").unwrap();
    let visits: [AtomicUsize; COUNT] = std::array::from_fn(|_| AtomicUsize::new(0));

    scheduler
        .for_each_indexed::<SyncTask, _>(Priority::Normal, None, COUNT, |index| {
            visits[index].fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    let sum = scheduler
        .map_reduce_indexed::<SyncTask, _, _, _>(
            Priority::Normal,
            None,
            COUNT,
            0usize,
            |index| index + 1,
            usize::wrapping_add,
        )
        .unwrap();
    scheduler.join().unwrap();
    let metrics = scheduler.metrics();
    scheduler.shutdown();

    assert_eq!(
        visits.map(|count| count.load(Ordering::Relaxed)),
        [1; COUNT]
    );
    assert_eq!(sum, COUNT * (COUNT + 1) / 2);
    assert_eq!(
        metrics.completed_tasks,
        2 * u64::try_from(WORKERS).expect("worker count must fit scheduler metrics")
    );
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
