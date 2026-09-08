//! Borrowing scopes: nesting, recursion, panics, and quiescence on exit.

use super::*;

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
    let Err(payload) = result else {
        panic!("invariant: the scope body panic must resume once the registered jobs complete");
    };
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("scope body panic"),
        "the body's own panic, not a job's, resumes after the scope drains"
    );
    assert_eq!(completed.load(Ordering::Relaxed), 8);
}
