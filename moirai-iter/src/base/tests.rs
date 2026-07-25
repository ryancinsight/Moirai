use super::*;

#[test]
fn thread_pool_always_has_a_worker() {
    // A pool with zero workers accepts jobs and runs none, so anything waiting
    // on them waits forever. Asserted structurally rather than by submitting a
    // job, so a regression fails the test instead of hanging the suite.
    let pool = ThreadPool::new(0);

    assert!(
        format!("{pool:?}").contains("workers: 1"),
        "a zero-sized pool must still start one worker, got {pool:?}"
    );
}

#[test]
fn thread_pool_worker_survives_a_panicking_job() {
    // Workers are never replaced, so a job that unwound its worker would shrink
    // the pool permanently; once all had died, `execute` would queue jobs nobody
    // runs and the next join would hang. The worker must absorb the panic and
    // go back to taking jobs.
    let pool = ThreadPool::new(1);
    let (tx, rx) = std::sync::mpsc::channel();

    // The panic below prints through the default hook. That output is expected,
    // and the hook is deliberately left alone: `set_hook` is process-wide, so
    // replacing it would suppress diagnostics from any test sharing the process
    // under a thread-per-test runner.
    pool.execute(|| panic!("deliberate: the worker must survive this"));

    let follow_up = tx.clone();
    pool.execute(move || {
        let _ = follow_up.send(());
    });

    // Releasing the test's own sender leaves the follow-up job holding the only
    // other one, which is what makes both outcomes observable without a timeout.
    drop(tx);

    // If the worker survived it runs the follow-up and this returns `Ok`. If it
    // unwound instead, it dropped the last reference to the shared receiver on
    // its way out; the queued follow-up job is destroyed with the channel, its
    // sender goes with it, and this returns `Err` rather than blocking.
    let survived = rx.recv().is_ok();

    assert!(
        survived,
        "the worker died with the panicking job, so the follow-up never ran"
    );
}

#[test]
fn pool_join_guard_accepts_a_full_set_of_completions() {
    let (tx, rx) = std::sync::mpsc::channel();
    let guard = PoolJoinGuard::new(rx, 3);
    for _ in 0..3 {
        tx.send(()).expect("receiver is alive");
    }
    drop(tx);

    guard.wait();
}

#[test]
#[should_panic(expected = "did not report completion")]
fn pool_join_guard_rejects_a_missing_completion() {
    // A worker that panics unwinds without sending and drops its sender, which
    // is exactly this shape: fewer messages than tasks, then a disconnected
    // channel. `recv` returns `Err` immediately from then on, so discarding the
    // result would let `wait` return as though every task had finished — and
    // `ZeroCopyParallelIter::map` would `assume_init` a slice no worker wrote.
    let (tx, rx) = std::sync::mpsc::channel();
    let guard = PoolJoinGuard::new(rx, 3);
    tx.send(()).expect("receiver is alive");
    tx.send(()).expect("receiver is alive");
    drop(tx); // the third task's sender dies with it, as in a panic

    guard.wait();
}

#[test]
fn test_tree_reduce() {
    let items = vec![1, 2, 3, 4, 5];
    let result = tree_reduce(items, |a, b| a + b);
    assert_eq!(result, Some(15));

    let empty: Vec<i32> = vec![];
    let result = tree_reduce(empty, |a, b| a + b);
    assert_eq!(result, None);
}

#[test]
fn test_process_in_batches() {
    let items = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let result = process_in_batches(items, 3, |chunk| vec![chunk.iter().sum::<i32>()]);
    assert_eq!(result, vec![6, 15, 15]);
}

#[test]
fn base_adapters_expose_components_without_dead_fields() {
    let base = BaseIterator::new(vec![1_u64, 2, 3], "context");
    assert_eq!(base.inner(), &vec![1, 2, 3]);
    assert_eq!(**base.context(), "context");
    let (inner, context) = base.into_parts();
    assert_eq!(inner, vec![1, 2, 3]);
    assert_eq!(*context, "context");

    let map = MapAdapter::<_, _, u64, u64>::new(vec![1_u64, 2, 3], |value| value + 1);
    assert_eq!(map.inner(), &vec![1, 2, 3]);
    assert_eq!((map.function())(4), 5);
    let (inner, map_fn) = map.into_parts();
    assert_eq!(inner, vec![1, 2, 3]);
    assert_eq!(map_fn(5), 6);

    let filter = FilterAdapter::<_, _, u64>::new(vec![1_u64, 2, 3], |value: &u64| *value > 1);
    assert_eq!(filter.inner(), &vec![1, 2, 3]);
    assert!(filter.predicate()(&2));
    let (inner, predicate) = filter.into_parts();
    assert_eq!(inner, vec![1, 2, 3]);
    assert!(predicate(&3));

    let batch = BatchAdapter::new(vec![1_u64, 2, 3], 0);
    assert_eq!(batch.inner(), &vec![1, 2, 3]);
    assert_eq!(batch.size(), 1);
    let (inner, size) = batch.into_parts();
    assert_eq!(inner, vec![1, 2, 3]);
    assert_eq!(size, 1);
}

#[test]
fn pool_fallback_only_on_pre_execution_shutdown() {
    use moirai_core::error::ExecutorError;
    assert!(!pool_fallback_permitted(&Ok(())));
    assert!(pool_fallback_permitted(&Err(ExecutorError::ShuttingDown)));
}

#[test]
#[should_panic(expected = "partial execution")]
fn pool_fallback_rejects_partial_execution_errors() {
    use moirai_core::error::ExecutorError;
    let _ = pool_fallback_permitted(&Err(ExecutorError::SpawnFailed(
        moirai_core::error::TaskError::Panicked,
    )));
}

#[test]
fn test_tree_reduce_parallel() {
    let items: Vec<i32> = (1..=1000).collect();
    let result = tree_reduce(items, |a, b| a + b);
    assert_eq!(result, Some(500500));
}

#[test]
fn test_thread_pool_graceful_shutdown() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    {
        let pool = ThreadPool::new(2);

        for _ in 0..4 {
            let counter = counter.clone();
            pool.execute(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            });
        }

        for _ in 0..10 {
            if counter.load(Ordering::SeqCst) == 4 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    assert_eq!(counter_clone.load(Ordering::SeqCst), 4);
}

#[test]
fn test_erased_thread_job_runs_once() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    let counter = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&counter);
    let job = ErasedThreadJob::new(move || {
        observed.fetch_add(1, Ordering::SeqCst);
    });

    job.run();

    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[test]
fn test_erased_thread_job_drops_unrun_capture() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));
    let captured = DropCounter(Arc::clone(&drops));
    let job = ErasedThreadJob::new(move || drop(captured));

    drop(job);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
