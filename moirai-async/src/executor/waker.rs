//! Wake protocol of [`AsyncTask`]: re-enqueue a pending task exactly once.

use crate::executor::task::AsyncTask;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::task::Wake;

impl Wake for AsyncTask {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        // A completed task's waker may still be held live by the reactor (a
        // read-waker registered against a socket fd, say) and fire after the
        // task finished via another path. Re-enqueuing it would poll a future
        // that already returned `Ready` and panic, so drop the wake for a
        // completed task. `process_pending_tasks` re-checks `completed` to
        // close the wake-races-completion window authoritatively.
        if self.completed.load(Ordering::Acquire) {
            return;
        }
        if !self
            .is_queued
            // `is_queued` is a linearization flag for enqueue deduplication,
            // not a publication channel. The queue's per-slot Release/Acquire
            // sequence publishes the task itself; this RMW only orders the
            // false -> true transition against `process_pending_tasks`'
            // corresponding clear. Relaxed is therefore sufficient and avoids
            // a global ordering edge on every wake.
            .swap(true, Ordering::Relaxed)
        {
            // The executor may be gone (a reactor or timer thread woke a task
            // after its executor dropped); nothing will poll it then.
            if let Some(run_queue) = self.run_queue.upgrade() {
                run_queue.enqueue(Arc::clone(self));
                if let Some(reactor) = self.reactor.upgrade() {
                    let _ = reactor.wake();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::executor::AsyncExecutor;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::{Poll, Waker};

    #[test]
    fn every_poll_of_a_task_presents_a_will_wake_equal_waker() {
        use std::sync::Mutex;

        const POLLS: usize = 1_000;

        // A task that re-wakes itself each poll and keeps every waker it was
        // handed. The executor drives all polls from one `process_pending_tasks`
        // call because each self-wake re-enqueues the task.
        let executor = AsyncExecutor::new().expect("a fresh AsyncExecutor must build");
        let seen = Arc::new(Mutex::new(Vec::<Waker>::with_capacity(POLLS)));
        let future_seen = Arc::clone(&seen);
        let _handle = executor.spawn(futures::future::poll_fn(move |context| {
            let mut seen = future_seen.lock().expect("waker log must stay available");
            seen.push(context.waker().clone());
            if seen.len() == POLLS {
                Poll::Ready(())
            } else {
                context.waker().wake_by_ref();
                Poll::Pending
            }
        }));

        executor.process_pending_tasks();

        let seen = seen.lock().expect("waker log must stay available");
        assert_eq!(seen.len(), POLLS);
        assert_eq!(executor.stats().tasks_completed, 1);
        assert!(
            seen.iter().all(|waker| waker.will_wake(&seen[0])),
            "a task must present one stable waker identity across polls"
        );
    }

    /// Flags its drop so a test can observe when the task owning it is freed.
    struct DropProbe(Arc<AtomicBool>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn dropping_the_executor_frees_a_task_still_in_the_run_queue() {
        // The task sits in the run queue, which holds the task strongly. The
        // task must hold the queue weakly, else queue -> task -> queue keeps
        // both alive after the executor drops.
        let dropped = Arc::new(AtomicBool::new(false));
        let probe = DropProbe(Arc::clone(&dropped));
        let executor = AsyncExecutor::new().expect("a fresh AsyncExecutor must build");
        let _handle = executor.spawn(async move {
            let _probe = probe;
            futures::future::pending::<()>().await;
        });
        assert!(!dropped.load(Ordering::SeqCst));

        drop(executor);

        assert!(
            dropped.load(Ordering::SeqCst),
            "the queued task's future must drop with its executor"
        );
    }

    #[test]
    fn a_pending_task_lives_and_dies_with_its_last_waker() {
        use std::sync::Mutex;

        // After its first poll the task is not queued; only the waker the
        // future stashed keeps it alive. Dropping that waker must free it,
        // which fails if the task owned a waker of itself (task -> waker ->
        // task).
        let dropped = Arc::new(AtomicBool::new(false));
        let probe = DropProbe(Arc::clone(&dropped));
        let stash = Arc::new(Mutex::new(None::<Waker>));
        let future_stash = Arc::clone(&stash);
        let executor = AsyncExecutor::new().expect("a fresh AsyncExecutor must build");
        let _handle = executor.spawn(futures::future::poll_fn(move |context| {
            let _probe = &probe;
            *future_stash
                .lock()
                .expect("waker stash must stay available") = Some(context.waker().clone());
            Poll::<()>::Pending
        }));

        executor.process_pending_tasks();
        assert!(
            !dropped.load(Ordering::SeqCst),
            "a stashed waker keeps the task alive"
        );

        let waker = stash
            .lock()
            .expect("waker stash must stay available")
            .take()
            .expect("the first poll must stash its waker");
        drop(executor);
        waker.wake_by_ref();
        assert!(
            !dropped.load(Ordering::SeqCst),
            "a wake after the executor dropped must neither panic nor free the task"
        );

        drop(waker);

        assert!(
            dropped.load(Ordering::SeqCst),
            "dropping the last waker must free the pending task"
        );
    }
}
