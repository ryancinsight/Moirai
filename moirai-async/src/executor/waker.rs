use crate::executor::task::AsyncTask;
use moirai_pal::reactor::IoReactor;
use std::sync::Arc;

pub(super) struct ExecutorWaker {
    pub(super) task: Arc<AsyncTask>,
    pub(super) run_queue: Arc<moirai_utils::queue::LockFreeQueue<Arc<AsyncTask>>>,
    pub(super) reactor: Arc<IoReactor>,
}

impl std::task::Wake for ExecutorWaker {
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
        if self
            .task
            .completed
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return;
        }
        if !self
            .task
            .is_queued
            // `is_queued` is a linearization flag for enqueue deduplication,
            // not a publication channel. The queue's per-slot Release/Acquire
            // sequence publishes the task itself; this RMW only orders the
            // false -> true transition against `process_pending_tasks`'
            // corresponding clear. Relaxed is therefore sufficient and avoids
            // a global ordering edge on every wake.
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            self.run_queue.enqueue(Arc::clone(&self.task));
            let _ = self.reactor.wake();
        }
    }
}
