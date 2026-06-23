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
        if !self
            .task
            .is_queued
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            self.run_queue.enqueue(Arc::clone(&self.task));
            let _ = self.reactor.wake();
        }
    }
}
