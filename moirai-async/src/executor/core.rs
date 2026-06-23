use moirai_core::{Priority, TaskId};
use moirai_pal::reactor::IoReactor;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Waker};
use std::time::Instant;

use crate::executor::handle::AsyncHandle;
use crate::executor::result_slot::AsyncResultSlot;
use crate::executor::stats::{AsyncExecutorStats, ExecutorStats};
use crate::executor::task::{AsyncTask, ErasedTaskFuture};
use crate::executor::waker::ExecutorWaker;

/// Native async executor with access to the PAL I/O reactor.
pub struct AsyncExecutor {
    /// Platform-specific I/O reactor
    reactor: Arc<IoReactor>,
    /// Run queue for ready tasks
    run_queue: Arc<moirai_utils::queue::LockFreeQueue<Arc<AsyncTask>>>,
    /// Runtime statistics
    stats: AsyncExecutorStats,
    /// Executor running state
    running: Arc<AtomicBool>,
    /// Monotonic task identifier source.
    next_task_id: AtomicU64,
}

impl AsyncExecutor {
    /// Create a new native async executor with a PAL I/O reactor handle.
    pub fn new() -> std::io::Result<Self> {
        let reactor = Arc::new(IoReactor::new()?);

        Ok(Self {
            reactor,
            run_queue: Arc::new(moirai_utils::queue::LockFreeQueue::new()),
            stats: AsyncExecutorStats::default(),
            running: Arc::new(AtomicBool::new(false)),
            next_task_id: AtomicU64::new(0),
        })
    }

    /// Spawn an async task with default priority.
    pub fn spawn<F, T>(&self, future: F) -> AsyncHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.spawn_with_priority(future, Priority::Normal)
    }

    /// Spawn an async task with specified priority.
    pub fn spawn_with_priority<F, T>(&self, future: F, priority: Priority) -> AsyncHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let task_id = TaskId::new(self.next_task_id.fetch_add(1, Ordering::Relaxed));
        let result_slot = Arc::new(AsyncResultSlot::new());
        let completion_slot = Arc::clone(&result_slot);

        let wrapped_future = async move {
            let result = future.await;
            completion_slot.complete(result);
        };

        let task = Arc::new(AsyncTask {
            task_id,
            future: std::cell::UnsafeCell::new(ErasedTaskFuture::new(wrapped_future)),
            future_lock: std::sync::Mutex::new(()),
            is_queued: AtomicBool::new(true),
            priority,
            created_at: Instant::now(),
        });

        self.run_queue.enqueue(Arc::clone(&task));

        self.stats.tasks_spawned.fetch_add(1, Ordering::Relaxed);
        self.stats.tasks_pending.fetch_add(1, Ordering::Relaxed);

        let _ = self.reactor.wake();

        AsyncHandle {
            task_id,
            result_slot,
        }
    }

    /// Run the native async executor and poll the PAL reactor between task passes.
    pub fn run(&self) -> std::io::Result<()> {
        self.running.store(true, Ordering::SeqCst);

        self.reactor.with_active(|| {
            while self.running.load(Ordering::SeqCst) {
                self.process_pending_tasks();

                let has_tasks = self.stats.tasks_pending.load(Ordering::Acquire) > 0;

                if !has_tasks {
                    if !self.running.load(Ordering::SeqCst) {
                        break;
                    }
                    self.reactor.run_iteration(None)?;
                } else {
                    let run_queue_empty = self.run_queue.is_empty();
                    if run_queue_empty {
                        self.reactor.run_iteration(None)?;
                    } else {
                        self.reactor
                            .run_iteration(Some(std::time::Duration::from_millis(0)))?;
                    }
                }
            }
            Ok(())
        })
    }

    /// Stop the async executor.
    pub fn stop(&self) -> std::io::Result<()> {
        self.running.store(false, Ordering::SeqCst);
        self.reactor.stop()
    }

    /// Process all pending tasks.
    pub(crate) fn process_pending_tasks(&self) {
        while let Some(task) = self.run_queue.try_dequeue() {
            task.is_queued.store(false, Ordering::SeqCst);

            let waker = self.create_executor_waker(Arc::clone(&task));
            let mut context = Context::from_waker(&waker);
            let task_start = Instant::now();

            let _lock = task.future_lock.lock().unwrap();
            let future_mut = unsafe { &mut *task.future.get() };
            match future_mut.poll(&mut context) {
                std::task::Poll::Ready(()) => {
                    self.stats.tasks_completed.fetch_add(1, Ordering::Relaxed);
                    self.stats.tasks_pending.fetch_sub(1, Ordering::Relaxed);

                    let execution_time = task_start.elapsed().as_nanos() as u64;
                    self.stats
                        .total_execution_time_ns
                        .fetch_add(execution_time, Ordering::Relaxed);
                }
                std::task::Poll::Pending => {}
            }
        }
    }

    /// Create an executor-local waker for polling queued futures.
    fn create_executor_waker(&self, task: Arc<AsyncTask>) -> Waker {
        let waker = Arc::new(ExecutorWaker {
            task,
            run_queue: Arc::clone(&self.run_queue),
            reactor: Arc::clone(&self.reactor),
        });
        Waker::from(waker)
    }

    /// Get current executor statistics.
    pub fn stats(&self) -> ExecutorStats {
        ExecutorStats {
            tasks_spawned: self.stats.tasks_spawned.load(Ordering::Relaxed),
            tasks_completed: self.stats.tasks_completed.load(Ordering::Relaxed),
            tasks_pending: self.stats.tasks_pending.load(Ordering::Relaxed),
            total_execution_time_ns: self.stats.total_execution_time_ns.load(Ordering::Relaxed),
            waker_notifications: self.stats.waker_notifications.load(Ordering::Relaxed),
            io_operations: self.stats.io_operations.load(Ordering::Relaxed),
        }
    }

    /// Get access to the underlying I/O reactor for advanced operations.
    pub fn reactor(&self) -> &IoReactor {
        &self.reactor
    }
}

impl Default for AsyncExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create default AsyncExecutor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_async_executor_creation() {
        let executor = AsyncExecutor::new();
        assert!(executor.is_ok());

        let executor = executor.unwrap();
        let stats = executor.stats();
        assert_eq!(stats.tasks_spawned, 0);
        assert_eq!(stats.tasks_completed, 0);
        assert_eq!(stats.tasks_pending, 0);
    }

    #[test]
    fn test_task_spawning() {
        let executor = AsyncExecutor::new().unwrap();

        let _handle = executor.spawn(async { 42 });

        let stats = executor.stats();
        assert_eq!(stats.tasks_spawned, 1);
        assert_eq!(stats.tasks_pending, 1);
    }

    #[test]
    fn test_task_ids_are_unique() {
        let executor = AsyncExecutor::new().unwrap();

        let first = executor.spawn(async { 1usize });
        let second = executor.spawn(async { 2usize });

        assert_ne!(first.id(), second.id());
    }

    #[test]
    fn test_ready_task_completion_publishes_result() {
        use std::task::Poll;

        let executor = AsyncExecutor::new().unwrap();
        let mut handle = Box::pin(executor.spawn(async { 7usize }));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);

        assert!(matches!(handle.as_mut().poll(&mut context), Poll::Pending));

        executor.process_pending_tasks();

        assert_eq!(executor.stats().tasks_completed, 1);
        assert!(matches!(handle.as_mut().poll(&mut context), Poll::Ready(7)));
    }

    #[test]
    fn test_ready_task_completion_wakes_registered_handle() {
        use futures::task::{waker_ref, ArcWake};
        use std::sync::atomic::AtomicUsize;
        use std::task::Poll;

        struct WakeCounter(AtomicUsize);

        impl ArcWake for WakeCounter {
            fn wake_by_ref(arc_self: &Arc<Self>) {
                arc_self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let executor = AsyncExecutor::new().unwrap();
        let mut handle = Box::pin(executor.spawn(async { 11usize }));
        let wake_counter = Arc::new(WakeCounter(AtomicUsize::new(0)));
        let waker = waker_ref(&wake_counter);
        let mut context = Context::from_waker(&waker);

        assert!(matches!(handle.as_mut().poll(&mut context), Poll::Pending));

        executor.process_pending_tasks();

        assert_eq!(wake_counter.0.load(Ordering::SeqCst), 1);
        assert!(matches!(
            handle.as_mut().poll(&mut context),
            Poll::Ready(11)
        ));
    }

    #[test]
    fn test_priority_scheduling() {
        let executor = AsyncExecutor::new().unwrap();

        let _high_priority = executor.spawn_with_priority(async { "high" }, Priority::High);
        let _normal_priority = executor.spawn_with_priority(async { "normal" }, Priority::Normal);
        let _low_priority = executor.spawn_with_priority(async { "low" }, Priority::Low);

        let stats = executor.stats();
        assert_eq!(stats.tasks_spawned, 3);
        assert_eq!(stats.tasks_pending, 3);
    }
}
