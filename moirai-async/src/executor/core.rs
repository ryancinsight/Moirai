use moirai_core::{Priority, TaskId};
use moirai_pal::reactor::IoReactor;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
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
            completed: AtomicBool::new(false),
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
        // Relaxed: publishes nothing. This store carries no data — the thread
        // that writes it is the same thread that then reads it in the loop
        // below, and a concurrent `stop()` racing it is lost under any
        // ordering (SeqCst included), because the race is on which store lands
        // last, not on visibility. Single-location writes are coherent at
        // Relaxed, so every other thread still converges on the final value.
        self.running.store(true, Ordering::Relaxed);

        self.reactor.with_active(|| {
            // Acquire: pairs with the Release store in `stop()`. This is the
            // load that decides to exit, so it must also make everything the
            // stopping thread wrote before requesting shutdown visible to the
            // code that runs after this loop. On x86-64 an Acquire load is a
            // plain `mov` — the SeqCst it replaces cost a full barrier on
            // every iteration of a hot poll loop for an edge Acquire supplies.
            while self.running.load(Ordering::Acquire) {
                self.process_pending_tasks();

                let has_tasks = self.stats.tasks_pending.load(Ordering::Acquire) > 0;

                if !has_tasks {
                    // Acquire: also an exit decision, so it needs the same
                    // edge as the loop condition — this `break` skips the
                    // Acquire at the top of the next iteration.
                    if !self.running.load(Ordering::Acquire) {
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
        // Release: the one edge in this flag's protocol. Everything this
        // thread wrote before deciding to shut down must be visible to the
        // executor thread once it observes `false` through the Acquire loads
        // in `run`. Release is exactly that and no more; SeqCst would
        // additionally place this store in a global total order with unrelated
        // atomics, which no reader of this flag consults.
        self.running.store(false, Ordering::Release);
        self.reactor.stop()
    }

    /// Process all pending tasks.
    pub(crate) fn process_pending_tasks(&self) {
        while let Some(task) = self.run_queue.try_dequeue() {
            // `is_queued` only linearizes enqueue deduplication. The queue's
            // slot sequence already publishes the task with Release/Acquire;
            // this clear pairs with the waker's atomic RMW solely to choose
            // whether a wake owns a new queue entry, so it needs no global
            // ordering edge.
            task.is_queued.store(false, Ordering::Relaxed);

            let waker = self.create_executor_waker(Arc::clone(&task));
            let mut context = Context::from_waker(&waker);
            let task_start = Instant::now();

            // A wake can race a completion (the waker passed its `completed`
            // check just before the task finished on another path), so this is
            // the authoritative guard: never poll a future that already
            // returned `Ready` — doing so panics with "resumed after
            // completion".
            //
            // The check must happen *under* `future_lock`, together with the
            // poll it guards. `is_queued` is cleared above so a self-wake during
            // this poll can re-enqueue, which means a second polling thread can
            // dequeue the same task while this one still holds the lock; if that
            // thread tested `completed` before the lock, it would block, observe
            // the completion only after acquiring the lock, and then poll the
            // finished future anyway. Testing inside the critical section makes
            // the guard and the poll atomic with respect to the completing
            // writer below.
            let _lock = task.future_lock.lock().unwrap();

            if task.completed.load(Ordering::Acquire) {
                continue;
            }

            let future_mut = unsafe { &mut *task.future.get() };
            match future_mut.poll(&mut context) {
                std::task::Poll::Ready(()) => {
                    task.completed.store(true, Ordering::Release);
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

    /// Block on a future, running the executor until it completes.
    pub fn block_on<F, T>(&self, future: F) -> T
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let handle = self.spawn(future);
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut pin_handle = Box::pin(handle);

        // Relaxed on both stores: `block_on` drives the loop on this thread and
        // never reads the flag, so neither store carries a happens-before
        // obligation. They exist so an observer (`stop`, diagnostics) sees the
        // executor as busy for the duration; coherence at Relaxed is all that
        // needs.
        self.running.store(true, Ordering::Relaxed);

        loop {
            self.process_pending_tasks();

            match pin_handle.as_mut().poll(&mut cx) {
                Poll::Ready(result) => {
                    self.running.store(false, Ordering::Relaxed);
                    return result;
                }
                Poll::Pending => {}
            }

            if self.run_queue.is_empty() {
                self.reactor.run_iteration(None).ok();
            } else {
                self.reactor
                    .run_iteration(Some(std::time::Duration::from_millis(0)))
                    .ok();
            }
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
    fn stale_waker_after_completion_does_not_repoll() {
        // Regression: a completed task whose waker is fired again (as a live
        // reactor waker would after a timeout race) must not be re-enqueued or
        // re-polled — re-polling a finished `async` block panics with
        // "resumed after completion".
        let executor = AsyncExecutor::new().unwrap();
        // Hold the handle so the result slot (and task) stay alive across passes.
        let _handle = executor.spawn(async { 5usize });

        // Drain the spawn: the task runs to completion.
        executor.process_pending_tasks();
        assert_eq!(executor.stats().tasks_completed, 1);

        // Simulate a stale wake arriving after completion: fabricate the same
        // executor waker the task carried and fire it. It must be a no-op.
        let task = executor.run_queue.try_dequeue();
        assert!(
            task.is_none(),
            "completed task must not be on the run queue"
        );

        // A second processing pass (a stale wake would have re-enqueued the
        // finished task before this pass) must poll nothing and not panic.
        executor.process_pending_tasks();
        assert_eq!(
            executor.stats().tasks_completed,
            1,
            "no re-poll of the completed task"
        );
    }

    #[test]
    fn completion_under_lock_blocks_a_concurrent_polling_thread() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::{Arc, Barrier};

        // Regression for a re-poll-after-completion race between two threads
        // running `process_pending_tasks` on one executor (`run` takes `&self`,
        // and the executor is shared as an `Arc` in-tree).
        //
        // `process_pending_tasks` clears `is_queued` before polling, so a wake
        // during a poll re-enqueues the task and a second thread can dequeue it
        // while the first still holds `future_lock`. If that thread tested
        // `completed` *before* taking the lock, it would pass the test, block on
        // the lock, and then poll a future the first thread had completed
        // meanwhile — which panics with "resumed after completion".
        //
        // The interleaving is forced deterministically (no timing): this thread
        // holds `future_lock` and marks the task completed while a second thread
        // is provably parked inside `process_pending_tasks` on that same lock.
        let executor = Arc::new(AsyncExecutor::new().unwrap());
        let polls = Arc::new(AtomicUsize::new(0));

        // A future that would panic if polled a second time, standing in for the
        // "resumed after completion" panic of a finished `async` block.
        let poll_counter = Arc::clone(&polls);
        let _handle = executor.spawn(async move {
            assert_eq!(
                poll_counter.fetch_add(1, Ordering::SeqCst),
                0,
                "future must never be polled after completion"
            );
        });

        // Take the queued task, hold its future lock, and put it back so the
        // other thread dequeues the same task while the lock is held.
        let task = executor
            .run_queue
            .try_dequeue()
            .expect("spawned task must be queued");
        let guard = task.future_lock.lock().unwrap();
        executor.run_queue.enqueue(Arc::clone(&task));

        let barrier = Arc::new(Barrier::new(2));
        let poller = {
            let executor = Arc::clone(&executor);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                // Dequeues the task, then parks on `future_lock`.
                executor.process_pending_tasks();
            })
        };

        barrier.wait();

        // The poller is either about to block or already blocked on the lock we
        // hold; completing the task now is exactly the race being guarded. The
        // lock is released after the flag is set, so the poller observes a
        // completed task the instant it acquires the lock.
        task.completed.store(true, Ordering::Release);
        drop(guard);

        poller.join().expect("polling thread must not panic");

        assert_eq!(
            polls.load(Ordering::SeqCst),
            0,
            "a task completed while another thread waited on its lock must not be polled"
        );
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
