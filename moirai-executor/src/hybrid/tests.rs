#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use super::super::HybridExecutor;
    use crate::SyncTask;
    use moirai_core::{
        executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner, TaskStatus},
        task::TaskBuilder,
        Priority,
    };

    #[test]
    fn spawn_blocking_returns_value_and_updates_status() {
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 2,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let handle = executor.spawn_blocking(|| 21 * 2).unwrap();
        let id = handle.id();
        let result = handle.join().unwrap().unwrap();

        assert_eq!(result, 42);
        assert_eq!(executor.task_status(id), Some(TaskStatus::Completed));
        executor.shutdown();
    }

    #[test]
    fn scoped_chunks_complete_before_inherent_shutdown() {
        const LEN: usize = 257;
        const CHUNK: usize = 7;
        const ROUNDS: usize = 64;

        let mut executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 2,
            ..ExecutorConfig::default()
        })
        .unwrap();

        for round in 0..ROUNDS {
            let mut values = vec![usize::MAX; LEN];
            executor
                .scope::<SyncTask, _>(|scope| {
                    for (chunk_index, chunk) in values.chunks_mut(CHUNK).enumerate() {
                        let first = chunk_index * CHUNK;
                        scope.spawn(move |_| {
                            for (offset, value) in chunk.iter_mut().enumerate() {
                                *value = first + offset;
                            }
                        })?;
                    }
                    Ok(())
                })
                .unwrap();

            for (index, value) in values.into_iter().enumerate() {
                assert_eq!(value, index, "round {round} lost logical slot {index}");
            }
        }
        HybridExecutor::shutdown(&mut executor).unwrap();
    }

    #[test]
    fn spawn_blocking_reports_panicked_result() {
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let handle = executor
            .spawn_blocking(|| -> usize { panic!("blocking task panic") })
            .unwrap();

        assert_eq!(handle.join(), Some(Err(moirai_core::TaskError::Panicked)));
        executor.shutdown();
    }

    #[test]
    fn spawn_detached_runs_every_task_and_drains_on_shutdown() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 4,
            ..ExecutorConfig::default()
        })
        .unwrap();

        const TASKS: usize = 256;
        let counter = Arc::new(AtomicUsize::new(0));
        for _ in 0..TASKS {
            let c = Arc::clone(&counter);
            // Returns `()`: no handle, no `Arc<TaskResultSlot>` allocated.
            executor
                .spawn_detached(move || {
                    c.fetch_add(1, Ordering::Relaxed);
                })
                .unwrap();
        }

        // `shutdown` drains all pending work before returning, so every detached
        // closure must have executed exactly once.
        executor.shutdown();
        assert_eq!(counter.load(Ordering::Relaxed), TASKS);
    }

    #[test]
    fn spawn_detached_isolates_panics() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        // A panicking detached task must not abort its worker thread.
        executor
            .spawn_detached(|| panic!("detached task panic"))
            .unwrap();

        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        executor
            .spawn_detached(move || {
                c.fetch_add(1, Ordering::Relaxed);
            })
            .unwrap();

        executor.shutdown();
        // The single worker survived the panic and ran the following task.
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn spawn_async_uses_unified_scheduler() {
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 2,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let handle = executor.spawn_async(async { 7usize }).unwrap();
        let result = handle.join().unwrap().unwrap();

        assert_eq!(result, 7);
        assert_eq!(executor.worker_count(), 2);
        executor.shutdown();
    }

    #[test]
    fn spawn_async_requeues_after_wake_without_blocking_worker() {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            mpsc, Arc, Mutex,
        };
        use std::task::Waker;
        use std::time::{Duration, Instant};

        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let ready = Arc::new(AtomicBool::new(false));
        let waker_slot = Arc::new(Mutex::new(None::<Waker>));
        let ready_for_future = Arc::clone(&ready);
        let waker_for_future = Arc::clone(&waker_slot);
        let handle = executor
            .spawn_async(async {
                std::future::poll_fn(move |cx| {
                    if ready_for_future.load(Ordering::Acquire) {
                        std::task::Poll::Ready(21usize)
                    } else {
                        *waker_for_future.lock().unwrap() = Some(cx.waker().clone());
                        std::task::Poll::Pending
                    }
                })
                .await
            })
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(1);
        let waker = loop {
            if let Some(waker) = waker_slot.lock().unwrap().take() {
                break waker;
            }

            assert!(
                Instant::now() < deadline,
                "async future must publish a waker before timeout"
            );
            std::thread::sleep(Duration::from_millis(1));
        };

        let (ran_sender, ran_receiver) = mpsc::channel();
        let independent = executor
            .spawn_blocking(move || {
                ran_sender.send(()).unwrap();
                13usize
            })
            .unwrap();

        ran_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("pending async future must not block the only worker");

        ready.store(true, Ordering::Release);
        waker.wake();

        assert_eq!(independent.join().unwrap().unwrap(), 13);
        assert_eq!(handle.join().unwrap().unwrap(), 21);
        executor.shutdown();
    }

    #[test]
    fn spawn_async_completes_single_self_wake() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let poll_count = Arc::new(AtomicUsize::new(0));
        let poll_count_for_future = Arc::clone(&poll_count);
        let handle = executor
            .spawn_async(async move {
                std::future::poll_fn(move |context| {
                    match poll_count_for_future.fetch_add(1, Ordering::AcqRel) {
                        0 => {
                            context.waker().wake_by_ref();
                            std::task::Poll::Pending
                        }
                        previous => std::task::Poll::Ready(previous + 1),
                    }
                })
                .await
            })
            .unwrap();

        assert_eq!(handle.join().unwrap().unwrap(), 2);
        assert_eq!(poll_count.load(Ordering::Acquire), 2);
        executor.shutdown();
    }

    #[test]
    fn priority_spawn_preserves_task_result() {
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let task = TaskBuilder::new()
            .priority(Priority::Critical)
            .build(|| 11usize);
        let handle = executor
            .spawn_with_priority(task, Priority::Critical, Some(0))
            .unwrap();

        assert_eq!(handle.join().unwrap().unwrap(), 11);
        executor.shutdown();
    }

    /// Waker that counts how many times it is woken.
    struct CountingWake(std::sync::atomic::AtomicUsize);

    impl std::task::Wake for CountingWake {
        fn wake(self: std::sync::Arc<Self>) {
            self.wake_by_ref();
        }
        fn wake_by_ref(self: &std::sync::Arc<Self>) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
    }

    /// Occupy the single worker with a job gated on a channel; returns the
    /// release sender and blocks until the gate job has started.
    fn gate_single_worker(
        executor: &HybridExecutor,
    ) -> (
        std::sync::mpsc::Sender<()>,
        moirai_core::task::TaskHandle<()>,
    ) {
        let (release_sender, release_receiver) = std::sync::mpsc::channel::<()>();
        let (started_sender, started_receiver) = std::sync::mpsc::channel::<()>();
        let handle = executor
            .spawn_blocking(move || {
                started_sender.send(()).unwrap();
                release_receiver
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .expect("gate must be released before the test deadline");
            })
            .unwrap();
        started_receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("gate task must start");
        (release_sender, handle)
    }

    #[test]
    fn cancel_queued_task_skips_body_and_completes_cancelled() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let (release, gate_handle) = gate_single_worker(&executor);

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_in_task = Arc::clone(&counter);
        let handle = executor
            .spawn_blocking(move || {
                counter_in_task.fetch_add(1, Ordering::Relaxed);
                7usize
            })
            .unwrap();
        let id = handle.id();

        // The single worker is busy with the gate, so the task is still queued.
        executor.cancel_task(id).unwrap();
        release.send(()).unwrap();

        // The handle resolves to the cancelled outcome and the body never ran.
        assert_eq!(handle.join(), Some(Err(moirai_core::TaskError::Cancelled)));
        assert_eq!(counter.load(Ordering::Relaxed), 0);
        assert_eq!(executor.task_status(id), Some(TaskStatus::Cancelled));
        assert_eq!(
            executor
                .metrics()
                .tasks_cancelled
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );

        gate_handle.join().unwrap().unwrap();
        executor.shutdown();
    }

    #[test]
    fn cancel_queued_async_task_skips_future_and_completes_cancelled() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let (release, gate_handle) = gate_single_worker(&executor);

        let polls = Arc::new(AtomicUsize::new(0));
        let polls_in_future = Arc::clone(&polls);
        let handle = executor
            .spawn_async(async move {
                polls_in_future.fetch_add(1, Ordering::Relaxed);
                3usize
            })
            .unwrap();
        let id = handle.id();

        executor.cancel_task(id).unwrap();
        release.send(()).unwrap();

        assert_eq!(handle.join(), Some(Err(moirai_core::TaskError::Cancelled)));
        assert_eq!(polls.load(Ordering::Relaxed), 0);
        assert_eq!(executor.task_status(id), Some(TaskStatus::Cancelled));

        gate_handle.join().unwrap().unwrap();
        executor.shutdown();
    }

    #[test]
    fn cancel_completed_task_is_noop_ok_and_unknown_id_errors() {
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let handle = executor.spawn_blocking(|| 5usize).unwrap();
        let id = handle.id();
        assert_eq!(handle.join().unwrap().unwrap(), 5);

        // Already completed: no-op Ok, status remains Completed.
        executor.cancel_task(id).unwrap();
        assert_eq!(executor.task_status(id), Some(TaskStatus::Completed));

        assert_eq!(
            executor.cancel_task(moirai_core::TaskId::new(u64::MAX / 2)),
            Err(moirai_core::error::ExecutorError::SpawnFailed(
                moirai_core::TaskError::InvalidOperation
            ))
        );
        executor.shutdown();
    }

    #[test]
    fn cancel_running_task_is_not_preempted() {
        // Contract: cancelling a task that already started has no effect — the
        // body runs to completion and the result is the real value.
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let (release_sender, release_receiver) = std::sync::mpsc::channel::<()>();
        let (started_sender, started_receiver) = std::sync::mpsc::channel::<()>();
        let handle = executor
            .spawn_blocking(move || {
                started_sender.send(()).unwrap();
                release_receiver
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .unwrap();
                11usize
            })
            .unwrap();
        started_receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();

        let id = handle.id();
        executor.cancel_task(id).unwrap();
        release_sender.send(()).unwrap();

        assert_eq!(handle.join(), Some(Ok(11)));
        assert_eq!(executor.task_status(id), Some(TaskStatus::Completed));
        executor.shutdown();
    }

    #[test]
    fn wait_for_task_is_woken_by_completion_not_polling() {
        use std::future::Future;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::task::{Context, Poll, Waker};

        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let (release, gate_handle) = gate_single_worker(&executor);
        let handle = executor.spawn_blocking(|| 9usize).unwrap();
        let id = handle.id();

        let wake = Arc::new(CountingWake(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&wake));
        let mut context = Context::from_waker(&waker);
        let mut wait = std::pin::pin!(executor.wait_for_task(id, None));

        // One poll registers the completion waker; the task is still queued.
        assert!(wait.as_mut().poll(&mut context).is_pending());
        assert_eq!(wake.0.load(Ordering::Acquire), 0);

        release.send(()).unwrap();
        assert_eq!(handle.join().unwrap().unwrap(), 9);

        // Completion wakes the registered waker exactly once (no poll loop);
        // observe the wake with a bounded deadline.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while wake.0.load(Ordering::Acquire) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "completion must wake the registered waiter"
            );
            std::thread::yield_now();
        }
        assert_eq!(wake.0.load(Ordering::Acquire), 1);
        assert_eq!(wait.as_mut().poll(&mut context), Poll::Ready(Ok(())));

        gate_handle.join().unwrap().unwrap();
        executor.shutdown();
    }

    #[test]
    fn wait_for_task_timeout_expires_and_unknown_task_errors() {
        use std::future::Future;
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;
        use std::task::{Context, Poll, Waker};

        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        // Unknown task resolves immediately with a typed error.
        let waker = Waker::from(Arc::new(CountingWake(AtomicUsize::new(0))));
        let mut context = Context::from_waker(&waker);
        let mut unknown =
            std::pin::pin!(executor.wait_for_task(moirai_core::TaskId::new(u64::MAX / 2), None));
        assert_eq!(
            unknown.as_mut().poll(&mut context),
            Poll::Ready(Err(moirai_core::error::ExecutorError::SpawnFailed(
                moirai_core::TaskError::InvalidOperation
            )))
        );

        // A never-completing task expires with the typed timeout once the
        // deadline passes (deadline is checked on every poll).
        let (release, gate_handle) = gate_single_worker(&executor);
        let handle = executor.spawn_blocking(|| 1usize).unwrap();
        let mut wait = std::pin::pin!(
            executor.wait_for_task(handle.id(), Some(std::time::Duration::from_millis(40)))
        );
        assert!(wait.as_mut().poll(&mut context).is_pending());
        std::thread::sleep(std::time::Duration::from_millis(80));
        assert_eq!(
            wait.as_mut().poll(&mut context),
            Poll::Ready(Err(moirai_core::error::ExecutorError::SpawnFailed(
                moirai_core::TaskError::Timeout
            )))
        );

        release.send(()).unwrap();
        assert_eq!(handle.join().unwrap().unwrap(), 1);
        gate_handle.join().unwrap().unwrap();
        executor.shutdown();
    }

    #[test]
    fn task_stats_reports_recorded_spawn_priority() {
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let task = TaskBuilder::new()
            .priority(Priority::Critical)
            .build(|| 2usize);
        let critical = executor
            .spawn_with_priority(task, Priority::Critical, None)
            .unwrap();
        let normal = executor.spawn_blocking(|| 4usize).unwrap();
        let critical_id = critical.id();
        let normal_id = normal.id();

        assert_eq!(critical.join().unwrap().unwrap(), 2);
        assert_eq!(normal.join().unwrap().unwrap(), 4);

        let critical_stats = executor.task_stats(critical_id).unwrap();
        assert_eq!(critical_stats.priority, Priority::Critical);
        assert_eq!(critical_stats.status, TaskStatus::Completed);

        let normal_stats = executor.task_stats(normal_id).unwrap();
        assert_eq!(normal_stats.priority, Priority::Normal);
        executor.shutdown();
    }

    #[test]
    fn spawn_honors_task_context_priority() {
        // `spawn` must record the task's own context priority (previously only
        // `spawn_with_priority` did).
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let task = TaskBuilder::new().priority(Priority::High).build(|| 8usize);
        let handle = executor.spawn(task).unwrap();
        let id = handle.id();
        assert_eq!(handle.join().unwrap().unwrap(), 8);
        assert_eq!(executor.task_stats(id).unwrap().priority, Priority::High);
        executor.shutdown();
    }

    #[test]
    fn shutdown_timeout_bounds_the_callers_wait() {
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let (release_sender, release_receiver) = std::sync::mpsc::channel::<()>();
        let (started_sender, started_receiver) = std::sync::mpsc::channel::<()>();
        executor
            .spawn_detached(move || {
                started_sender.send(()).unwrap();
                release_receiver
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .unwrap();
            })
            .unwrap();
        started_receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();

        // The worker is blocked, so a full drain cannot finish; the call must
        // return once the bound elapses while the drain continues behind it.
        let start = std::time::Instant::now();
        executor.shutdown_timeout(std::time::Duration::from_millis(50));
        assert!(
            start.elapsed() < std::time::Duration::from_secs(5),
            "shutdown_timeout must bound the caller's wait"
        );
        assert!(executor.is_shutting_down());

        // Release the worker so the background drain and drop complete.
        release_sender.send(()).unwrap();
    }

    #[test]
    fn join_waits_for_public_result_tasks_without_shutdown() {
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 2,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let handles = (0..8)
            .map(|value| executor.spawn_blocking(move || value + 1).unwrap())
            .collect::<Vec<_>>();

        assert!(executor.has_work());
        executor.join().unwrap();
        assert!(!executor.has_work());

        let results = handles
            .into_iter()
            .map(|handle| handle.join().unwrap().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(results, (1..=8).collect::<Vec<_>>());
        executor.shutdown();
    }
}
