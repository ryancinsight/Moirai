#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use super::super::HybridExecutor;
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
