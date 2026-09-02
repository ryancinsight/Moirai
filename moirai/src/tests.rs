#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::*;

#[test]
fn test_moirai_creation() {
    let moirai = Moirai::new().unwrap();
    assert!(moirai.worker_count() > 0);
}

#[test]
fn test_builder() {
    let moirai = Moirai::builder()
        .worker_threads(4)
        .async_threads(2)
        .build()
        .unwrap();

    assert_eq!(moirai.worker_count(), 4);
}

#[test]
fn local_queue_builder_policy_reaches_executor_configuration() {
    let moirai = Moirai::builder()
        .worker_threads(1)
        .local_queue_initial_capacity(17)
        .build()
        .unwrap();

    assert_eq!(moirai.executor.config().local_queue_initial_capacity, 17);
    moirai.shutdown();
}

#[test]
fn test_spawn_fn() {
    let moirai = Moirai::new().unwrap();

    let handle = moirai.spawn_fn(|| (0..100).sum::<i32>());

    assert!(handle.id().0 > 0 && handle.id().0 < 100);

    moirai.join().unwrap();
    assert_eq!(
        handle.join().expect("spawned task must retain a result"),
        Ok(4950)
    );
    moirai.shutdown();
}

#[test]
fn test_task_panic_handling() {
    let moirai = Moirai::new().unwrap();

    let handle = moirai.spawn_fn(|| {
        panic!("Task intentionally panicked!");
    });

    assert!(handle.id().0 > 0);

    moirai.join().unwrap();
    assert!(matches!(handle.join(), Some(Err(_))));
    moirai.shutdown();
}

#[test]
fn test_spawn_async() {
    let moirai = Moirai::new().unwrap();
    let handle = moirai.spawn_async(async { 42 });
    assert!(handle.id().0 > 0 && handle.id().0 < 100);
    moirai.join().unwrap();
    assert_eq!(
        handle.join().expect("async task must retain a result"),
        Ok(42)
    );
    moirai.shutdown();
}

#[test]
fn test_scope_completes_borrowed_jobs() {
    let moirai = Moirai::builder().worker_threads(2).build().unwrap();
    let sum = std::sync::atomic::AtomicUsize::new(0);

    moirai
        .scope(|scope| {
            for value in 1..=32 {
                let sum = &sum;
                scope.spawn(move |_| {
                    sum.fetch_add(value, std::sync::atomic::Ordering::Relaxed);
                })?;
            }
            Ok(())
        })
        .unwrap();

    assert_eq!(sum.load(std::sync::atomic::Ordering::Relaxed), 528);
    moirai.shutdown();
}

#[test]
fn test_indexed_fan_out_completes_borrowed_jobs() {
    let moirai = Moirai::builder().worker_threads(2).build().unwrap();
    let sum = std::sync::atomic::AtomicUsize::new(0);

    moirai
        .for_each_indexed(32, |index| {
            sum.fetch_add(index + 1, std::sync::atomic::Ordering::Relaxed);
        })
        .unwrap();

    assert_eq!(sum.load(std::sync::atomic::Ordering::Relaxed), 528);
    moirai.shutdown();
}

#[test]
fn test_indexed_map_reduce_returns_value() {
    let moirai = Moirai::builder().worker_threads(2).build().unwrap();

    let sum = moirai
        .map_reduce_indexed(32, 0usize, |index| index + 1, usize::wrapping_add)
        .unwrap();

    assert_eq!(sum, 528);
    moirai.shutdown();
}

#[test]
fn indexed_cpu_work_uses_compute_workers() {
    use std::sync::atomic::{AtomicBool, Ordering};

    fn observe_lane(saw_compute: &AtomicBool, saw_blocking: &AtomicBool) {
        let current = std::thread::current();
        let Some(name) = current.name() else {
            return;
        };
        saw_compute.fetch_or(name.contains("-worker-"), Ordering::Relaxed);
        saw_blocking.fetch_or(name.contains("-blocking-"), Ordering::Relaxed);
    }

    let moirai = Moirai::builder().worker_threads(2).build().unwrap();
    let saw_compute = AtomicBool::new(false);
    let saw_blocking = AtomicBool::new(false);

    moirai
        .for_each_indexed(4_096, |_| observe_lane(&saw_compute, &saw_blocking))
        .unwrap();
    let reduced = moirai
        .map_reduce_indexed(
            4_096,
            0usize,
            |index| {
                observe_lane(&saw_compute, &saw_blocking);
                index + 1
            },
            usize::wrapping_add,
        )
        .unwrap();

    assert_eq!(reduced, 4_096 * 4_097 / 2);
    assert!(
        saw_compute.load(Ordering::Relaxed),
        "indexed CPU work must reach the compute-worker pool"
    );
    assert!(
        !saw_blocking.load(Ordering::Relaxed),
        "indexed CPU work must not initialize or execute on the blocking lane"
    );
    moirai.shutdown();
}

#[test]
fn test_join_waits_for_public_spawned_tasks() {
    let moirai = Moirai::builder().worker_threads(2).build().unwrap();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));

    let b = barrier.clone();
    let handle1 = moirai.spawn_fn(move || {
        b.wait();
        1
    });
    let b = barrier.clone();
    let handle2 = moirai.spawn_fn(move || {
        b.wait();
        2
    });

    // Verify there is active/queued work
    assert!(moirai.has_work());

    // Release the tasks from the barrier
    barrier.wait();

    moirai.join().unwrap();
    assert!(!moirai.has_work());

    let results = vec![
        handle1.join().unwrap().unwrap(),
        handle2.join().unwrap().unwrap(),
    ];
    assert_eq!(results, vec![1, 2]);
    moirai.shutdown();
}

#[test]
fn test_repeated_public_spawn_join_completes() {
    let moirai = Moirai::builder().worker_threads(4).build().unwrap();

    for value in 0..1_048_576usize {
        let handle = moirai.spawn_fn(move || value.wrapping_add(1));
        assert_eq!(handle.join().unwrap().unwrap(), value.wrapping_add(1));
    }

    moirai.shutdown();
}

#[test]
fn test_global_runtime() {
    let runtime1 = global();
    let runtime2 = global();

    // Should be the same instance
    assert!(std::ptr::eq(runtime1, runtime2));
}

#[test]
fn test_global_spawn() {
    let handle = spawn_fn(|| "hello world");
    // For now, we'll just test that the handle was created (task ID should be valid)
    assert!(handle.id().0 < 100); // Reasonable upper bound for task IDs in tests
}

#[test]
fn test_task_with_priority() {
    let moirai = Moirai::new().unwrap();

    // Create a task with high priority
    let _context = TaskContext::new(TaskId::new(42))
        .with_priority(Priority::High)
        .with_name("test_task");

    let task = moirai_core::task::TaskBuilder::new()
        .with_id(TaskId::new(0))
        .build(|| "high priority task");
    let handle = moirai.spawn_with_priority(task, Priority::High);

    // Verify the handle was created with a valid task ID
    assert!(handle.id().0 > 0 && handle.id().0 < 100);
}

#[test]
fn test_task_builder() {
    let task = TaskBuilder::new()
        .priority(Priority::High)
        .name("test_task")
        .build(|| 42);

    assert_eq!(task.context().priority, Priority::High);
    assert_eq!(task.context().name, Some("test_task"));
    assert_eq!(task.execute(), 42);
}

#[test]
fn test_task_chaining() {
    let task = moirai_core::task::TaskBuilder::new()
        .with_id(TaskId::new(1))
        .build(|| 21);

    let chained = task.then(|x| x * 2);
    assert_eq!(chained.execute(), 42);
}

#[test]
fn test_task_mapping() {
    let task = moirai_core::task::TaskBuilder::new()
        .with_id(TaskId::new(1))
        .build(|| 21);

    let mapped = task.map(|x| x * 2);
    assert_eq!(mapped.execute(), 42);
}

#[test]
fn test_task_result_retrieval() {
    let moirai = Moirai::new().unwrap();

    // Test simple computation
    let handle1 = moirai.spawn_fn(|| 42 * 2);

    // Test string computation
    let handle2 = moirai.spawn_fn(|| format!("Hello, {}", "Moirai"));

    // Test complex computation
    let handle3 = moirai.spawn_fn(|| (1..=10).product::<i32>());

    // At least verify the handles were created with valid task IDs
    assert!(handle1.id().0 < 100);
    assert!(handle2.id().0 < 100);
    assert!(handle3.id().0 < 100);

    moirai.join().unwrap();
    assert_eq!(
        handle1.join().expect("numeric task must retain a result"),
        Ok(84)
    );
    assert_eq!(
        handle2.join().expect("string task must retain a result"),
        Ok("Hello, Moirai".to_string())
    );
    assert_eq!(
        handle3.join().expect("product task must retain a result"),
        Ok(3_628_800)
    );
    moirai.shutdown();
}

#[test]
fn distributed_feature_does_not_add_facade_remote_closure_execution() {
    let moirai = Moirai::builder().build().unwrap();
    let handle = moirai.spawn_fn(|| "computed locally".to_string());
    let result = handle.join().expect("local task handle must be attached");
    assert_eq!(result, Ok("computed locally".to_string()));
    moirai.shutdown();
}
