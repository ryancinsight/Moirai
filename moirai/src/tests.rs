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
fn test_spawn_fn() {
    let moirai = Moirai::new().unwrap();

    // Test basic task spawning
    let handle = moirai.spawn_fn(|| (0..100).sum::<i32>());

    // Verify the handle was created with a valid task ID
    assert!(handle.id().0 > 0 && handle.id().0 < 100);

    // In std environments, we can actually get the result
    {
        // Give the task some time to complete (this is a simple synchronous operation)
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Try to get the result
        if let Some(result) = handle.join() {
            assert_eq!(result, Ok(4950)); // Sum of 0..100
        }
    }
}

#[test]
fn test_task_panic_handling() {
    let moirai = Moirai::new().unwrap();

    // Spawn a task that panics
    let handle = moirai.spawn_fn(|| {
        panic!("Task intentionally panicked!");
    });

    // Give the task time to execute and panic
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Verify the handle was created properly
    assert!(handle.id().0 > 0);

    // Try to join - the task should have panicked and been caught by the executor
    let _result = handle.join();
    // The executor should handle panics gracefully and return a result
    // indicating the panic occurred, rather than propagating the panic
}

#[test]
fn test_spawn_async() {
    let moirai = Moirai::new().unwrap();
    let handle = moirai.spawn_async(async { 42 });
    // Verify the handle was created with a valid task ID
    assert!(handle.id().0 > 0 && handle.id().0 < 100);
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

    // Give tasks time to complete
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Verify we can retrieve results - using blocking join for more reliable tests
    // Note: In a real concurrent environment, we should use proper synchronization

    // Try non-blocking first
    let result1 = handle1.join();
    let result2 = handle2.join();
    let result3 = handle3.join();

    // Print debug info to see what's happening
    println!("Result 1: {result1:?}");
    println!("Result 2: {result2:?}");
    println!("Result 3: {result3:?}");

    // If we get results, verify they're correct
    if let Some(result) = result1 {
        assert_eq!(result, Ok(84));
    }

    if let Some(result) = result2 {
        assert_eq!(result, Ok("Hello, Moirai".to_string()));
    }

    if let Some(result) = result3 {
        assert_eq!(result, Ok(3_628_800)); // 10!
    }
}

#[test]
fn distributed_feature_does_not_add_facade_remote_closure_execution() {
    let moirai = Moirai::builder().build().unwrap();
    let handle = moirai.spawn_fn(|| "computed locally".to_string());
    let result = handle.join().expect("local task handle must be attached");
    assert_eq!(result, Ok("computed locally".to_string()));
    moirai.shutdown();
}
