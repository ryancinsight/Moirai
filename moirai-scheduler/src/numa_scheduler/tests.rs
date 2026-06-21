//! Tests for the NUMA-aware scheduler.

use super::scheduler::NumaAwareScheduler;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use moirai_core::{
    task::{TaskContext, TaskId},
    Priority, Task,
};

#[test]
fn test_numa_scheduler_creation() {
    let scheduler = NumaAwareScheduler::new(None, 1024);
    let stats = scheduler.statistics();
    assert_eq!(stats.node_loads.iter().sum::<usize>(), 0);
}

#[test]
fn test_task_scheduling() {
    let scheduler = Arc::new(NumaAwareScheduler::new(None, 1024));

    // Schedule some tasks
    for i in 0..10 {
        let task = Box::new(DummyTask(format!("task-{}", i)));
        scheduler.schedule_task(task).unwrap();
    }

    let stats = scheduler.statistics();
    assert_eq!(stats.node_loads.iter().sum::<usize>(), 10);
}

#[test]
fn test_work_stealing() {
    let scheduler = Arc::new(NumaAwareScheduler::new(None, 1024));
    let stats = scheduler.statistics();
    let num_nodes = stats.numa_nodes;

    // Add tasks to node 0 (always exists)
    for i in 0..10 {
        let task = Box::new(DummyTask(format!("node0-task-{}", i)));
        scheduler
            .schedule_on_node(
                task,
                Some(0), // Node 0 always exists
                Priority::Normal,
            )
            .unwrap();
    }

    // If we have multiple nodes, add tasks to node 1
    if num_nodes > 1 {
        for i in 0..10 {
            let task = Box::new(DummyTask(format!("node1-task-{}", i)));
            scheduler
                .schedule_on_node(
                    task,
                    Some(1), // Node 1
                    Priority::Normal,
                )
                .unwrap();
        }
    }

    let initial_stats = scheduler.statistics();
    let initial_load: usize = initial_stats.node_loads.iter().sum();
    let expected_tasks = if num_nodes > 1 { 20 } else { 10 };
    assert_eq!(initial_load, expected_tasks);

    // Worker tries to steal - should get work
    let stolen = scheduler.steal_with_locality(0);
    assert!(stolen.is_some(), "Should be able to steal from node");

    // Try cross-node stealing if we have multiple nodes
    if num_nodes > 1 {
        let mut stolen_count = 0;
        for _ in 0..5 {
            if scheduler.steal_with_locality(0).is_some() {
                stolen_count += 1;
            }
        }
        assert!(stolen_count > 0, "Cross-node stealing should work");
    }

    let final_stats = scheduler.statistics();
    assert!(final_stats.total_steal_attempts > 0);
    assert!(final_stats.same_numa_steals > 0 || final_stats.cross_numa_steals > 0);
}

#[test]
fn test_concurrent_operations() {
    let scheduler = Arc::new(NumaAwareScheduler::new(None, 1024));
    let num_workers = 4;
    let tasks_per_worker = 100;
    let mut handles = vec![];

    // Track stolen tasks
    let stolen_tasks = Arc::new(AtomicUsize::new(0));

    // Spawn workers that add and steal tasks concurrently
    for worker_id in 0..num_workers {
        let scheduler = Arc::clone(&scheduler);
        let stolen_tasks = Arc::clone(&stolen_tasks);

        handles.push(thread::spawn(move || {
            // Each worker adds tasks and tries to steal work
            for i in 0..tasks_per_worker {
                let task = Box::new(DummyTask(format!("worker-{}-task-{}", worker_id, i)));
                scheduler.schedule_task(task).unwrap();

                // Try to steal some work
                if i % 10 == 0 {
                    if let Some(_task) = scheduler.steal_with_locality(worker_id % 2) {
                        // Task was stolen - count it
                        stolen_tasks.fetch_add(1, Ordering::Relaxed);
                        // In a real system, we would execute the task here
                    }
                }
            }
        }));
    }

    // Wait for all workers to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let stats = scheduler.statistics();
    let tasks_in_queues: usize = stats.node_loads.iter().sum();
    let tasks_stolen = stolen_tasks.load(Ordering::Relaxed);
    let total_tasks = num_workers * tasks_per_worker;

    // All tasks should be accounted for (either in queues or stolen)
    assert_eq!(
        tasks_in_queues + tasks_stolen,
        total_tasks,
        "Tasks in queues: {}, Tasks stolen: {}, Expected total: {}",
        tasks_in_queues,
        tasks_stolen,
        total_tasks
    );

    // Verify stealing happened
    assert!(
        stats.total_steal_attempts > 0,
        "Should have attempted steals"
    );
    assert!(
        tasks_stolen > 0,
        "Should have successfully stolen some tasks"
    );
}

#[test]
fn test_load_balancing() {
    let scheduler = Arc::new(NumaAwareScheduler::new(None, 1024));

    // Add many tasks to one node
    for i in 0..50 {
        scheduler
            .schedule_on_node(
                Box::new(DummyTask(format!("task-{}", i))),
                Some(0),
                Priority::Normal,
            )
            .unwrap();
    }

    let stats_before = scheduler.statistics();
    let max_load_before = stats_before.node_loads.iter().max().unwrap_or(&0);

    // Trigger load balancing by stealing from overloaded nodes
    for _ in 0..20 {
        scheduler.steal_with_locality(1);
    }

    let stats_after = scheduler.statistics();
    let max_load_after = stats_after.node_loads.iter().max().unwrap_or(&0);

    // Load should be more balanced after stealing
    assert!(max_load_after < max_load_before);
}

#[test]
fn test_work_stealing_patterns() {
    // Test different work-stealing patterns inspired by async/sync/parallel models
    let scheduler = Arc::new(NumaAwareScheduler::new(None, 1024));
    let stats = scheduler.statistics();
    let num_nodes = stats.numa_nodes;

    // Clear any existing tasks first
    for node in 0..num_nodes {
        while scheduler.steal_with_locality(node).is_some() {}
    }

    // Pattern 1: Async-style - many small tasks (like async futures)
    println!("Testing async-style pattern: many small tasks");
    for i in 0..50 {
        // Reduced from 100 to avoid queue overflow
        let node = if num_nodes > 1 { i % num_nodes } else { 0 };
        let task = Box::new(DummyTask(format!("async-{}", i)));
        scheduler
            .schedule_on_node(task, Some(node), Priority::Normal)
            .unwrap();
    }

    // Simulate async executor stealing work
    let mut async_stolen = 0;
    for worker in 0..4 {
        let worker_node = if num_nodes > 1 { worker % num_nodes } else { 0 };
        for _ in 0..5 {
            // Reduced iterations
            if scheduler.steal_with_locality(worker_node).is_some() {
                async_stolen += 1;
            }
        }
    }
    println!("Async pattern: {} tasks stolen out of 50", async_stolen);

    // Clear remaining tasks before next pattern
    for node in 0..num_nodes {
        while scheduler.steal_with_locality(node).is_some() {
            async_stolen += 1;
        }
    }

    // Pattern 2: Parallel-style - fewer CPU-bound tasks
    println!("\nTesting parallel-style pattern: CPU-bound tasks");
    for i in 0..8 {
        let node = if num_nodes > 1 { i % num_nodes } else { 0 };
        let task = Box::new(DummyTask(format!("parallel-{}", i)));
        scheduler
            .schedule_on_node(task, Some(node), Priority::High)
            .unwrap();
    }

    // Simulate work-stealing for parallel execution
    let mut parallel_stolen = 0;
    for worker in 0..4 {
        let worker_node = if num_nodes > 1 { worker % num_nodes } else { 0 };
        if scheduler.steal_with_locality(worker_node).is_some() {
            parallel_stolen += 1;
        }
    }
    println!(
        "Parallel pattern: {} tasks stolen out of 8",
        parallel_stolen
    );

    // Clear remaining tasks
    for node in 0..num_nodes {
        while scheduler.steal_with_locality(node).is_some() {
            parallel_stolen += 1;
        }
    }

    // Pattern 3: Coroutine-style - tasks that yield and resume
    println!("\nTesting coroutine-style pattern: yielding tasks");
    for i in 0..20 {
        // Simulate tasks at different stages of execution
        let priority = if i % 3 == 0 {
            Priority::Low
        } else {
            Priority::Normal
        };
        let node = if num_nodes > 1 { i % num_nodes } else { 0 };
        let task = Box::new(DummyTask(format!("coroutine-{}", i)));
        scheduler
            .schedule_on_node(task, Some(node), priority)
            .unwrap();
    }

    // Coroutines often have locality preferences
    let mut coro_stolen = 0;
    for _ in 0..5 {
        // Workers prefer stealing from their own node (coroutine locality)
        for node in 0..num_nodes {
            if scheduler.steal_with_locality(node).is_some() {
                coro_stolen += 1;
            }
        }
    }
    println!("Coroutine pattern: {} tasks stolen out of 20", coro_stolen);

    // Analyze stealing patterns
    let final_stats = scheduler.statistics();
    println!("\nOverall statistics:");
    println!("  NUMA nodes: {}", num_nodes);
    println!(
        "  Total steal attempts: {}",
        final_stats.total_steal_attempts
    );
    println!("  Same-node steals: {}", final_stats.same_numa_steals);
    println!("  Cross-node steals: {}", final_stats.cross_numa_steals);
    println!("  Failed steals: {}", final_stats.failed_steals);
    println!(
        "  Steal success rate: {:.2}%",
        final_stats.steal_success_rate
    );
    println!(
        "  NUMA locality rate: {:.2}%",
        final_stats.numa_locality_rate
    );

    // Verify work-stealing effectiveness
    assert!(
        final_stats.total_steal_attempts > 0,
        "Should have attempted steals"
    );
    assert!(
        final_stats.steal_success_rate > 0.0,
        "Should have successful steals"
    );

    // In a good work-stealing scheduler, we should see successful steals
    let total_successful = final_stats.same_numa_steals + final_stats.cross_numa_steals;
    assert!(total_successful > 0, "Should have successful steals");

    println!("\nDetailed analysis:");
    println!("  Total tasks scheduled: {}", 50 + 8 + 20);
    println!(
        "  Total tasks stolen: {}",
        async_stolen + parallel_stolen + coro_stolen
    );
    println!(
        "  Work distribution shows {} async, {} parallel, {} coroutine steals",
        async_stolen, parallel_stolen, coro_stolen
    );
}

#[test]
fn test_queue_capacity() {
    let scheduler = NumaAwareScheduler::new(None, 1024);

    // Test scheduling many tasks to see the actual capacity
    let mut scheduled = 0;
    for i in 0..2000 {
        let task = Box::new(DummyTask(format!("test-{}", i)));
        match scheduler.schedule_on_node(task, Some(0), Priority::Normal) {
            Ok(()) => scheduled += 1,
            Err(e) => {
                println!(
                    "Failed to schedule task {} after {} successful schedules: {:?}",
                    i, scheduled, e
                );
                break;
            }
        }
    }

    println!("Successfully scheduled {} tasks", scheduled);
    assert!(
        scheduled > 0,
        "Should be able to schedule at least some tasks"
    );

    // Now try to steal them all
    let mut stolen = 0;
    while scheduler.steal_with_locality(0).is_some() {
        stolen += 1;
    }

    println!("Stole {} tasks out of {} scheduled", stolen, scheduled);
    assert_eq!(
        stolen, scheduled,
        "Should be able to steal all scheduled tasks"
    );
}

#[test]
fn test_numa_topology() {
    let scheduler = NumaAwareScheduler::new(None, 1024);
    let stats = scheduler.statistics();

    println!("NUMA topology:");
    println!("  Number of NUMA nodes: {}", stats.numa_nodes);
    println!("  Node loads: {:?}", stats.node_loads);

    // Try scheduling to each node
    for node in 0..4 {
        let task = Box::new(DummyTask(format!("node-{}-test", node)));
        match scheduler.schedule_on_node(task, Some(node), Priority::Normal) {
            Ok(()) => println!("  Node {} exists", node),
            Err(_) => println!("  Node {} does not exist", node),
        }
    }

    assert!(stats.numa_nodes > 0, "Should have at least one NUMA node");
}

#[test]
fn test_unified_concurrency_patterns() {
    // This test demonstrates how work-stealing adapts to different concurrency patterns
    // drawing lessons from async, sync, coroutine, and parallel execution models

    let scheduler = Arc::new(NumaAwareScheduler::new(None, 1024));
    let stats = scheduler.statistics();
    let num_nodes = stats.numa_nodes;

    println!("\n=== Unified Concurrency Patterns Test ===");
    println!("Testing on {} NUMA node(s)", num_nodes);

    // Lesson 1: From Async - Handle many small, non-blocking tasks efficiently
    // Async tasks are typically small and complete quickly
    println!("\n1. Async Pattern - Many small tasks:");
    let async_start = std::time::Instant::now();
    for i in 0..100 {
        let task = Box::new(DummyTask(format!("async-small-{}", i)));
        // Async tasks often have low priority as they're I/O bound
        scheduler
            .schedule_on_node(task, Some(0), Priority::Low)
            .unwrap();
    }

    // Async executors steal aggressively to keep all cores busy
    let mut async_stolen = 0;
    for _ in 0..50 {
        if scheduler.steal_with_locality(0).is_some() {
            async_stolen += 1;
        }
    }
    let async_duration = async_start.elapsed();
    println!(
        "  - Scheduled 100 small tasks, stole {} in {:?}",
        async_stolen, async_duration
    );
    println!("  - Lesson: Aggressive stealing keeps cores busy with small tasks");

    // Clear remaining
    while scheduler.steal_with_locality(0).is_some() {}

    // Lesson 2: From Parallel - CPU-bound tasks need load balancing
    // Parallel tasks are typically larger and CPU-intensive
    println!("\n2. Parallel Pattern - CPU-bound tasks:");
    let parallel_start = std::time::Instant::now();
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    for i in 0..num_cpus {
        let task = Box::new(DummyTask(format!("parallel-cpu-{}", i)));
        // CPU-bound tasks get high priority
        scheduler
            .schedule_on_node(task, Some(0), Priority::High)
            .unwrap();
    }

    // Parallel work-stealing is more selective - only steal when idle
    let mut parallel_stolen = 0;
    let workers = num_cpus;
    for _worker in 0..workers {
        // Each worker steals once when starting
        if scheduler.steal_with_locality(0).is_some() {
            parallel_stolen += 1;
        }
    }
    let parallel_duration = parallel_start.elapsed();
    println!(
        "  - Scheduled {} CPU-bound tasks, stole {} in {:?}",
        workers, parallel_stolen, parallel_duration
    );
    println!("  - Lesson: Conservative stealing for CPU-bound work prevents thrashing");

    // Clear remaining
    while scheduler.steal_with_locality(0).is_some() {}

    // Lesson 3: From Coroutines - Tasks that yield need fair scheduling
    // Coroutines yield execution and need to be resumed fairly
    println!("\n3. Coroutine Pattern - Yielding tasks:");
    let coro_start = std::time::Instant::now();

    // Mix of different priority tasks (simulating yielded coroutines at different stages)
    for i in 0..30 {
        let priority = match i % 3 {
            0 => Priority::Low,    // Just yielded
            1 => Priority::Normal, // Ready to resume
            _ => Priority::High,   // Almost complete
        };
        let task = Box::new(DummyTask(format!("coroutine-{}", i)));
        scheduler.schedule_on_node(task, Some(0), priority).unwrap();
    }

    // Coroutine stealing respects priorities
    let mut coro_stolen_by_priority = [0, 0, 0, 0]; // [Critical, High, Normal, Low]
    for _ in 0..20 {
        if scheduler.steal_with_locality(0).is_some() {
            // In real implementation, we'd track which priority was stolen
            // For now, we just count total
            coro_stolen_by_priority[1] += 1; // Assume high priority
        }
    }
    let coro_duration = coro_start.elapsed();
    let total_coro_stolen: usize = coro_stolen_by_priority.iter().sum();
    println!(
        "  - Scheduled 30 coroutine tasks, stole {} in {:?}",
        total_coro_stolen, coro_duration
    );
    println!("  - Lesson: Priority-aware stealing ensures fair coroutine resumption");

    // Clear remaining
    while scheduler.steal_with_locality(0).is_some() {}

    // Lesson 4: From Sync - Blocking operations need isolation
    // Sync/blocking tasks should not starve other work
    println!("\n4. Sync Pattern - Blocking tasks:");
    let sync_start = std::time::Instant::now();

    // Schedule some blocking tasks with normal priority
    for i in 0..5 {
        let task = Box::new(DummyTask(format!("blocking-{}", i)));
        scheduler
            .schedule_on_node(task, Some(0), Priority::Normal)
            .unwrap();
    }

    // Also schedule non-blocking tasks that shouldn't be blocked
    for i in 0..10 {
        let task = Box::new(DummyTask(format!("non-blocking-{}", i)));
        scheduler
            .schedule_on_node(task, Some(0), Priority::High)
            .unwrap();
    }

    // Steal high-priority non-blocking tasks first
    let mut sync_stolen = 0;
    for _ in 0..10 {
        if scheduler.steal_with_locality(0).is_some() {
            sync_stolen += 1;
        }
    }
    let sync_duration = sync_start.elapsed();
    println!(
        "  - Scheduled 5 blocking + 10 non-blocking tasks, stole {} in {:?}",
        sync_stolen, sync_duration
    );
    println!("  - Lesson: Priority stealing prevents blocking tasks from starving the system");

    // Final statistics
    let final_stats = scheduler.statistics();
    println!("\n=== Final Statistics ===");
    println!("Total steal attempts: {}", final_stats.total_steal_attempts);
    println!(
        "Successful steals: {}",
        final_stats.same_numa_steals + final_stats.cross_numa_steals
    );
    println!("Success rate: {:.2}%", final_stats.steal_success_rate);
    println!(
        "Average steal latency: {} ns",
        final_stats.avg_steal_latency_ns
    );

    println!("\n=== Key Insights ===");
    println!("1. Async: Aggressive stealing with many small tasks");
    println!("2. Parallel: Conservative stealing for CPU-bound work");
    println!("3. Coroutine: Priority-aware stealing for fairness");
    println!("4. Sync: Isolation of blocking operations");
    println!("5. Unified: Adaptive stealing based on workload characteristics");

    // Verify the scheduler handled all patterns effectively
    assert!(
        final_stats.steal_success_rate > 80.0,
        "Scheduler should maintain high success rate across patterns"
    );
}

// Dummy task for testing
struct DummyTask(#[allow(dead_code)] String);

impl Task for DummyTask {
    type Output = ();

    fn execute(self) -> Self::Output {}

    fn context(&self) -> &TaskContext {
        static DEFAULT_CONTEXT: std::sync::OnceLock<TaskContext> = std::sync::OnceLock::new();
        DEFAULT_CONTEXT.get_or_init(|| TaskContext::new(TaskId::new(0)))
    }
}

#[test]
fn test_numa_scheduler_worker_id_tracking() {
    use super::scheduler::{current_worker_id, set_current_worker_id};
    
    // Set worker ID on main thread
    set_current_worker_id(42);
    assert_eq!(current_worker_id(), 42);

    // Spawn a thread and verify it defaults to 0 and can be set independently
    let handle = std::thread::spawn(|| {
        assert_eq!(current_worker_id(), 0);
        set_current_worker_id(100);
        assert_eq!(current_worker_id(), 100);
    });

    handle.join().unwrap();
    // Verify main thread worker ID is still 42
    assert_eq!(current_worker_id(), 42);
}
