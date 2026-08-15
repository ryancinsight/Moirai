//! Comprehensive Edge Case Tests for Interleaved Execution Patterns
//!
//! This module implements rigorous edge case testing for scenarios that interleave
//! asynchronous, synchronous, and parallel tasks using unified task management.
//! Tests are designed to validate production-ready behavior under extreme conditions.

use moirai::{Moirai, Priority};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc, Barrier,
};
use std::time::Duration;

/// Constants for test configuration (SSOT principle)
const STRESS_TASK_COUNT: usize = 100;
const CONCURRENT_WORKER_COUNT: usize = 8;
const MIXED_WORKLOAD_SIZE: usize = 50;
const HEAVY_COMPUTATION_ITERATIONS: usize = 1000;
const TASK_COMPLETION_TIMEOUT: Duration = Duration::from_secs(5);

/// Test fixture for interleaved execution testing
struct InterleavedTestFixture {
    runtime: Moirai,
    counter: Arc<AtomicUsize>,
}

impl InterleavedTestFixture {
    fn new() -> Self {
        let runtime = Moirai::builder()
            .worker_threads(CONCURRENT_WORKER_COUNT)
            .build()
            .expect("Failed to create runtime for interleaved testing");
        Self {
            runtime,
            counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn verify_completion(&self, expected_count: usize) {
        let actual = self.counter.load(Ordering::SeqCst);
        assert_eq!(
            actual, expected_count,
            "Task completion mismatch: expected {}, got {}",
            expected_count, actual
        );
    }
}

fn receive_completions(receiver: &mpsc::Receiver<()>, expected: usize, label: &str) {
    for _ in 0..expected {
        receiver
            .recv_timeout(TASK_COMPLETION_TIMEOUT)
            .unwrap_or_else(|error| panic!("{label} did not complete: {error}"));
    }
}

/// Edge Case 1: Rapid Task Type Switching
/// Tests the executor's ability to handle rapid switches between task types
#[test]
fn test_rapid_task_type_switching() {
    let fixture = InterleavedTestFixture::new();
    let counter = fixture.counter.clone();
    let (completion_sender, completion_receiver) = mpsc::channel();
    let mut handles = Vec::with_capacity(MIXED_WORKLOAD_SIZE);

    // Rapidly alternate between different execution patterns
    for i in 0..MIXED_WORKLOAD_SIZE {
        let counter_clone = counter.clone();
        let completion_sender = completion_sender.clone();

        match i % 3 {
            0 => {
                // Synchronous CPU-bound task using spawn_fn
                let handle = fixture.runtime.spawn_fn(move || {
                    // Simulate CPU work
                    let mut sum = 0;
                    for j in 0..HEAVY_COMPUTATION_ITERATIONS {
                        sum += j;
                    }
                    std::hint::black_box(sum);
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    completion_sender
                        .send(())
                        .expect("completion receiver must remain connected");
                });
                handles.push(handle);
            }
            1 => {
                // Blocking-work path without wall-clock synchronization.
                let handle = fixture.runtime.spawn_fn(move || {
                    std::hint::black_box((0..HEAVY_COMPUTATION_ITERATIONS).sum::<usize>());
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    completion_sender
                        .send(())
                        .expect("completion receiver must remain connected");
                });
                handles.push(handle);
            }
            2 => {
                // Parallel computation task
                let handle = fixture.runtime.spawn_fn(move || {
                    // Quick parallel work simulation
                    let result = (0..100).map(|x| x * x).sum::<usize>();
                    std::hint::black_box(result);
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    completion_sender
                        .send(())
                        .expect("completion receiver must remain connected");
                });
                handles.push(handle);
            }
            _ => unreachable!(),
        }
    }
    drop(completion_sender);

    for handle in handles {
        handle
            .join()
            .expect("rapid switching task must join")
            .expect("rapid switching task must complete");
    }
    receive_completions(
        &completion_receiver,
        MIXED_WORKLOAD_SIZE,
        "rapid switching tasks",
    );

    fixture.verify_completion(MIXED_WORKLOAD_SIZE);
}

/// Edge Case 2: Priority Inversion with Mixed Task Types
/// Tests priority handling across different execution contexts
#[test]
fn test_priority_inversion_resistance() {
    let runtime = Moirai::builder()
        .worker_threads(1)
        .build()
        .expect("Failed to create runtime for priority testing");
    let high_priority_counter = Arc::new(AtomicUsize::new(0));
    let low_priority_counter = Arc::new(AtomicUsize::new(0));
    let high_priority_low_count = Arc::new(AtomicUsize::new(usize::MAX));
    let mut handles = Vec::with_capacity(25);
    let (started_sender, started_receiver) = mpsc::channel();
    let (release_sender, release_receiver) = mpsc::channel();

    // Keep one worker so the test exercises queued priority selection rather
    // than depending on the host's logical-processor count.
    let first_counter = low_priority_counter.clone();
    let first_handle = runtime.spawn_fn_with_priority(
        move || {
            started_sender
                .send(())
                .expect("priority test receiver must remain connected");
            release_receiver
                .recv()
                .expect("priority test release must arrive");
            first_counter.fetch_add(1, Ordering::SeqCst);
            "low"
        },
        Priority::Low,
    );
    started_receiver
        .recv()
        .expect("first low-priority task must start");
    handles.push(("low", first_handle));

    for _ in 1..20 {
        let counter = low_priority_counter.clone();
        let handle = runtime.spawn_fn_with_priority(
            move || {
                counter.fetch_add(1, Ordering::SeqCst);
                "low"
            },
            Priority::Low,
        );
        handles.push(("low", handle));
    }

    for _ in 0..5 {
        let counter = high_priority_counter.clone();
        let low_count = low_priority_counter.clone();
        let observed_low_count = high_priority_low_count.clone();
        let handle = runtime.spawn_fn_with_priority(
            move || {
                observed_low_count.fetch_min(low_count.load(Ordering::SeqCst), Ordering::SeqCst);
                counter.fetch_add(1, Ordering::SeqCst);
                "high"
            },
            Priority::High,
        );
        handles.push(("high", handle));
    }
    release_sender
        .send(())
        .expect("priority test worker must remain connected");

    for (expected, handle) in handles {
        let result = handle.join().expect("Priority task should complete");
        assert_eq!(result, Ok(expected));
    }

    assert_eq!(high_priority_counter.load(Ordering::SeqCst), 5);
    assert_eq!(low_priority_counter.load(Ordering::SeqCst), 20);
    let low_completed = high_priority_low_count.load(Ordering::SeqCst);
    assert!(
        low_completed < 20,
        "Priority inversion detected: all low-priority tasks completed before high-priority ones"
    );

    println!(
        "Priority test: high-priority completed while {} low-priority remained",
        20 - low_completed
    );
}

/// Edge Case 3: Resource Contention Under Mixed Load
/// Tests resource sharing between different task types
#[test]
fn test_resource_contention_handling() {
    let fixture = InterleavedTestFixture::new();
    let shared_resource = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(CONCURRENT_WORKER_COUNT));
    let mut handles = Vec::with_capacity(CONCURRENT_WORKER_COUNT);

    // Create contention scenario with mixed task types
    for i in 0..CONCURRENT_WORKER_COUNT {
        let resource = shared_resource.clone();
        let barrier_clone = barrier.clone();
        let counter = fixture.counter.clone();

        let handle = fixture.runtime.spawn_fn(move || {
            barrier_clone.wait();
            for _ in 0..100 {
                resource.fetch_add(1, Ordering::SeqCst);
            }
            counter.fetch_add(1, Ordering::SeqCst);
            i // Return worker id for verification
        });
        handles.push((i, handle));
    }

    for (expected, handle) in handles {
        let result = handle.join().expect("Contention task should complete");
        assert_eq!(result, Ok(expected));
    }

    // Verify resource integrity
    let expected_value = CONCURRENT_WORKER_COUNT * 100;
    let actual_value = shared_resource.load(Ordering::SeqCst);
    assert_eq!(
        actual_value, expected_value,
        "Resource corruption detected: expected {}, got {}",
        expected_value, actual_value
    );

    fixture.verify_completion(CONCURRENT_WORKER_COUNT);
}

/// Edge Case 4: Cascading Task Dependencies
/// Tests complex dependency chains across execution contexts
#[test]
fn test_cascading_dependencies() {
    let runtime = Moirai::builder()
        .worker_threads(32)
        .build()
        .expect("Failed to create runtime for cascading dependencies testing");

    let counter = Arc::new(AtomicUsize::new(0));
    let stage_counters = [
        Arc::new(AtomicUsize::new(0)), // Stage 1: Parallel
        Arc::new(AtomicUsize::new(0)), // Stage 2: Processing
        Arc::new(AtomicUsize::new(0)), // Stage 3: Finalization
    ];
    let (stage1_sender, stage1_receiver) = mpsc::channel();
    let stage1_receiver = Arc::new(std::sync::Mutex::new(stage1_receiver));
    let (stage2_sender, stage2_receiver) = mpsc::channel();
    let stage2_receiver = Arc::new(std::sync::Mutex::new(stage2_receiver));
    let (final_sender, final_receiver) = mpsc::channel();
    let mut handles = Vec::with_capacity(18);

    // Stage 1: Parallel computation producers
    for i in 0..10 {
        let stage_counter = stage_counters[0].clone();
        let stage1_sender = stage1_sender.clone();
        let handle = runtime.spawn_fn(move || {
            // Simulate parallel work
            let result = (0..100).map(|x| x * i).sum::<usize>();
            std::hint::black_box(result);
            stage_counter.fetch_add(1, Ordering::SeqCst);
            stage1_sender
                .send(())
                .expect("stage 1 receiver must remain connected");
        });
        handles.push(handle);
    }
    drop(stage1_sender);

    // Stage 2: Processing, released by five stage-1 completion events.
    for _ in 0..5 {
        let stage1_receiver = stage1_receiver.clone();
        let stage_counter = stage_counters[1].clone();
        let stage2_sender = stage2_sender.clone();
        let handle = runtime.spawn_fn(move || {
            stage1_receiver
                .lock()
                .expect("stage 1 receiver lock must remain healthy")
                .recv()
                .expect("stage 1 completion must arrive");
            stage_counter.fetch_add(1, Ordering::SeqCst);
            stage2_sender
                .send(())
                .expect("stage 2 receiver must remain connected");
        });
        handles.push(handle);
    }
    drop(stage2_sender);

    // Stage 3: Finalizers, released by three stage-2 completion events.
    for _ in 0..3 {
        let stage2_receiver = stage2_receiver.clone();
        let final_counter = counter.clone();
        let final_sender = final_sender.clone();
        let handle = runtime.spawn_fn(move || {
            stage2_receiver
                .lock()
                .expect("stage 2 receiver lock must remain healthy")
                .recv()
                .expect("stage 2 completion must arrive");
            final_counter.fetch_add(1, Ordering::SeqCst);
            final_sender
                .send(())
                .expect("final receiver must remain connected");
        });
        handles.push(handle);
    }
    drop(final_sender);

    receive_completions(&final_receiver, 3, "cascading finalizers");
    for handle in handles {
        handle
            .join()
            .expect("cascading task must join")
            .expect("cascading task must complete");
    }

    // Verify all stages completed appropriately
    assert_eq!(stage_counters[0].load(Ordering::SeqCst), 10);
    assert_eq!(stage_counters[1].load(Ordering::SeqCst), 5);

    let actual = counter.load(Ordering::SeqCst);
    assert_eq!(
        actual, 3,
        "Task completion mismatch: expected {}, got {}",
        3, actual
    );
}

/// Edge Case 5: Burst Load with Mixed Priorities
/// Tests system behavior under sudden load spikes
#[test]
fn test_burst_load_handling() {
    let fixture = InterleavedTestFixture::new();
    let mut handles = Vec::with_capacity(STRESS_TASK_COUNT);

    // Create a burst of mixed tasks at different priorities.
    for i in 0..STRESS_TASK_COUNT {
        let priority = match i % 4 {
            0 => Priority::High,
            1 => Priority::Normal,
            2 => Priority::Low,
            3 => Priority::Normal,
            _ => unreachable!(),
        };

        let handle = fixture.runtime.spawn_fn_with_priority(
            move || {
                // Variable work amount
                let work_amount = match i % 3 {
                    0 => 50,  // Light work
                    1 => 200, // Medium work
                    2 => 500, // Heavy work
                    _ => unreachable!(),
                };

                let mut sum = 0;
                for j in 0..work_amount {
                    sum += j;
                }

                sum
            },
            priority,
        );
        handles.push(handle);
    }

    let actual_work = handles
        .into_iter()
        .map(|handle| {
            handle
                .join()
                .expect("burst task must join")
                .expect("burst task must complete")
        })
        .sum::<usize>();
    let expected_work = (0..STRESS_TASK_COUNT)
        .map(|i| {
            let work_amount = match i % 3 {
                0 => 50,
                1 => 200,
                2 => 500,
                _ => unreachable!(),
            };
            (0..work_amount).sum::<usize>()
        })
        .sum::<usize>();

    assert_eq!(actual_work, expected_work);
}

/// Edge Case 6: Interleaved Error Handling
/// Tests error propagation across different execution contexts
#[test]
fn test_interleaved_error_handling() {
    let fixture = InterleavedTestFixture::new();
    let mut handles = Vec::with_capacity(20);

    // Mix successful and error-handling tasks
    for i in 0..20 {
        let handle = fixture.runtime.spawn_fn(move || match i % 4 {
            0 => "success",
            1 => "sync_success",
            2 => "handled_success",
            3 => "error_handled",
            _ => unreachable!(),
        });
        handles.push(handle);
    }

    let results = handles
        .into_iter()
        .map(|handle| {
            handle
                .join()
                .expect("error-handling task must join")
                .expect("error-handling task must complete")
        })
        .collect::<Vec<_>>();
    let successes = results
        .iter()
        .filter(|result| **result != "error_handled")
        .count();
    let errors = results
        .iter()
        .filter(|result| **result == "error_handled")
        .count();

    assert_eq!(successes, 15);
    assert_eq!(errors, 5);
}
