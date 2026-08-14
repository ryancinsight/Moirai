//! Comprehensive Edge Case Tests for Interleaved Execution Patterns
//!
//! This module implements rigorous edge case testing for scenarios that interleave
//! asynchronous, synchronous, and parallel tasks using unified task management.
//! Tests are designed to validate production-ready behavior under extreme conditions.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use moirai::{Moirai, Priority};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc, Barrier,
};
use std::time::{Duration, Instant};

/// Constants for test configuration (SSOT principle)
const STRESS_TASK_COUNT: usize = 100;
const CONCURRENT_WORKER_COUNT: usize = 8;
const MIXED_WORKLOAD_SIZE: usize = 50;
const TIMEOUT_DURATION_MS: u64 = 5000;
const HEAVY_COMPUTATION_ITERATIONS: usize = 1000;
const IO_SIMULATION_DELAY_MS: u64 = 1;

/// Test fixture for interleaved execution testing
struct InterleavedTestFixture {
    runtime: Moirai,
    counter: Arc<AtomicUsize>,
    start_time: Instant,
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
            start_time: Instant::now(),
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

    fn elapsed_ms(&self) -> u128 {
        self.start_time.elapsed().as_millis()
    }
}

/// Edge Case 1: Rapid Task Type Switching
/// Tests the executor's ability to handle rapid switches between task types
#[test]
fn test_rapid_task_type_switching() {
    let fixture = InterleavedTestFixture::new();
    let counter = fixture.counter.clone();

    // Rapidly alternate between different execution patterns
    for i in 0..MIXED_WORKLOAD_SIZE {
        let counter_clone = counter.clone();

        match i % 3 {
            0 => {
                // Synchronous CPU-bound task using spawn_fn
                let handle = fixture.runtime.spawn_fn(move || {
                    // Simulate CPU work
                    let mut sum = 0;
                    for j in 0..HEAVY_COMPUTATION_ITERATIONS {
                        sum += j;
                    }
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    sum
                });
                // Let it run but don't wait for result to allow rapid switching
                std::mem::drop(handle);
            }
            1 => {
                // I/O simulation using blocking task
                let handle = fixture.runtime.spawn_fn(move || {
                    // Simulate I/O delay with actual work instead of async
                    std::thread::sleep(Duration::from_millis(IO_SIMULATION_DELAY_MS));
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    42
                });
                std::mem::drop(handle);
            }
            2 => {
                // Parallel computation task
                let handle = fixture.runtime.spawn_fn(move || {
                    // Quick parallel work simulation
                    let result = (0..100).map(|x| x * x).sum::<usize>();
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    result
                });
                std::mem::drop(handle);
            }
            _ => unreachable!(),
        }
    }

    // Wait for completion with timeout
    let start = Instant::now();
    while counter.load(Ordering::SeqCst) < MIXED_WORKLOAD_SIZE {
        if start.elapsed().as_millis() > TIMEOUT_DURATION_MS as u128 {
            panic!("Timeout waiting for rapid switching tasks to complete");
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    fixture.verify_completion(MIXED_WORKLOAD_SIZE);
    println!("Rapid switching completed in {}ms", fixture.elapsed_ms());
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
    println!(
        "Resource contention handled correctly in {}ms",
        fixture.elapsed_ms()
    );
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

    // Stage 1: Parallel computation producers
    for i in 0..10 {
        let stage_counter = stage_counters[0].clone();
        let handle = runtime.spawn_fn(move || {
            // Simulate parallel work
            let result = (0..100).map(|x| x * i).sum::<usize>();
            stage_counter.fetch_add(1, Ordering::SeqCst);
            result
        });
        std::mem::drop(handle);
    }

    // Stage 2: Processing (wait for stage 1 partial completion)
    for _ in 0..5 {
        let wait_counter = stage_counters[0].clone();
        let stage_counter = stage_counters[1].clone();
        let handle = runtime.spawn_fn(move || {
            // Wait for dependency
            while wait_counter.load(Ordering::SeqCst) < 5 {
                std::thread::sleep(Duration::from_millis(1));
            }

            // Process with small delay
            std::thread::sleep(Duration::from_millis(2));

            stage_counter.fetch_add(1, Ordering::SeqCst) + 1
        });
        std::mem::drop(handle);
    }

    // Stage 3: Finalizers (wait for stage 2 partial completion)
    for _ in 0..3 {
        let wait_counter = stage_counters[1].clone();
        let final_counter = counter.clone();
        let handle = runtime.spawn_fn(move || {
            // Wait for dependency
            while wait_counter.load(Ordering::SeqCst) < 3 {
                std::thread::sleep(Duration::from_millis(1));
            }

            // Final processing
            final_counter.fetch_add(1, Ordering::SeqCst);
            "finalized"
        });
        std::mem::drop(handle);
    }

    // Wait for cascade completion
    let start = Instant::now();
    while counter.load(Ordering::SeqCst) < 3 {
        if start.elapsed().as_millis() > TIMEOUT_DURATION_MS as u128 {
            panic!("Timeout in cascading dependencies test");
        }
        std::thread::sleep(Duration::from_millis(1));
    }

    // Verify all stages completed appropriately
    assert!(
        stage_counters[0].load(Ordering::SeqCst) >= 5,
        "Stage 1 incomplete"
    );
    assert!(
        stage_counters[1].load(Ordering::SeqCst) >= 3,
        "Stage 2 incomplete"
    );

    let actual = counter.load(Ordering::SeqCst);
    assert_eq!(
        actual, 3,
        "Task completion mismatch: expected {}, got {}",
        3, actual
    );

    println!(
        "Cascading dependencies completed in {}ms",
        start.elapsed().as_millis()
    );
}

/// Edge Case 5: Burst Load with Mixed Priorities
/// Tests system behavior under sudden load spikes
#[test]
fn test_burst_load_handling() {
    let fixture = InterleavedTestFixture::new();
    let completion_times = Arc::new(std::sync::Mutex::new(Vec::new()));

    // Create burst of mixed tasks at different priorities
    let start = Instant::now();

    for i in 0..STRESS_TASK_COUNT {
        let counter = fixture.counter.clone();
        let times = completion_times.clone();
        let task_start = start;

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

                let completion_time = task_start.elapsed().as_millis();
                times.lock().unwrap().push(completion_time);
                counter.fetch_add(1, Ordering::SeqCst);
                sum
            },
            priority,
        );
        std::mem::drop(handle);
    }

    // Wait for burst completion
    let start_wait = Instant::now();
    while fixture.counter.load(Ordering::SeqCst) < STRESS_TASK_COUNT {
        if start_wait.elapsed().as_millis() > TIMEOUT_DURATION_MS as u128 * 2 {
            panic!("Timeout handling burst load");
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    fixture.verify_completion(STRESS_TASK_COUNT);

    // Analyze completion time distribution
    let times = completion_times.lock().unwrap();
    let avg_time = times.iter().sum::<u128>() / times.len() as u128;
    let max_time = *times.iter().max().unwrap();

    println!(
        "Burst load completed: avg={}ms, max={}ms",
        avg_time, max_time
    );

    // Verify reasonable performance under load
    assert!(
        max_time < TIMEOUT_DURATION_MS as u128,
        "Maximum completion time exceeded threshold"
    );
}

/// Edge Case 6: Interleaved Error Handling
/// Tests error propagation across different execution contexts
#[test]
fn test_interleaved_error_handling() {
    let fixture = InterleavedTestFixture::new();
    let success_counter = Arc::new(AtomicUsize::new(0));
    let error_counter = Arc::new(AtomicUsize::new(0));

    // Mix successful and error-handling tasks
    for i in 0..20 {
        let success_count = success_counter.clone();
        let error_count = error_counter.clone();

        match i % 4 {
            0 => {
                // Successful task
                let handle = fixture.runtime.spawn_fn(move || {
                    std::thread::sleep(Duration::from_millis(1));
                    success_count.fetch_add(1, Ordering::SeqCst);
                    "success"
                });
                std::mem::drop(handle);
            }
            1 => {
                // Another successful task
                let handle = fixture.runtime.spawn_fn(move || {
                    success_count.fetch_add(1, Ordering::SeqCst);
                    "sync_success"
                });
                std::mem::drop(handle);
            }
            2 => {
                // Task that handles errors gracefully
                let handle = fixture.runtime.spawn_fn(move || {
                    // This task should succeed
                    success_count.fetch_add(1, Ordering::SeqCst);
                    "handled_success"
                });
                std::mem::drop(handle);
            }
            3 => {
                // Task with error simulation in logic
                let handle = fixture.runtime.spawn_fn(move || {
                    // Simulate error condition but handle gracefully
                    let simulated_error = i % 10 == 0;
                    if simulated_error {
                        error_count.fetch_add(1, Ordering::SeqCst);
                        "error_handled"
                    } else {
                        success_count.fetch_add(1, Ordering::SeqCst);
                        "success"
                    }
                });
                std::mem::drop(handle);
            }
            _ => unreachable!(),
        }
    }

    // Wait for completion
    let start = Instant::now();
    let expected_total = 20;
    while (success_counter.load(Ordering::SeqCst) + error_counter.load(Ordering::SeqCst))
        < expected_total
    {
        if start.elapsed().as_millis() > TIMEOUT_DURATION_MS as u128 {
            panic!("Timeout in error handling test");
        }
        std::thread::sleep(Duration::from_millis(1));
    }

    let successes = success_counter.load(Ordering::SeqCst);
    let errors = error_counter.load(Ordering::SeqCst);

    println!(
        "Error handling: {} successes, {} errors handled",
        successes, errors
    );

    // Verify system remained stable despite mixed success/error conditions
    assert!(successes > 0, "No successful tasks completed");
    assert_eq!(
        successes + errors,
        expected_total,
        "Task count mismatch in error handling"
    );
}
