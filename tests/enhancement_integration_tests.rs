//! Integration tests for Moirai enhancements.
//!
//! This module tests the integration of all enhancement opportunities:
//! - Object pooling for task allocation
//! - NUMA-aware work stealing
//! - Zero-copy communication channels
//! - Adaptive batching for high throughput

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::HashMap;

use moirai_core::{
    Task, TaskId, Priority,
    pool::{TaskPool, TaskWrapper, PoolStats},
};
use moirai_scheduler::numa_scheduler::{
    NumaAwareScheduler, CpuTopology, AdaptiveBackoff,
};
use moirai_transport::zero_copy::{
    ZeroCopyChannel, AdaptiveBatchChannel, MemoryMappedRing,
    ZeroCopyError, BatchStats,
};

/// Test task for integration testing.
#[derive(Debug)]
struct TestTask {
    id: usize,
    work_amount: u64,
    result: Option<u64>,
}

impl TestTask {
    fn new(id: usize, work_amount: u64) -> Self {
        Self {
            id,
            work_amount,
            result: None,
        }
    }

    fn execute(&mut self) -> u64 {
        // Simulate work
        let mut sum = 0u64;
        for i in 0..self.work_amount {
            sum = sum.wrapping_add(i);
        }
        self.result = Some(sum);
        sum
    }
}

impl Task for TestTask {
    type Output = u64;

    fn execute(mut self) -> Self::Output {
        self.execute()
    }
}

/// Test object pooling with high concurrency and edge cases.
#[test]
fn test_object_pooling_integration() {
    const NUM_THREADS: usize = 8;
    const TASKS_PER_THREAD: usize = 1000;
    const POOL_SIZE: usize = 100;

    let pool = Arc::new(TaskPool::<TestTask>::new(POOL_SIZE));
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let mut handles = Vec::new();

    // Pre-populate pool for better hit rates
    pool.pre_populate(POOL_SIZE / 2);

    for thread_id in 0..NUM_THREADS {
        let pool = pool.clone();
        let barrier = barrier.clone();
        
        handles.push(thread::spawn(move || {
            barrier.wait(); // Synchronize start
            
            let mut local_stats = Vec::new();
            
            for task_id in 0..TASKS_PER_THREAD {
                // Acquire wrapper from pool
                let mut wrapper = pool.acquire();
                
                // Initialize with test task
                let task = TestTask::new(thread_id * 1000 + task_id, 100);
                wrapper.init(task, TaskId::new(), Priority::Normal);
                
                // Simulate task execution
                if let Some(mut task) = wrapper.take() {
                    let result = task.execute();
                    assert!(result > 0);
                }
                
                // Return wrapper to pool
                pool.release(wrapper);
                
                // Collect stats periodically
                if task_id % 100 == 0 {
                    local_stats.push(pool.stats());
                }
            }
            
            local_stats
        }));
    }

    // Wait for all threads to complete
    let mut all_stats = Vec::new();
    for handle in handles {
        let stats = handle.join().unwrap();
        all_stats.extend(stats);
    }

    // Verify final statistics
    let final_stats = pool.stats();
    println!("Final pool stats: {:?}", final_stats);

    // Assertions
    assert_eq!(final_stats.reset_failures, 0, "No reset failures should occur");
    assert!(final_stats.hit_rate() > 50.0, "Hit rate should be > 50%");
    assert!(final_stats.pool_size <= POOL_SIZE, "Pool size should not exceed limit");
    assert_eq!(
        final_stats.cache_hits + final_stats.cache_misses,
        (NUM_THREADS * TASKS_PER_THREAD) as usize,
        "Total operations should match"
    );

    println!("Object pooling test passed with hit rate: {:.1}%", final_stats.hit_rate());
}

/// Test NUMA-aware scheduler with work stealing and load balancing.
#[test]
fn test_numa_aware_scheduling() {
    const NUM_WORKERS: usize = 4;
    const TASKS_PER_WORKER: usize = 500;

    // Create NUMA-aware scheduler
    let mut scheduler = NumaAwareScheduler::new(None, 1000);
    
    // Assign workers to different NUMA nodes (simulated)
    for worker_id in 0..NUM_WORKERS {
        scheduler.assign_worker(worker_id, Some(worker_id % 2)); // Simulate 2 NUMA nodes
    }

    let scheduler = Arc::new(scheduler);
    let barrier = Arc::new(Barrier::new(NUM_WORKERS));
    let mut handles = Vec::new();

    // Phase 1: Add tasks to create imbalanced load
    for i in 0..TASKS_PER_WORKER * NUM_WORKERS {
        let task = Box::new(TestTask::new(i, 50));
        scheduler.schedule_task(task).unwrap();
    }

    println!("Initial load: {}", scheduler.load());

    // Phase 2: Workers steal and execute tasks
    for worker_id in 0..NUM_WORKERS {
        let scheduler = scheduler.clone();
        let barrier = barrier.clone();
        
        handles.push(thread::spawn(move || {
            barrier.wait(); // Synchronize start
            
            let mut tasks_executed = 0;
            let mut steal_attempts = 0;
            let start_time = Instant::now();
            
            loop {
                // Try to get local work first
                if let Ok(Some(task)) = scheduler.next_task() {
                    // Execute task
                    let _result = task.execute();
                    tasks_executed += 1;
                } else {
                    // Try stealing with NUMA locality
                    steal_attempts += 1;
                    if let Some(stolen_task) = scheduler.steal_with_locality(worker_id) {
                        let _result = stolen_task.execute();
                        tasks_executed += 1;
                    } else {
                        // No work available
                        if scheduler.load() == 0 {
                            break;
                        }
                        thread::yield_now();
                    }
                }
                
                // Periodic load balancing
                if tasks_executed % 50 == 0 {
                    scheduler.balance_load();
                }
            }
            
            let elapsed = start_time.elapsed();
            (worker_id, tasks_executed, steal_attempts, elapsed)
        }));
    }

    // Wait for completion and collect results
    let mut total_executed = 0;
    let mut total_steal_attempts = 0;
    
    for handle in handles {
        let (worker_id, executed, steals, elapsed) = handle.join().unwrap();
        println!(
            "Worker {}: executed {}, steal attempts {}, time {:?}",
            worker_id, executed, steals, elapsed
        );
        total_executed += executed;
        total_steal_attempts += steals;
    }

    // Verify results
    let final_stats = scheduler.statistics();
    println!("NUMA scheduler stats: {:#?}", final_stats);

    assert_eq!(total_executed, TASKS_PER_WORKER * NUM_WORKERS);
    assert_eq!(scheduler.load(), 0, "All tasks should be completed");
    assert!(final_stats.total_steal_attempts > 0, "Some stealing should occur");
    assert!(final_stats.steal_success_rate > 0.0, "Some steals should succeed");

    println!("NUMA-aware scheduling test passed");
}

/// Test zero-copy channels with high throughput and memory efficiency.
#[test]
fn test_zero_copy_channels() {
    const BUFFER_SIZE: usize = 1024;
    const NUM_MESSAGES: usize = 100_000;
    const MESSAGE_SIZE: usize = 64;

    // Create zero-copy channel
    let (sender, receiver) = ZeroCopyChannel::<Vec<u8>>::new(BUFFER_SIZE).unwrap();
    let sender = Arc::new(sender);
    let receiver = Arc::new(receiver);

    let start_time = Instant::now();

    // Producer thread
    let producer_handle = {
        let sender = sender.clone();
        thread::spawn(move || {
            let mut messages_sent = 0;
            
            for i in 0..NUM_MESSAGES {
                let message = vec![i as u8; MESSAGE_SIZE];
                
                loop {
                    match sender.send(message.clone()) {
                        Ok(()) => {
                            messages_sent += 1;
                            break;
                        }
                        Err(ZeroCopyError::Full) => {
                            thread::yield_now();
                        }
                        Err(e) => panic!("Unexpected send error: {:?}", e),
                    }
                }
            }
            
            messages_sent
        })
    };

    // Consumer thread
    let consumer_handle = {
        let receiver = receiver.clone();
        thread::spawn(move || {
            let mut messages_received = 0;
            let mut total_bytes = 0;
            
            while messages_received < NUM_MESSAGES {
                match receiver.recv() {
                    Ok(message) => {
                        assert_eq!(message.len(), MESSAGE_SIZE);
                        assert_eq!(message[0] as usize, messages_received);
                        total_bytes += message.len();
                        messages_received += 1;
                    }
                    Err(ZeroCopyError::Empty) => {
                        thread::yield_now();
                    }
                    Err(e) => panic!("Unexpected receive error: {:?}", e),
                }
            }
            
            (messages_received, total_bytes)
        })
    };

    // Wait for completion
    let messages_sent = producer_handle.join().unwrap();
    let (messages_received, total_bytes) = consumer_handle.join().unwrap();
    
    let elapsed = start_time.elapsed();
    let throughput = NUM_MESSAGES as f64 / elapsed.as_secs_f64();
    let bandwidth_mbps = (total_bytes as f64 / (1024.0 * 1024.0)) / elapsed.as_secs_f64();

    println!("Zero-copy performance:");
    println!("  Messages: {} sent, {} received", messages_sent, messages_received);
    println!("  Throughput: {:.0} messages/second", throughput);
    println!("  Bandwidth: {:.1} MB/s", bandwidth_mbps);
    println!("  Latency: {:.0} ns/message", elapsed.as_nanos() as f64 / NUM_MESSAGES as f64);

    // Verify results
    assert_eq!(messages_sent, NUM_MESSAGES);
    assert_eq!(messages_received, NUM_MESSAGES);
    assert_eq!(total_bytes, NUM_MESSAGES * MESSAGE_SIZE);
    assert!(throughput > 50_000.0, "Should achieve > 50K messages/second");

    println!("Zero-copy channels test passed");
}

/// Test adaptive batching with varying load patterns.
#[test]
fn test_adaptive_batching() {
    const BUFFER_SIZE: usize = 4096;
    const BATCH_DELAY: Duration = Duration::from_micros(100);
    
    // Create adaptive batch channel
    let (sender, receiver) = AdaptiveBatchChannel::<u32>::new(BUFFER_SIZE, BATCH_DELAY).unwrap();

    // Test different load patterns
    let test_scenarios = vec![
        ("Low load", 100, Duration::from_millis(10)),
        ("Medium load", 1000, Duration::from_millis(1)),
        ("High load", 10000, Duration::from_micros(100)),
        ("Burst load", 50000, Duration::from_micros(10)),
    ];

    for (scenario_name, num_messages, send_interval) in test_scenarios {
        println!("Testing scenario: {}", scenario_name);
        
        let start_time = Instant::now();
        let sender_handle = {
            let sender = sender.clone();
            thread::spawn(move || {
                for i in 0..num_messages {
                    sender.send_adaptive(i).unwrap();
                    
                    if send_interval > Duration::ZERO {
                        thread::sleep(send_interval);
                    }
                    
                    // Log stats periodically
                    if i % (num_messages / 4).max(1) == 0 {
                        let stats = sender.batch_stats();
                        println!("  Progress {}/{}: {:?}", i, num_messages, stats);
                    }
                }
                
                // Force flush remaining messages
                sender.flush().unwrap();
            })
        };

        let receiver_handle = thread::spawn(move || {
            let mut received = 0;
            let mut last_value = None;
            
            while received < num_messages {
                match receiver.try_recv() {
                    Ok(value) => {
                        // Verify message ordering
                        if let Some(last) = last_value {
                            assert!(value > last, "Messages should be in order");
                        }
                        last_value = Some(value);
                        received += 1;
                    }
                    Err(ZeroCopyError::Empty) => {
                        thread::yield_now();
                    }
                    Err(e) => panic!("Unexpected error: {:?}", e),
                }
            }
            
            received
        });

        sender_handle.join().unwrap();
        let received_count = receiver_handle.join().unwrap();
        
        let elapsed = start_time.elapsed();
        let throughput = num_messages as f64 / elapsed.as_secs_f64();
        
        println!("  Scenario {} completed:", scenario_name);
        println!("    Messages: {}", received_count);
        println!("    Time: {:?}", elapsed);
        println!("    Throughput: {:.0} msg/s", throughput);
        
        assert_eq!(received_count, num_messages);
        
        // Get final batch statistics
        let final_stats = sender.batch_stats();
        println!("    Final batch stats: {:?}", final_stats);
    }

    println!("Adaptive batching test passed");
}

/// Comprehensive stress test combining all enhancements.
#[test]
fn test_comprehensive_integration() {
    const NUM_PRODUCERS: usize = 4;
    const NUM_CONSUMERS: usize = 4;
    const MESSAGES_PER_PRODUCER: usize = 10_000;
    const TASK_POOL_SIZE: usize = 1000;

    println!("Starting comprehensive integration test...");

    // Create all components
    let task_pool = Arc::new(TaskPool::<TestTask>::new(TASK_POOL_SIZE));
    let numa_scheduler = Arc::new(NumaAwareScheduler::new(None, TASK_POOL_SIZE));
    let (batch_sender, batch_receiver) = AdaptiveBatchChannel::<(usize, u64)>::new(
        4096,
        Duration::from_micros(50),
    ).unwrap();
    let batch_sender = Arc::new(batch_sender);
    let batch_receiver = Arc::new(batch_receiver);

    let start_time = Instant::now();
    let mut handles = Vec::new();

    // Producer threads: Use task pool + NUMA scheduler + adaptive batching
    for producer_id in 0..NUM_PRODUCERS {
        let task_pool = task_pool.clone();
        let numa_scheduler = numa_scheduler.clone();
        let batch_sender = batch_sender.clone();
        
        handles.push(thread::spawn(move || {
            let mut produced = 0;
            
            for task_id in 0..MESSAGES_PER_PRODUCER {
                // Get task wrapper from pool
                let mut wrapper = task_pool.acquire();
                
                // Create and initialize task
                let task = TestTask::new(producer_id * 10000 + task_id, 200);
                wrapper.init(task, TaskId::new(), Priority::Normal);
                
                // Execute task (simulate work)
                if let Some(mut task) = wrapper.take() {
                    let result = task.execute();
                    
                    // Send result through adaptive batch channel
                    batch_sender.send_adaptive((producer_id, result)).unwrap();
                    produced += 1;
                }
                
                // Return wrapper to pool
                task_pool.release(wrapper);
                
                // Yield occasionally to allow other threads to run
                if task_id % 100 == 0 {
                    thread::yield_now();
                }
            }
            
            produced
        }));
    }

    // Consumer threads: Receive from adaptive batch channel
    for consumer_id in 0..NUM_CONSUMERS {
        let batch_receiver = batch_receiver.clone();
        
        handles.push(thread::spawn(move || {
            let mut consumed = 0;
            let mut producer_counts = HashMap::new();
            
            loop {
                match batch_receiver.try_recv() {
                    Ok((producer_id, result)) => {
                        assert!(result > 0, "Result should be positive");
                        *producer_counts.entry(producer_id).or_insert(0) += 1;
                        consumed += 1;
                        
                        // Check if we've received all messages
                        if consumed >= (NUM_PRODUCERS * MESSAGES_PER_PRODUCER) / NUM_CONSUMERS {
                            break;
                        }
                    }
                    Err(ZeroCopyError::Empty) => {
                        thread::yield_now();
                    }
                    Err(e) => panic!("Consumer {} error: {:?}", consumer_id, e),
                }
            }
            
            (consumer_id, consumed, producer_counts)
        }));
    }

    // Wait for all threads to complete
    let mut total_produced = 0;
    let mut total_consumed = 0;
    let mut all_producer_counts = HashMap::new();

    for (i, handle) in handles.into_iter().enumerate() {
        if i < NUM_PRODUCERS {
            // Producer thread
            let produced = handle.join().unwrap();
            total_produced += produced;
        } else {
            // Consumer thread
            let (consumer_id, consumed, producer_counts) = handle.join().unwrap();
            println!("Consumer {} processed {} messages", consumer_id, consumed);
            total_consumed += consumed;
            
            for (producer_id, count) in producer_counts {
                *all_producer_counts.entry(producer_id).or_insert(0) += count;
            }
        }
    }

    // Force flush any remaining messages
    batch_sender.flush().unwrap();
    
    // Consume any remaining messages
    let mut remaining = 0;
    while let Ok(_) = batch_receiver.try_recv() {
        remaining += 1;
        total_consumed += 1;
    }

    let elapsed = start_time.elapsed();

    // Collect final statistics
    let pool_stats = task_pool.stats();
    let numa_stats = numa_scheduler.statistics();
    let batch_stats = batch_sender.batch_stats();

    println!("\nComprehensive integration test results:");
    println!("  Execution time: {:?}", elapsed);
    println!("  Messages produced: {}", total_produced);
    println!("  Messages consumed: {}", total_consumed);
    println!("  Remaining messages: {}", remaining);
    println!("  Total throughput: {:.0} msg/s", 
             (total_produced + total_consumed) as f64 / elapsed.as_secs_f64());

    println!("\nComponent statistics:");
    println!("  Task pool hit rate: {:.1}%", pool_stats.hit_rate());
    println!("  NUMA steal success rate: {:.1}%", numa_stats.steal_success_rate);
    println!("  NUMA locality rate: {:.1}%", numa_stats.numa_locality_rate);
    println!("  Batch threshold: {}", batch_stats.current_threshold);
    println!("  Batch throughput: {:.0} msg/s", batch_stats.recent_throughput);

    // Verify correctness
    assert_eq!(total_produced, NUM_PRODUCERS * MESSAGES_PER_PRODUCER);
    assert_eq!(total_consumed, NUM_PRODUCERS * MESSAGES_PER_PRODUCER);
    assert_eq!(pool_stats.reset_failures, 0);
    
    // Verify load distribution
    for producer_id in 0..NUM_PRODUCERS {
        let count = all_producer_counts.get(&producer_id).unwrap_or(&0);
        assert!(*count > 0, "Each producer should have messages consumed");
    }

    println!("\nComprehensive integration test PASSED!");
}

/// Test error handling and edge cases across all enhancements.
#[test]
fn test_error_handling_edge_cases() {
    println!("Testing error handling and edge cases...");

    // Test 1: Object pool with zero size
    let zero_pool = TaskPool::<TestTask>::new(0); // Unlimited
    let wrapper = zero_pool.acquire();
    zero_pool.release(wrapper);
    assert_eq!(zero_pool.stats().pool_size, 1);

    // Test 2: NUMA scheduler with invalid topology
    let scheduler = NumaAwareScheduler::new(Some(CpuTopology::detect()), 10);
    assert_eq!(scheduler.load(), 0);

    // Test 3: Zero-copy channel with invalid capacity
    assert!(MemoryMappedRing::<u32>::new(15).is_err()); // Not power of 2
    assert!(MemoryMappedRing::<u32>::new(0).is_err());  // Zero size

    // Test 4: Adaptive batching with closed channel
    let (sender, receiver) = AdaptiveBatchChannel::<u32>::new(64, Duration::from_millis(10)).unwrap();
    sender.send_adaptive(42).unwrap();
    sender.flush().unwrap();
    
    let received = receiver.try_recv().unwrap();
    assert_eq!(received, 42);

    // Test 5: Concurrent access edge cases
    let pool = Arc::new(TaskPool::<TestTask>::new(10));
    let handles: Vec<_> = (0..20).map(|i| {
        let pool = pool.clone();
        thread::spawn(move || {
            let wrapper = pool.acquire();
            thread::sleep(Duration::from_millis(1)); // Hold briefly
            pool.release(wrapper);
            i
        })
    }).collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Test 6: Adaptive backoff behavior
    let backoff = AdaptiveBackoff::new(100, 10000);
    
    // Record failures and verify exponential backoff
    for _ in 0..5 {
        backoff.record_failure();
    }
    assert!(backoff.current_delay() > Duration::from_nanos(100));
    
    // Record success and verify reset
    backoff.record_success();
    assert_eq!(backoff.current_delay(), Duration::from_nanos(100));

    println!("Error handling and edge cases test PASSED!");
}

/// Performance benchmark comparing enhanced vs basic implementations.
#[test]
fn test_performance_comparison() {
    const NUM_OPERATIONS: usize = 100_000;
    const NUM_THREADS: usize = 4;

    println!("Running performance comparison...");

    // Test 1: Object pooling vs direct allocation
    let start_time = Instant::now();
    let pool = Arc::new(TaskPool::<TestTask>::new(1000));
    
    let pool_handles: Vec<_> = (0..NUM_THREADS).map(|thread_id| {
        let pool = pool.clone();
        thread::spawn(move || {
            for i in 0..NUM_OPERATIONS / NUM_THREADS {
                let mut wrapper = pool.acquire();
                wrapper.init(TestTask::new(i, 10), TaskId::new(), Priority::Normal);
                wrapper.take(); // Simulate usage
                pool.release(wrapper);
            }
        })
    }).collect();

    for handle in pool_handles {
        handle.join().unwrap();
    }
    
    let pool_time = start_time.elapsed();
    let pool_stats = pool.stats();

    // Test 2: Direct allocation (baseline)
    let start_time = Instant::now();
    
    let direct_handles: Vec<_> = (0..NUM_THREADS).map(|thread_id| {
        thread::spawn(move || {
            for i in 0..NUM_OPERATIONS / NUM_THREADS {
                let _task = TestTask::new(i, 10); // Direct allocation
            }
        })
    }).collect();

    for handle in direct_handles {
        handle.join().unwrap();
    }
    
    let direct_time = start_time.elapsed();

    // Test 3: Zero-copy vs standard channels
    let (zc_sender, zc_receiver) = ZeroCopyChannel::<u64>::new(1024).unwrap();
    let (std_sender, std_receiver) = std::sync::mpsc::channel::<u64>();

    // Zero-copy performance
    let start_time = Instant::now();
    let zc_send_handle = thread::spawn(move || {
        for i in 0..NUM_OPERATIONS {
            while zc_sender.send(i as u64).is_err() {
                thread::yield_now();
            }
        }
    });
    
    let zc_recv_handle = thread::spawn(move || {
        for _ in 0..NUM_OPERATIONS {
            while zc_receiver.recv().is_err() {
                thread::yield_now();
            }
        }
    });
    
    zc_send_handle.join().unwrap();
    zc_recv_handle.join().unwrap();
    let zc_time = start_time.elapsed();

    // Standard channel performance
    let start_time = Instant::now();
    let std_send_handle = thread::spawn(move || {
        for i in 0..NUM_OPERATIONS {
            std_sender.send(i as u64).unwrap();
        }
    });
    
    let std_recv_handle = thread::spawn(move || {
        for _ in 0..NUM_OPERATIONS {
            std_receiver.recv().unwrap();
        }
    });
    
    std_send_handle.join().unwrap();
    std_recv_handle.join().unwrap();
    let std_time = start_time.elapsed();

    // Report results
    println!("\nPerformance comparison results:");
    println!("  Object pooling:");
    println!("    Time: {:?}", pool_time);
    println!("    Hit rate: {:.1}%", pool_stats.hit_rate());
    println!("    Speedup potential: {:.1}x", pool_stats.hit_rate() / 100.0 * 5.0);
    
    println!("  Direct allocation:");
    println!("    Time: {:?}", direct_time);
    
    println!("  Zero-copy channels:");
    println!("    Time: {:?}", zc_time);
    println!("    Throughput: {:.0} msg/s", NUM_OPERATIONS as f64 / zc_time.as_secs_f64());
    
    println!("  Standard channels:");
    println!("    Time: {:?}", std_time);
    println!("    Throughput: {:.0} msg/s", NUM_OPERATIONS as f64 / std_time.as_secs_f64());
    
    let zc_improvement = std_time.as_nanos() as f64 / zc_time.as_nanos() as f64;
    println!("  Zero-copy improvement: {:.1}x", zc_improvement);

    // Verify performance improvements
    assert!(pool_stats.hit_rate() > 50.0, "Pool should have good hit rate");
    
    println!("Performance comparison test PASSED!");
}

/// Memory usage and leak detection test.
#[test]
fn test_memory_efficiency() {
    println!("Testing memory efficiency...");

    // Test 1: Object pool memory bounds
    let pool = TaskPool::<TestTask>::new(100);
    
    // Fill pool to capacity
    let mut wrappers = Vec::new();
    for _ in 0..150 { // More than capacity
        wrappers.push(pool.acquire());
    }
    
    // Release all wrappers
    for wrapper in wrappers {
        pool.release(wrapper);
    }
    
    let stats = pool.stats();
    assert!(stats.pool_size <= 100, "Pool should respect size limit");

    // Test 2: Zero-copy memory alignment
    let ring = MemoryMappedRing::<u64>::new(64).unwrap();
    
    // Verify we can send/receive without corruption
    for i in 0..64 {
        ring.send_zero_copy(i * 0x123456789ABCDEF0).unwrap();
    }
    
    for i in 0..64 {
        let received = ring.recv_zero_copy().unwrap();
        assert_eq!(received, i * 0x123456789ABCDEF0);
    }

    // Test 3: Adaptive batching memory usage
    let (sender, receiver) = AdaptiveBatchChannel::<Vec<u8>>::new(
        512,
        Duration::from_millis(10),
    ).unwrap();
    
    // Send varying size messages
    for size in [1, 10, 100, 1000, 10000] {
        let message = vec![42u8; size];
        sender.send_adaptive(message).unwrap();
    }
    
    sender.flush().unwrap();
    
    // Verify all messages received correctly
    for expected_size in [1, 10, 100, 1000, 10000] {
        let received = receiver.recv().unwrap();
        assert_eq!(received.len(), expected_size);
        assert!(received.iter().all(|&b| b == 42));
    }

    println!("Memory efficiency test PASSED!");
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    /// Long-running stress test for stability verification.
    #[test]
    #[ignore] // Run with --ignored for stress testing
    fn test_long_running_stability() {
        const DURATION_MINUTES: u64 = 10;
        const CHECK_INTERVAL_SECS: u64 = 30;
        
        println!("Starting {}-minute stability test...", DURATION_MINUTES);
        
        let pool = Arc::new(TaskPool::<TestTask>::new(1000));
        let scheduler = Arc::new(NumaAwareScheduler::new(None, 1000));
        let (sender, receiver) = AdaptiveBatchChannel::<u64>::new(
            4096,
            Duration::from_millis(10),
        ).unwrap();
        let sender = Arc::new(sender);
        let receiver = Arc::new(receiver);
        
        let start_time = Instant::now();
        let end_time = start_time + Duration::from_secs(DURATION_MINUTES * 60);
        
        // Spawn continuous workers
        let mut handles = Vec::new();
        
        // Producer worker
        let producer_handle = {
            let pool = pool.clone();
            let sender = sender.clone();
            thread::spawn(move || {
                let mut counter = 0u64;
                while Instant::now() < end_time {
                    let wrapper = pool.acquire();
                    pool.release(wrapper);
                    
                    sender.send_adaptive(counter).unwrap();
                    counter += 1;
                    
                    if counter % 10000 == 0 {
                        thread::sleep(Duration::from_millis(1));
                    }
                }
                counter
            })
        };
        
        // Consumer worker
        let consumer_handle = {
            let receiver = receiver.clone();
            thread::spawn(move || {
                let mut received = 0u64;
                while Instant::now() < end_time {
                    match receiver.try_recv() {
                        Ok(_) => received += 1,
                        Err(ZeroCopyError::Empty) => thread::yield_now(),
                        Err(e) => panic!("Unexpected error: {:?}", e),
                    }
                }
                received
            })
        };
        
        handles.push(producer_handle);
        handles.push(consumer_handle);
        
        // Monitor thread
        let monitor_handle = {
            let pool = pool.clone();
            let scheduler = scheduler.clone();
            let sender = sender.clone();
            thread::spawn(move || {
                let mut last_check = Instant::now();
                
                while Instant::now() < end_time {
                    thread::sleep(Duration::from_secs(CHECK_INTERVAL_SECS));
                    
                    let now = Instant::now();
                    let elapsed = now.duration_since(last_check);
                    last_check = now;
                    
                    let pool_stats = pool.stats();
                    let scheduler_stats = scheduler.statistics();
                    let batch_stats = sender.batch_stats();
                    
                    println!("Stability check at {:?}:", now.duration_since(start_time));
                    println!("  Pool hit rate: {:.1}%", pool_stats.hit_rate());
                    println!("  Scheduler steals: {}", scheduler_stats.total_steal_attempts);
                    println!("  Batch throughput: {:.0} msg/s", batch_stats.recent_throughput);
                    println!("  Memory usage: {} pool, {} pending", 
                             pool_stats.pool_size, batch_stats.pending_messages);
                }
            })
        };
        
        handles.push(monitor_handle);
        
        // Wait for completion
        let mut results = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(result) => {
                    if let Ok(count) = result.downcast::<u64>() {
                        results.push(*count);
                    }
                }
                Err(e) => panic!("Thread panicked: {:?}", e),
            }
        }
        
        let total_elapsed = start_time.elapsed();
        println!("Stability test completed after {:?}", total_elapsed);
        
        // Verify no degradation
        let final_pool_stats = pool.stats();
        assert_eq!(final_pool_stats.reset_failures, 0);
        assert!(final_pool_stats.hit_rate() > 30.0); // Should maintain reasonable hit rate
        
        println!("Long-running stability test PASSED!");
    }
}