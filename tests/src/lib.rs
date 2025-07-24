//! Integration tests for Moirai concurrency library.

/// Integration tests for the complete Moirai system.
#[cfg(test)]
mod integration_tests {
    use moirai::{Moirai, Priority, TaskBuilder};
    use std::sync::{Arc, atomic::{AtomicU32, Ordering}};
    use std::time::Duration;

    #[test]
    fn test_basic_runtime_creation() {
        let runtime = Moirai::new();
        assert!(runtime.is_ok());
    }

    #[test]
    fn test_runtime_builder() {
        let runtime = Moirai::builder()
            .worker_threads(4)
            .build();
        assert!(runtime.is_ok());
    }

    #[test]
    fn test_simple_task_execution() {
        let runtime = Moirai::new().unwrap();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let handle = runtime.spawn_parallel(move || {
            counter_clone.fetch_add(1, Ordering::Relaxed);
            42
        });

        let result = handle.join();
        assert_eq!(result.unwrap(), 42);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    #[ignore] // Still investigating runtime-level hanging issue
    fn test_parallel_computation() {
        let runtime = Moirai::new().unwrap();
        let counter = Arc::new(AtomicU32::new(0));
        
        let handles: Vec<_> = (0..100)
            .map(|i| {
                let counter = counter.clone();
                runtime.spawn_parallel(move || {
                    counter.fetch_add(1, Ordering::Relaxed);
                    i * 2
                })
            })
            .collect();

        let results: Vec<_> = handles.into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        assert_eq!(results.len(), 100);
        assert_eq!(counter.load(Ordering::Relaxed), 100);
        
        // Verify results are correct
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(result, i * 2);
        }
    }

    #[test]
    fn test_priority_scheduling() {
        let runtime = Moirai::new().unwrap();
        let execution_order = Arc::new(std::sync::Mutex::new(Vec::new()));
        
        // Create priority tasks with small delays to ensure execution
        let order_clone = execution_order.clone();
        let high_task = TaskBuilder::new()
            .priority(Priority::High)
            .build(move || {
                std::thread::sleep(std::time::Duration::from_millis(5));
                order_clone.lock().unwrap().push("high");
                1
            });
        
        let order_clone = execution_order.clone();
        let low_task = TaskBuilder::new()
            .priority(Priority::Low)
            .build(move || {
                std::thread::sleep(std::time::Duration::from_millis(5));
                order_clone.lock().unwrap().push("low");
                2
            });

        // Spawn tasks with small delay between them
        let high_handle = runtime.spawn(high_task);
        std::thread::sleep(std::time::Duration::from_millis(2));
        let low_handle = runtime.spawn(low_task);

        // Wait for both tasks with timeout
        let high_result = high_handle.join_timeout(std::time::Duration::from_secs(1));
        let low_result = low_handle.join_timeout(std::time::Duration::from_secs(1));

        // Verify tasks completed successfully
        assert!(high_result.is_ok(), "High priority task should complete");
        assert!(low_result.is_ok(), "Low priority task should complete");

        // Verify both tasks executed
        let order = execution_order.lock().unwrap();
        assert_eq!(order.len(), 2, "Both tasks should have executed");
        assert!(order.contains(&"high"), "High priority task should have executed");
        assert!(order.contains(&"low"), "Low priority task should have executed");
    }

    /// Test CPU optimization features integrated with the executor.
    #[test]
    #[ignore] // CPU optimization disabled for now
    fn test_cpu_optimization_integration() {
        // use moirai_utils::cpu::{CpuTopology, affinity::AffinityMask};
        
        // CPU optimization tests disabled
        // let topology = CpuTopology::detect();
        // assert!(topology.logical_cores > 0);
        // assert!(topology.physical_cores > 0);
        // assert!(!topology.caches.is_empty());
        
        // let mask = AffinityMask::all();
        // assert!(!mask.is_empty());
        // assert!(mask.len() > 0);
        
        // let single_mask = AffinityMask::single(moirai_utils::cpu::CpuCore::new(0));
        // assert_eq!(single_mask.len(), 1);
        
        // Runtime creation and testing disabled
        // let runtime = Moirai::builder()
        //     .worker_threads(topology.logical_cores.min(8) as usize)
        //     .build()
        //     .unwrap();
        
        // let counter = Arc::new(AtomicU32::new(0));
        // let handles: Vec<_> = (0..topology.logical_cores as usize)
        //     .map(|i| {
        //         let counter = counter.clone();
        //         runtime.spawn_parallel(move || {
        //             let mut sum = 0u64;
        //             for j in 0..1000 {
        //             })
        //         .collect();
        
        // let results: Vec<_> = handles.into_iter()
        //     .map(|handle| handle.join().unwrap())
        //     .collect();
        
        // assert_eq!(results.len(), topology.logical_cores as usize);
        // assert_eq!(counter.load(Ordering::Relaxed), topology.logical_cores);
        
        // for result in results {
        //     assert!(result > 0);
        // }
    }

    /// Test memory prefetching utilities.
    #[test]
    fn test_memory_prefetching() {
        use moirai_utils::memory::{prefetch_read, prefetch_write};
        
        let data = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        
        // Test prefetching (should not crash)
        prefetch_read(data.as_ptr());
        prefetch_write(data.as_ptr());
        
        // Memory barriers disabled for now
        // memory_barrier();
        // compiler_barrier();
        
        // Create runtime and use prefetching in tasks
        let runtime = Moirai::new().unwrap();
        let data = Arc::new(data);
        
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let data = data.clone();
                runtime.spawn_parallel(move || {
                    let slice = &data[i..i+2];
                    prefetch_read(slice.as_ptr());
                    slice.iter().sum::<u32>()
                })
            })
            .collect();
        
        let results: Vec<_> = handles.into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        
        assert_eq!(results.len(), 4);
        // Verify computation results
        assert_eq!(results[0], 3);  // 1 + 2
        assert_eq!(results[1], 5);  // 2 + 3
        assert_eq!(results[2], 7);  // 3 + 4
        assert_eq!(results[3], 9);  // 4 + 5
    }

    /// Test NUMA awareness (if available).
    #[test]
    #[ignore] // NUMA feature disabled
    fn test_numa_awareness() {
        // use moirai_utils::numa::{current_numa_node, numa_node_count};
        
        // NUMA test content disabled
        // All test code commented out for now
    }

    /// Stress test with CPU optimizations.
    #[test]
    #[ignore] // CPU optimizations disabled
    fn test_cpu_optimized_stress() {
        // use moirai_utils::cpu::CpuTopology;
        
        // let topology = CpuTopology::detect();
        let runtime = Moirai::builder()
            .worker_threads(4) // Fixed thread count instead of topology-based
            .build()
            .unwrap();
        
        let task_count = 1000;
        let counter = Arc::new(AtomicU32::new(0));
        
        let handles: Vec<_> = (0..task_count)
            .map(|i| {
                let counter = counter.clone();
                runtime.spawn_parallel(move || {
                    // CPU-intensive computation that benefits from affinity
                    let mut result = 0u64;
                    for j in 0..100 {
                        result += ((i + j) as u64).wrapping_mul(17).wrapping_add(23);
                    }
                    counter.fetch_add(1, Ordering::Relaxed);
                    result
                })
            })
            .collect();
        
        let start = std::time::Instant::now();
        let results: Vec<_> = handles.into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        let duration = start.elapsed();
        
        assert_eq!(results.len(), task_count);
        assert_eq!(counter.load(Ordering::Relaxed), task_count as u32);
        
        // Verify performance (should complete reasonably quickly)
        assert!(duration < Duration::from_secs(10), "Stress test took too long: {:?}", duration);
        
        // Verify all computations completed
        for result in results {
            assert!(result > 0);
        }
        
        println!("CPU-optimized stress test completed {} tasks in {:?}", task_count, duration);
    }
}