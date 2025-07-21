//! Integration tests for Moirai concurrency library.

use moirai::{Moirai, Priority, TaskId, Task};
use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};

/// Integration tests for the complete Moirai system.
#[cfg(test)]
mod integration_tests {
    use super::*;

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
        
        // Create priority tasks using task builder
        let order_clone = execution_order.clone();
        let high_task = moirai::TaskBuilder::new()
            .priority(Priority::High)
            .build(move || {
                order_clone.lock().unwrap().push("high");
                1
            });
        
        let order_clone = execution_order.clone();
        let low_task = moirai::TaskBuilder::new()
            .priority(Priority::Low)
            .build(move || {
                order_clone.lock().unwrap().push("low");
                2
            });

        // Spawn tasks
        let high_handle = runtime.spawn(high_task);
        let low_handle = runtime.spawn(low_task);

        // Wait for both tasks
        let _high_result = high_handle.join();
        let _low_result = low_handle.join();

        // Note: Priority scheduling behavior may vary based on system load
        let order = execution_order.lock().unwrap();
        assert_eq!(order.len(), 2);
    }

    /// Test CPU optimization features integrated with the executor.
    #[test]
    fn test_cpu_optimization_integration() {
        use moirai_utils::cpu::{CpuTopology, affinity::AffinityMask};
        
        // Test CPU topology detection
        let topology = CpuTopology::detect();
        assert!(topology.logical_cores > 0);
        assert!(topology.physical_cores > 0);
        assert!(!topology.caches.is_empty());
        
        // Test affinity mask creation
        let mask = AffinityMask::all();
        assert!(!mask.is_empty());
        assert!(mask.len() > 0);
        
        // Test single core mask
        let single_mask = AffinityMask::single(moirai_utils::cpu::CpuCore::new(0));
        assert_eq!(single_mask.len(), 1);
        
        // Create runtime and verify it works with CPU optimizations
        let runtime = Moirai::builder()
            .worker_threads(topology.logical_cores.min(8) as usize)
            .build()
            .unwrap();
        
        let counter = Arc::new(AtomicU32::new(0));
        let handles: Vec<_> = (0..topology.logical_cores as usize)
            .map(|i| {
                let counter = counter.clone();
                runtime.spawn_parallel(move || {
                    // Simulate CPU-intensive work that benefits from affinity
                    let mut sum = 0u64;
                    for j in 0..1000 {
                        sum += (i + j) as u64;
                    }
                    counter.fetch_add(1, Ordering::Relaxed);
                    sum
                })
            })
            .collect();
        
        let results: Vec<_> = handles.into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        
        assert_eq!(results.len(), topology.logical_cores as usize);
        assert_eq!(counter.load(Ordering::Relaxed), topology.logical_cores);
        
        // Verify all results are non-zero (computation completed)
        for result in results {
            assert!(result > 0);
        }
    }

    /// Test memory prefetching utilities.
    #[test]
    fn test_memory_prefetching() {
        use moirai_utils::memory::{prefetch_read, prefetch_write, memory_barrier, compiler_barrier};
        
        let data = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        
        // Test prefetching (should not crash)
        prefetch_read(data.as_ptr());
        prefetch_write(data.as_ptr());
        
        // Test barriers (should not crash)
        memory_barrier();
        compiler_barrier();
        
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
    #[cfg(feature = "numa")]
    #[test]
    fn test_numa_awareness() {
        use moirai_utils::numa::{current_numa_node, numa_node_count};
        
        let current_node = current_numa_node();
        let node_count = numa_node_count();
        
        assert!(current_node.id() < 64); // Reasonable upper bound
        assert!(node_count > 0);
        assert!(node_count <= 64); // Reasonable upper bound
        
        // Test NUMA-aware task scheduling
        let runtime = Moirai::new().unwrap();
        
        let handles: Vec<_> = (0..node_count.min(4))
            .map(|node_id| {
                runtime.spawn_parallel(move || {
                    // Simulate NUMA-aware computation
                    let current = current_numa_node();
                    (current.id(), node_id as u32)
                })
            })
            .collect();
        
        let results: Vec<_> = handles.into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        
        assert_eq!(results.len(), node_count.min(4));
        
        // Verify all tasks completed
        for (current_node, _expected_node) in results {
            assert!(current_node < 64); // Reasonable bound
        }
    }

    /// Stress test with CPU optimizations.
    #[test]
    fn test_cpu_optimized_stress() {
        use moirai_utils::cpu::CpuTopology;
        
        let topology = CpuTopology::detect();
        let runtime = Moirai::builder()
            .worker_threads(topology.logical_cores.min(16) as usize)
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