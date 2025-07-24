//! Comprehensive benchmarking suite for Moirai concurrency library.
//!
//! This module provides industry-standard benchmarks to measure and compare
//! the performance of Moirai against other concurrency libraries.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use moirai::*;
use moirai_core::{TaskBuilder, Priority};
use moirai_utils::simd;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Benchmark task spawning performance.
fn bench_task_spawning(c: &mut Criterion) {
    let mut group = c.benchmark_group("task_spawning");
    
    for &count in &[100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::new("moirai", count), &count, |b, &count| {
            b.iter(|| {
                let moirai = Moirai::new().unwrap();
                
                for i in 0..count {
                    let task = TaskBuilder::new()
                        .name("bench_task")
                        .build(move || black_box(i * 2));
                    
                    let _handle = moirai.spawn(task);
                }
                
                moirai.shutdown();
            });
        });
        
        // Compare with std::thread spawning
        group.bench_with_input(BenchmarkId::new("std_thread", count), &count, |b, &count| {
            b.iter(|| {
                let handles: Vec<_> = (0..count)
                    .map(|i| {
                        thread::spawn(move || black_box(i * 2))
                    })
                    .collect();
                
                for handle in handles {
                    let _ = handle.join();
                }
            });
        });
    }
    
    group.finish();
}

/// Benchmark async task performance.
fn bench_async_tasks(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_tasks");
    
    for &count in &[100, 1000, 5000] {
        group.bench_with_input(BenchmarkId::new("moirai_async", count), &count, |b, &count| {
            b.iter(|| {
                let moirai = Moirai::new().unwrap();
                let mut handles = Vec::new();
                
                for i in 0..count {
                    let handle = moirai.spawn_async(async move {
                        black_box(i * 2)
                    });
                    handles.push(handle);
                }
                
                // Wait for completion
                for handle in handles {
                    let _ = handle.try_join();
                }
                
                moirai.shutdown();
            });
        });
    }
    
    group.finish();
}

/// Benchmark work-stealing performance.
fn bench_work_stealing(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_stealing");
    
    for &worker_count in &[2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("parallel_work", worker_count), 
            &worker_count, 
            |b, &worker_count| {
                b.iter(|| {
                    let moirai = Moirai::builder()
                        .worker_threads(worker_count)
                        .build()
                        .unwrap();
                    
                    let counter = Arc::new(Mutex::new(0));
                    let mut handles = Vec::new();
                    
                    // Create many small tasks to trigger work stealing
                    for i in 0..1000 {
                        let counter_clone = counter.clone();
                        let task = TaskBuilder::new()
                            .build(move || {
                                let mut count = counter_clone.lock().unwrap();
                                *count += i;
                                black_box(*count);
                            });
                        
                        handles.push(moirai.spawn(task));
                    }
                    
                    // Wait for completion
                    for handle in handles {
                        let _ = handle.try_join();
                    }
                    
                    moirai.shutdown();
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark priority scheduling.
fn bench_priority_scheduling(c: &mut Criterion) {
    let mut group = c.benchmark_group("priority_scheduling");
    
    group.bench_function("mixed_priorities", |b| {
        b.iter(|| {
            let moirai = Moirai::new().unwrap();
            let mut handles = Vec::new();
            
            // Mix of different priority tasks
            for i in 0..100 {
                let priority = match i % 3 {
                    0 => Priority::High,
                    1 => Priority::Normal,
                    _ => Priority::Low,
                };
                
                let task = TaskBuilder::new()
                    .priority(priority)
                    .build(move || black_box(i * i));
                
                handles.push(moirai.spawn(task));
            }
            
            // Wait for completion
            for handle in handles {
                let _ = handle.try_join();
            }
            
            moirai.shutdown();
        });
    });
    
    group.finish();
}

/// Benchmark SIMD operations.
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");
    
    for &size in &[64, 512, 4096] {
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();
        let mut result = vec![0.0; size];
        
        // Benchmark scalar addition
        group.bench_with_input(
            BenchmarkId::new("scalar_add", size), 
            &size, 
            |bench, _| {
                bench.iter(|| {
                    for i in 0..size {
                        result[i] = a[i] + b[i];
                    }
                    black_box(&result);
                });
            }
        );
        
        // Benchmark SIMD addition (if available)
        if simd::has_avx2_support() && size % 8 == 0 {
            group.bench_with_input(
                BenchmarkId::new("simd_add", size), 
                &size, 
                |bench, _| {
                    bench.iter(|| {
                        unsafe {
                            simd::vectorized_add_f32(&a, &b, &mut result);
                        }
                        black_box(&result);
                    });
                }
            );
        }
        
        // Benchmark safe vectorized addition
        group.bench_with_input(
            BenchmarkId::new("safe_simd_add", size), 
            &size, 
            |bench, _| {
                bench.iter(|| {
                    simd::safe_vectorized_add_f32(&a, &b, &mut result);
                    black_box(&result);
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark memory allocation patterns.
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    
    group.bench_function("task_metadata_allocation", |b| {
        b.iter(|| {
            let moirai = Moirai::new().unwrap();
            
            // Create and destroy many tasks to test allocation patterns
            for i in 0..100 {
                let task = TaskBuilder::new()
                    .name("alloc_test")
                    .build(move || black_box(i));
                
                let handle = moirai.spawn(task);
                let _ = handle.try_join();
            }
            
            moirai.shutdown();
        });
    });
    
    group.finish();
}

/// Benchmark synchronization primitives.
fn bench_synchronization(c: &mut Criterion) {
    let mut group = c.benchmark_group("synchronization");
    
    group.bench_function("lock_contention", |b| {
        b.iter(|| {
            let moirai = Moirai::builder()
                .worker_threads(4)
                .build()
                .unwrap();
            
            let shared_data = Arc::new(Mutex::new(0));
            let mut handles = Vec::new();
            
            for i in 0..50 {
                let data_clone = shared_data.clone();
                let task = TaskBuilder::new()
                    .build(move || {
                        let mut data = data_clone.lock().unwrap();
                        *data += i;
                        black_box(*data);
                    });
                
                handles.push(moirai.spawn(task));
            }
            
            for handle in handles {
                let _ = handle.try_join();
            }
            
            moirai.shutdown();
        });
    });
    
    group.finish();
}

/// Comprehensive performance regression test.
fn bench_performance_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_regression");
    
    group.bench_function("baseline_performance", |b| {
        b.iter(|| {
            let moirai = Moirai::builder()
                .worker_threads(4)
                .async_threads(2)
                .build()
                .unwrap();
            
            let mut handles = Vec::new();
            
            // Mixed workload: CPU tasks, async tasks, and I/O simulation
            for i in 0..200 {
                if i % 3 == 0 {
                    // CPU-intensive task
                    let task = TaskBuilder::new()
                        .priority(Priority::High)
                        .build(move || {
                            let mut sum = 0;
                            for j in 0..100 {
                                sum += j * i;
                            }
                            black_box(sum)
                        });
                    handles.push(moirai.spawn(task));
                } else if i % 3 == 1 {
                    // Async task
                    let handle = moirai.spawn_async(async move {
                        tokio::time::sleep(Duration::from_micros(1)).await;
                        black_box(i * 42)
                    });
                    handles.push(handle);
                } else {
                    // Blocking I/O simulation
                    let handle = moirai.spawn_blocking(move || {
                        thread::sleep(Duration::from_micros(1));
                        black_box(i * 3)
                    });
                    handles.push(handle);
                }
            }
            
            // Wait for all tasks to complete
            for handle in handles {
                let _ = handle.try_join();
            }
            
            moirai.shutdown();
        });
    });
    
    group.finish();
}

/// Latency measurement benchmarks.
fn bench_latency_measurements(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency");
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("task_spawn_latency", |b| {
        let moirai = Moirai::new().unwrap();
        
        b.iter(|| {
            let task = TaskBuilder::new()
                .build(|| black_box(42));
            
            let _handle = moirai.spawn(task);
        });
        
        moirai.shutdown();
    });
    
    group.bench_function("async_spawn_latency", |b| {
        let moirai = Moirai::new().unwrap();
        
        b.iter(|| {
            let _handle = moirai.spawn_async(async {
                black_box(42)
            });
        });
        
        moirai.shutdown();
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_task_spawning,
    bench_async_tasks,
    bench_work_stealing,
    bench_priority_scheduling,
    bench_simd_operations,
    bench_memory_allocation,
    bench_synchronization,
    bench_performance_regression,
    bench_latency_measurements
);

criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_infrastructure() {
        // Verify that benchmarking infrastructure works
        let moirai = Moirai::new().unwrap();
        
        let task = TaskBuilder::new()
            .name("test_task")
            .build(|| 42);
        
        let handle = moirai.spawn(task);
        let result = handle.try_join();
        
        assert!(result.is_some());
        moirai.shutdown();
    }
    
    #[test]
    fn test_simd_benchmarks() {
        // Test SIMD benchmark setup
        let a = vec![1.0; 64];
        let b = vec![2.0; 64];
        let mut result = vec![0.0; 64];
        
        simd::safe_vectorized_add_f32(&a, &b, &mut result);
        
        for &val in &result {
            assert_eq!(val, 3.0);
        }
    }
    
    #[test]
    fn test_performance_regression_setup() {
        // Verify performance regression test setup
        let moirai = Moirai::builder()
            .worker_threads(2)
            .build()
            .unwrap();
        
        let mut handles = Vec::new();
        
        for i in 0..10 {
            let task = TaskBuilder::new()
                .build(move || i * 2);
            handles.push(moirai.spawn(task));
        }
        
        for handle in handles {
            let result = handle.try_join();
            assert!(result.is_some());
        }
        
        moirai.shutdown();
    }
}