use criterion::{black_box, criterion_group, criterion_main, Criterion};
use moirai::Moirai;
use std::sync::Arc;

/// Benchmark task scheduling overhead - should be < 1μs per task
/// Runtime created once outside the benchmark loop
fn benchmark_task_scheduling_overhead(c: &mut Criterion) {
    // Create runtime ONCE outside the benchmark
    let runtime = Moirai::builder()
        .worker_threads(4)
        .build()
        .expect("Failed to create runtime");

    c.bench_function("task_scheduling_overhead", |b| {
        b.iter(|| {
            // Only measure the actual task scheduling and execution
            let handle = runtime.spawn_parallel(|| black_box(42));
            black_box(handle.join().expect("Task failed"))
        });
    });

    // Cleanup after all iterations
    runtime.shutdown();
}

/// Benchmark parallel computation scalability
/// Each thread count gets its own runtime instance
fn benchmark_parallel_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_scalability");
    
    for thread_count in [1, 2, 4, 8].iter() {
        // Create runtime ONCE per thread count configuration
        let runtime = Moirai::builder()
            .worker_threads(*thread_count)
            .build()
            .expect("Failed to create runtime");

        group.bench_with_input(
            format!("threads_{}", thread_count),
            thread_count,
            |b, _| {
                b.iter(|| {
                    // Only measure the actual parallel computation
                    let mut handles = Vec::with_capacity(100);
                    for i in 0..100 {
                        let handle = runtime.spawn_parallel(move || {
                            // CPU-intensive computation
                            let mut sum = 0;
                            for j in 0..1000 {
                                sum += (i * j) % 997; // Prime modulo for variation
                            }
                            black_box(sum)
                        });
                        handles.push(handle);
                    }
                    
                    // Wait for all tasks to complete
                    for handle in handles {
                        black_box(handle.join().expect("Task failed"));
                    }
                });
            },
        );

        // Cleanup after this thread count is done
        runtime.shutdown();
    }
    
    group.finish();
}

/// Benchmark memory efficiency with zero-copy task passing
/// Runtime created once, data recreated per iteration
fn benchmark_memory_efficiency(c: &mut Criterion) {
    // Create runtime ONCE outside the benchmark
    let runtime = Moirai::builder()
        .worker_threads(4)
        .build()
        .expect("Failed to create runtime");

    c.bench_function("memory_efficiency_large_data", |b| {
        b.iter(|| {
            // Create fresh data for each iteration to avoid state carryover
            let large_data = vec![42u64; 10000];
            let handle = runtime.spawn_parallel(move || {
                black_box(large_data.iter().sum::<u64>())
            });
            black_box(handle.join().expect("Task failed"))
        });
    });

    // Cleanup after all iterations
    runtime.shutdown();
}

/// Benchmark SIMD optimization performance improvement
/// No runtime needed for pure SIMD operations
fn benchmark_simd_performance(c: &mut Criterion) {
    use moirai_utils::simd::{safe_vectorized_add_f32, safe_vectorized_mul_f32};
    
    let mut group = c.benchmark_group("simd_performance");
    
    // Create test data ONCE outside all benchmarks
    let data_a = vec![1.0f32; 1024];
    let data_b = vec![2.0f32; 1024];
    
    // Scalar version benchmark
    group.bench_function("scalar_add", |b| {
        b.iter(|| {
            let mut result = vec![0.0f32; 1024];
            for i in 0..1024 {
                result[i] = data_a[i] + data_b[i];
            }
            black_box(result)
        });
    });
    
    // SIMD version benchmark
    group.bench_function("simd_add", |b| {
        b.iter(|| {
            let mut result = vec![0.0f32; 1024];
            safe_vectorized_add_f32(&data_a, &data_b, &mut result);
            black_box(result)
        });
    });
    
    // Scalar multiplication
    group.bench_function("scalar_multiply", |b| {
        b.iter(|| {
            let mut result = vec![0.0f32; 1024];
            for i in 0..1024 {
                result[i] = data_a[i] * data_b[i];
            }
            black_box(result)
        });
    });
    
    // SIMD multiplication
    group.bench_function("simd_multiply", |b| {
        b.iter(|| {
            let mut result = vec![0.0f32; 1024];
            safe_vectorized_mul_f32(&data_a, &data_b, &mut result);
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark concurrent data structure performance
/// Runtime created once, data structures reset per iteration
fn benchmark_concurrent_data_structures(c: &mut Criterion) {
    use moirai_sync::{AtomicCounter, ConcurrentHashMap};
    
    let mut group = c.benchmark_group("concurrent_data_structures");
    
    // Create runtime ONCE outside all benchmarks
    let runtime = Moirai::builder()
        .worker_threads(4)
        .build()
        .expect("Failed to create runtime");
    
    // AtomicCounter performance
    group.bench_function("atomic_counter", |b| {
        b.iter(|| {
            // Create fresh counter for each iteration to avoid state carryover
            let counter = Arc::new(AtomicCounter::new(0));
            
            let mut handles = Vec::with_capacity(100);
            for _ in 0..100 {
                let counter_clone = counter.clone();
                let handle = runtime.spawn_parallel(move || {
                    for _ in 0..100 {
                        counter_clone.increment();
                    }
                });
                handles.push(handle);
            }
            
            // Wait for all tasks to complete
            for handle in handles {
                handle.join().expect("Task failed");
            }
            
            black_box(counter.get())
        });
    });
    
    // ConcurrentHashMap performance
    group.bench_function("concurrent_hashmap", |b| {
        b.iter(|| {
            // Create fresh map for each iteration to avoid state carryover
            let map = Arc::new(ConcurrentHashMap::new());
            
            let mut handles = Vec::with_capacity(50);
            for i in 0..50 {
                let map_clone = map.clone();
                let handle = runtime.spawn_parallel(move || {
                    for j in 0..100 {
                        let key = format!("key_{}_{}", i, j);
                        map_clone.insert(key, i * j);
                    }
                });
                handles.push(handle);
            }
            
            // Wait for all tasks to complete
            for handle in handles {
                handle.join().expect("Task failed");
            }
            
            black_box(map.len())
        });
    });
    
    // Cleanup after all concurrent data structure benchmarks
    runtime.shutdown();
    group.finish();
}

/// Benchmark work-stealing scheduler efficiency
/// Runtime created once outside the benchmark
fn benchmark_work_stealing(c: &mut Criterion) {
    // Create runtime ONCE outside the benchmark
    let runtime = Moirai::builder()
        .worker_threads(4)
        .build()
        .expect("Failed to create runtime");

    c.bench_function("work_stealing_efficiency", |b| {
        b.iter(|| {
            let mut handles = Vec::with_capacity(200);
            
            // Create tasks with varying computational costs
            for i in 0..200 {
                let cost = (i % 10) + 1; // Varying cost from 1 to 10
                let handle = runtime.spawn_parallel(move || {
                    let mut sum = 0;
                    for j in 0..(cost * 1000) {
                        sum += (i * j) % 991; // Different prime for variation
                    }
                    black_box(sum)
                });
                handles.push(handle);
            }
            
            // Wait for all tasks to complete
            for handle in handles {
                black_box(handle.join().expect("Task failed"));
            }
        });
    });

    // Cleanup after all iterations
    runtime.shutdown();
}

/// Benchmark error handling performance
/// Runtime created once, fresh error scenarios per iteration
fn benchmark_error_handling(c: &mut Criterion) {
    // Create runtime ONCE outside the benchmark
    let runtime = Moirai::builder()
        .worker_threads(2)
        .build()
        .expect("Failed to create runtime");

    c.bench_function("error_handling_overhead", |b| {
        b.iter(|| {
            let mut handles = Vec::with_capacity(100);
            
            // Mix of successful and failing tasks (fresh each iteration)
            for i in 0..100 {
                let handle = runtime.spawn_parallel(move || -> Result<i32, &'static str> {
                    if i % 10 == 0 {
                        Err("intentional error")
                    } else {
                        Ok(i * i)
                    }
                });
                handles.push(handle);
            }
            
            // Process all results
            for handle in handles {
                let result = handle.join().expect("Task join failed");
                let _ = black_box(result);
            }
        });
    });

    // Cleanup after all iterations
    runtime.shutdown();
}

criterion_group!(
    benches,
    benchmark_task_scheduling_overhead,
    benchmark_parallel_scalability,
    benchmark_memory_efficiency,
    benchmark_simd_performance,
    benchmark_concurrent_data_structures,
    benchmark_work_stealing,
    benchmark_error_handling
);

criterion_main!(benches);