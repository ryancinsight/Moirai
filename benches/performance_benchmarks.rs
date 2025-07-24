use criterion::{black_box, criterion_group, criterion_main, Criterion};
use moirai::Moirai;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Benchmark task scheduling overhead - should be < 1μs per task
fn benchmark_task_scheduling_overhead(c: &mut Criterion) {
    let runtime = Moirai::builder()
        .worker_threads(4)
        .build()
        .expect("Failed to create runtime");

    c.bench_function("task_scheduling_overhead", |b| {
        b.iter(|| {
            let handle = runtime.spawn_parallel(|| black_box(42));
            black_box(handle.join().expect("Task failed"));
        });
    });

    runtime.shutdown();
}

/// Benchmark parallel computation scalability
fn benchmark_parallel_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_scalability");
    
    for thread_count in [1, 2, 4, 8].iter() {
        let runtime = Moirai::builder()
            .worker_threads(*thread_count)
            .build()
            .expect("Failed to create runtime");

        group.bench_with_input(
            format!("threads_{}", thread_count),
            thread_count,
            |b, _| {
                b.iter(|| {
                    let mut handles = Vec::new();
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
                    
                    for handle in handles {
                        black_box(handle.join().expect("Task failed"));
                    }
                });
            },
        );

        runtime.shutdown();
    }
    
    group.finish();
}

/// Benchmark memory efficiency with zero-copy task passing
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let runtime = Moirai::builder()
        .worker_threads(4)
        .build()
        .expect("Failed to create runtime");

    c.bench_function("memory_efficiency_large_data", |b| {
        b.iter(|| {
            // Large data structure to test zero-copy efficiency
            let large_data = vec![42u64; 10000];
            let handle = runtime.spawn_parallel(move || {
                black_box(large_data.iter().sum::<u64>())
            });
            black_box(handle.join().expect("Task failed"));
        });
    });

    runtime.shutdown();
}

/// Benchmark SIMD optimization performance improvement
fn benchmark_simd_performance(c: &mut Criterion) {
    use moirai_utils::simd::{vectorized_add, vectorized_multiply};
    
    let mut group = c.benchmark_group("simd_performance");
    
    // Test data for SIMD operations
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
            let result = vectorized_add(&data_a, &data_b);
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
            let result = vectorized_multiply(&data_a, &data_b);
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark concurrent data structure performance
fn benchmark_concurrent_data_structures(c: &mut Criterion) {
    use moirai_sync::{AtomicCounter, ConcurrentHashMap};
    
    let mut group = c.benchmark_group("concurrent_data_structures");
    
    // AtomicCounter performance
    group.bench_function("atomic_counter", |b| {
        let counter = Arc::new(AtomicCounter::new());
        b.iter(|| {
            let counter_clone = counter.clone();
            let runtime = Moirai::new().expect("Failed to create runtime");
            
            let mut handles = Vec::new();
            for _ in 0..100 {
                let counter = counter_clone.clone();
                let handle = runtime.spawn_parallel(move || {
                    for _ in 0..100 {
                        counter.increment();
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().expect("Task failed");
            }
            
            runtime.shutdown();
            black_box(counter_clone.get())
        });
    });
    
    // ConcurrentHashMap performance
    group.bench_function("concurrent_hashmap", |b| {
        let map = Arc::new(ConcurrentHashMap::new());
        b.iter(|| {
            let map_clone = map.clone();
            let runtime = Moirai::new().expect("Failed to create runtime");
            
            let mut handles = Vec::new();
            for i in 0..50 {
                let map = map_clone.clone();
                let handle = runtime.spawn_parallel(move || {
                    for j in 0..100 {
                        let key = format!("key_{}_{}", i, j);
                        map.insert(key, i * j);
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().expect("Task failed");
            }
            
            runtime.shutdown();
            black_box(map_clone.len())
        });
    });
    
    group.finish();
}

/// Benchmark work-stealing scheduler efficiency
fn benchmark_work_stealing(c: &mut Criterion) {
    let runtime = Moirai::builder()
        .worker_threads(4)
        .build()
        .expect("Failed to create runtime");

    c.bench_function("work_stealing_efficiency", |b| {
        b.iter(|| {
            let mut handles = Vec::new();
            
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
            
            for handle in handles {
                black_box(handle.join().expect("Task failed"));
            }
        });
    });

    runtime.shutdown();
}

/// Benchmark error handling performance
fn benchmark_error_handling(c: &mut Criterion) {
    let runtime = Moirai::builder()
        .worker_threads(2)
        .build()
        .expect("Failed to create runtime");

    c.bench_function("error_handling_overhead", |b| {
        b.iter(|| {
            let mut handles = Vec::new();
            
            // Mix of successful and failing tasks
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
            
            for handle in handles {
                let result = handle.join().expect("Task join failed");
                black_box(result);
            }
        });
    });

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