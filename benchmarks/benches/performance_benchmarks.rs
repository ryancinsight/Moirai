use criterion::{black_box, criterion_group, criterion_main, Criterion};
use moirai::Moirai;
use moirai_core::constants::{
    BENCHMARK_PRIME_MODULO, CPU_UTILIZATION_PRECISION, DEFAULT_BENCHMARK_OPS, LARGE_BENCHMARK_SIZE,
    SIMD_BENCHMARK_SIZE,
};
use std::{sync::Arc, time::Duration};

const BENCHMARK_SAMPLE_SIZE: usize = 10;
const BENCHMARK_MEASUREMENT_SECONDS: u64 = 1;
const BENCHMARK_WARM_UP_MILLIS: u64 = 250;

fn expected_parallel_sum(seed: usize) -> usize {
    (0..DEFAULT_BENCHMARK_OPS)
        .map(|value| (seed * value) % BENCHMARK_PRIME_MODULO)
        .sum()
}

fn expected_stealing_sum(seed: usize, cost: usize) -> usize {
    (0..(cost * 1000)).map(|value| (seed * value) % 991).sum()
}

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
            let handle = runtime.spawn_fn(|| black_box(42));
            let result = handle
                .join()
                .expect("task handle must be attached")
                .expect("task must not fail");
            assert_eq!(result, 42);
            black_box(result)
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

        group.bench_with_input(format!("threads_{}", thread_count), thread_count, |b, _| {
            b.iter(|| {
                // Only measure the actual parallel computation
                let mut handles = Vec::with_capacity(CPU_UTILIZATION_PRECISION as usize);
                for i in 0..CPU_UTILIZATION_PRECISION as usize {
                    let handle = runtime.spawn_fn(move || {
                        // CPU-intensive computation
                        let mut sum = 0;
                        for j in 0..DEFAULT_BENCHMARK_OPS {
                            sum += (i * j) % BENCHMARK_PRIME_MODULO; // Prime modulo for variation
                        }
                        black_box(sum)
                    });
                    handles.push(handle);
                }

                // Wait for all tasks to complete
                for (i, handle) in handles.into_iter().enumerate() {
                    let result = handle
                        .join()
                        .expect("task handle must be attached")
                        .expect("task must not fail");
                    assert_eq!(result, expected_parallel_sum(i));
                    black_box(result);
                }
            });
        });

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
            let large_data = vec![42u64; LARGE_BENCHMARK_SIZE];
            let handle = runtime.spawn_fn(move || black_box(large_data.iter().sum::<u64>()));
            let result = handle
                .join()
                .expect("task handle must be attached")
                .expect("task must not fail");
            assert_eq!(result, 42u64 * LARGE_BENCHMARK_SIZE as u64);
            black_box(result)
        });
    });

    // Cleanup after all iterations
    runtime.shutdown();
}

/// Benchmark SIMD optimization performance improvement
/// No runtime needed for pure SIMD operations
fn benchmark_simd_performance(c: &mut Criterion) {
    use moirai_utils::simd::{add, mul};

    let mut group = c.benchmark_group("simd_performance");

    // Create test data ONCE outside all benchmarks
    let data_a = vec![1.0f32; SIMD_BENCHMARK_SIZE];
    let data_b = vec![2.0f32; SIMD_BENCHMARK_SIZE];

    // Scalar version benchmark
    group.bench_function("scalar_add", |b| {
        b.iter(|| {
            let mut result = vec![0.0f32; SIMD_BENCHMARK_SIZE];
            for i in 0..SIMD_BENCHMARK_SIZE {
                result[i] = data_a[i] + data_b[i];
            }
            assert_eq!(result, vec![3.0f32; SIMD_BENCHMARK_SIZE]);
            black_box(result)
        });
    });

    // SIMD version benchmark
    group.bench_function("simd_add", |b| {
        b.iter(|| {
            let mut result = vec![0.0f32; 1024];
            add(&data_a, &data_b, &mut result);
            assert_eq!(result, vec![3.0f32; 1024]);
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
            assert_eq!(result, vec![2.0f32; 1024]);
            black_box(result)
        });
    });

    // SIMD multiplication
    group.bench_function("simd_multiply", |b| {
        b.iter(|| {
            let mut result = vec![0.0f32; 1024];
            mul(&data_a, &data_b, &mut result);
            assert_eq!(result, vec![2.0f32; 1024]);
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark concurrent data structure performance
/// Runtime created once, data structures reset per iteration
fn benchmark_concurrent_data_structures(c: &mut Criterion) {
    use moirai_sync::ConcurrentHashMap;

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
            let counter = Arc::new(moirai_utils::AtomicCounter::new());

            let mut handles = Vec::with_capacity(100);
            for _ in 0..100 {
                let counter_clone = counter.clone();
                let handle = runtime.spawn_fn(move || {
                    for _ in 0..100 {
                        counter_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                });
                handles.push(handle);
            }

            // Wait for all tasks to complete
            for handle in handles {
                handle
                    .join()
                    .expect("task handle must be attached")
                    .expect("task must not fail");
            }

            let result = counter.get();
            assert_eq!(result, 10_000);
            black_box(result)
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
                let handle = runtime.spawn_fn(move || {
                    for j in 0..100 {
                        let key = format!("key_{}_{}", i, j);
                        map_clone.insert(key, i * j).expect("map insert failed");
                    }
                });
                handles.push(handle);
            }

            // Wait for all tasks to complete
            for handle in handles {
                handle
                    .join()
                    .expect("task handle must be attached")
                    .expect("task must not fail");
            }

            for i in 0..50 {
                for j in 0..100 {
                    let key = format!("key_{}_{}", i, j);
                    assert_eq!(map.get(&key).expect("map get failed"), Some(i * j));
                }
            }

            black_box(50 * 100)
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
                let handle = runtime.spawn_fn(move || {
                    let mut sum = 0;
                    for j in 0..(cost * 1000) {
                        sum += (i * j) % 991; // Different prime for variation
                    }
                    black_box(sum)
                });
                handles.push(handle);
            }

            // Wait for all tasks to complete
            for (i, handle) in handles.into_iter().enumerate() {
                let cost = (i % 10) + 1;
                let result = handle
                    .join()
                    .expect("task handle must be attached")
                    .expect("task must not fail");
                assert_eq!(result, expected_stealing_sum(i, cost));
                let _ = black_box(result);
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
                let handle = runtime.spawn_fn(move || -> Result<i32, &'static str> {
                    if i % 10 == 0 {
                        Err("intentional error")
                    } else {
                        Ok(i * i)
                    }
                });
                handles.push(handle);
            }

            // Process all results
            for (i, handle) in handles.into_iter().enumerate() {
                let result = handle
                    .join()
                    .expect("task handle must be attached")
                    .expect("task must not fail");
                if i % 10 == 0 {
                    assert_eq!(result, Err("intentional error"));
                } else {
                    assert_eq!(result, Ok((i * i) as i32));
                }
                let _ = black_box(result);
            }
        });
    });

    // Cleanup after all iterations
    runtime.shutdown();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(BENCHMARK_SAMPLE_SIZE)
        .measurement_time(Duration::from_secs(BENCHMARK_MEASUREMENT_SECONDS))
        .warm_up_time(Duration::from_millis(BENCHMARK_WARM_UP_MILLIS))
        .without_plots();
    targets =
        benchmark_task_scheduling_overhead,
        benchmark_parallel_scalability,
        benchmark_memory_efficiency,
        benchmark_simd_performance,
        benchmark_concurrent_data_structures,
        benchmark_work_stealing,
        benchmark_error_handling
}

criterion_main!(benches);
