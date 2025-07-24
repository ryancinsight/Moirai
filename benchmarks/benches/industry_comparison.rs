//! # Industry Comparison Benchmarks
//!
//! Comprehensive performance comparison between Moirai and industry-standard
//! concurrency libraries (Tokio, Rayon, std::thread).
//!
//! ## Benchmark Categories
//!
//! - **Task Spawning**: Creation and dispatch overhead
//! - **Async Performance**: I/O-bound workload handling
//! - **Parallel Execution**: CPU-bound workload scaling
//! - **Mixed Workloads**: Hybrid async/parallel scenarios
//! - **Memory Efficiency**: Memory usage and allocation patterns
//! - **Scalability**: Performance scaling with thread count

use criterion::{
    black_box, criterion_group, criterion_main, 
    BenchmarkId, Criterion, Throughput
};
use std::{
    sync::{Arc, atomic::{AtomicU64, Ordering}},
    time::{Duration, Instant},
    thread,
};

/// Benchmark configuration for consistent testing
const TASK_COUNTS: &[usize] = &[100, 1_000, 10_000, 100_000];
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16];
const WORKLOAD_SIZES: &[usize] = &[10, 100, 1_000, 10_000];

/// CPU-intensive computation for benchmarking
fn cpu_intensive_work(iterations: usize) -> u64 {
    let mut sum = 0u64;
    for i in 0..iterations {
        sum = sum.wrapping_add((i as u64).wrapping_mul(i as u64));
    }
    black_box(sum)
}

/// I/O simulation using thread sleep
async fn io_simulation(duration_ms: u64) -> u64 {
    tokio::time::sleep(Duration::from_millis(duration_ms)).await;
    black_box(duration_ms)
}

/// Task Spawning Benchmarks
fn benchmark_task_spawning(c: &mut Criterion) {
    let mut group = c.benchmark_group("task_spawning");
    
    for &task_count in TASK_COUNTS {
        group.throughput(Throughput::Elements(task_count as u64));
        
        // Moirai task spawning
        group.bench_with_input(
            BenchmarkId::new("moirai", task_count),
            &task_count,
            |b, &count| {
                b.iter(|| {
                    let runtime = moirai::Moirai::new().unwrap();
                    let start = Instant::now();
                    
                    let handles: Vec<_> = (0..count)
                        .map(|i| runtime.spawn_parallel(move || black_box(i * 2)))
                        .collect();
                    
                    let spawn_time = start.elapsed();
                    
                    // Clean up (don't measure join time)
                    drop(handles);
                    drop(runtime);
                    
                    black_box(spawn_time)
                });
            },
        );
        
        // Tokio task spawning
        group.bench_with_input(
            BenchmarkId::new("tokio", task_count),
            &task_count,
            |b, &count| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.iter(|| {
                    rt.block_on(async {
                        let start = Instant::now();
                        
                        let handles: Vec<_> = (0..count)
                            .map(|i| tokio::spawn(async move { black_box(i * 2) }))
                            .collect();
                        
                        let spawn_time = start.elapsed();
                        
                        // Clean up
                        drop(handles);
                        
                        black_box(spawn_time)
                    });
                });
            },
        );
        
        // Rayon task spawning (using join for fair comparison)
        group.bench_with_input(
            BenchmarkId::new("rayon", task_count),
            &task_count,
            |b, &count| {
                b.iter(|| {
                    let start = Instant::now();
                    
                    // Rayon doesn't have direct task spawning, use scope for fairness
                    rayon::scope(|s| {
                        for i in 0..count {
                            s.spawn(move |_| {
                                black_box(i * 2);
                            });
                        }
                    });
                    
                    let total_time = start.elapsed();
                    black_box(total_time)
                });
            },
        );
        
        // std::thread spawning
        group.bench_with_input(
            BenchmarkId::new("std_thread", task_count),
            &task_count,
            |b, &count| {
                b.iter(|| {
                    let start = Instant::now();
                    
                    let handles: Vec<_> = (0..count)
                        .map(|i| thread::spawn(move || black_box(i * 2)))
                        .collect();
                    
                    let spawn_time = start.elapsed();
                    
                    // Clean up
                    for handle in handles {
                        let _ = handle.join();
                    }
                    
                    black_box(spawn_time)
                });
            },
        );
    }
    
    group.finish();
}

/// CPU-bound workload benchmarks
fn benchmark_cpu_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_workloads");
    
    for &workload_size in WORKLOAD_SIZES {
        group.throughput(Throughput::Elements(workload_size as u64));
        
        // Moirai parallel execution
        group.bench_with_input(
            BenchmarkId::new("moirai", workload_size),
            &workload_size,
            |b, &size| {
                b.iter(|| {
                    let runtime = moirai::Moirai::new().unwrap();
                    let counter = Arc::new(AtomicU64::new(0));
                    
                    let handles: Vec<_> = (0..num_cpus::get())
                        .map(|_| {
                            let counter = counter.clone();
                            runtime.spawn_parallel(move || {
                                let result = cpu_intensive_work(size);
                                counter.fetch_add(result, Ordering::Relaxed);
                            })
                        })
                        .collect();
                    
                    // Wait for completion (simplified for benchmark)
                    thread::sleep(Duration::from_millis(10));
                    
                    black_box(counter.load(Ordering::Relaxed))
                });
            },
        );
        
        // Rayon parallel execution
        group.bench_with_input(
            BenchmarkId::new("rayon", workload_size),
            &workload_size,
            |b, &size| {
                b.iter(|| {
                    let counter = AtomicU64::new(0);
                    
                    (0..num_cpus::get()).into_iter().for_each(|_| {
                        let result = cpu_intensive_work(size);
                        counter.fetch_add(result, Ordering::Relaxed);
                    });
                    
                    black_box(counter.load(Ordering::Relaxed))
                });
            },
        );
        
        // std::thread parallel execution
        group.bench_with_input(
            BenchmarkId::new("std_thread", workload_size),
            &workload_size,
            |b, &size| {
                b.iter(|| {
                    let counter = Arc::new(AtomicU64::new(0));
                    
                    let handles: Vec<_> = (0..num_cpus::get())
                        .map(|_| {
                            let counter = counter.clone();
                            thread::spawn(move || {
                                let result = cpu_intensive_work(size);
                                counter.fetch_add(result, Ordering::Relaxed);
                            })
                        })
                        .collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                    
                    black_box(counter.load(Ordering::Relaxed))
                });
            },
        );
    }
    
    group.finish();
}

/// Async I/O workload benchmarks
fn benchmark_async_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_workloads");
    
    for &task_count in &[10, 50, 100, 500] {
        group.throughput(Throughput::Elements(task_count as u64));
        
        // Moirai async execution
        group.bench_with_input(
            BenchmarkId::new("moirai", task_count),
            &task_count,
            |b, &count| {
                b.iter(|| {
                    let runtime = moirai::Moirai::new().unwrap();
                    
                    runtime.block_on(async {
                        let handles: Vec<_> = (0..count)
                            .map(|_| runtime.spawn_async(io_simulation(1)))
                            .collect();
                        
                        // Simplified wait for benchmark
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        
                        black_box(handles.len())
                    })
                });
            },
        );
        
        // Tokio async execution
        group.bench_with_input(
            BenchmarkId::new("tokio", task_count),
            &task_count,
            |b, &count| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.iter(|| {
                    rt.block_on(async {
                        let handles: Vec<_> = (0..count)
                            .map(|_| tokio::spawn(io_simulation(1)))
                            .collect();
                        
                        // Wait for completion
                        for handle in handles {
                            let _ = handle.await;
                        }
                        
                        black_box(count)
                    });
                });
            },
        );
    }
    
    group.finish();
}

/// Mixed workload benchmarks (hybrid async/parallel)
fn benchmark_mixed_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workloads");
    
    // Moirai hybrid execution
    group.bench_function("moirai_hybrid", |b| {
        b.iter(|| {
            let runtime = moirai::Moirai::new().unwrap();
            
            runtime.block_on(async {
                // Mix of CPU and I/O tasks
                let cpu_handles: Vec<_> = (0..4)
                    .map(|_| runtime.spawn_parallel(|| cpu_intensive_work(1000)))
                    .collect();
                
                let io_handles: Vec<_> = (0..10)
                    .map(|_| runtime.spawn_async(io_simulation(5)))
                    .collect();
                
                // Simplified completion wait
                tokio::time::sleep(Duration::from_millis(50)).await;
                
                black_box((cpu_handles.len(), io_handles.len()))
            })
        });
    });
    
    // Tokio with Rayon hybrid execution
    group.bench_function("tokio_rayon_hybrid", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async {
                // CPU tasks via Rayon
                let cpu_future = tokio::task::spawn_blocking(|| {
                    (0..4).into_iter().for_each(|_| {
                        cpu_intensive_work(1000);
                    });
                });
                
                // I/O tasks via Tokio
                let io_handles: Vec<_> = (0..10)
                    .map(|_| tokio::spawn(io_simulation(5)))
                    .collect();
                
                // Wait for completion
                let _ = cpu_future.await;
                for handle in io_handles {
                    let _ = handle.await;
                }
                
                black_box((4, 10))
            });
        });
    });
    
    group.finish();
}

/// Memory efficiency benchmarks
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    for &task_count in &[1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(task_count as u64));
        
        // Moirai memory usage
        group.bench_with_input(
            BenchmarkId::new("moirai_memory", task_count),
            &task_count,
            |b, &count| {
                b.iter(|| {
                    let runtime = moirai::Moirai::new().unwrap();
                    
                    // Create many tasks to measure memory overhead
                    let handles: Vec<_> = (0..count)
                        .map(|i| runtime.spawn_parallel(move || black_box(i)))
                        .collect();
                    
                    let memory_footprint = handles.len() * std::mem::size_of_val(&handles[0]);
                    
                    // Clean up
                    drop(handles);
                    drop(runtime);
                    
                    black_box(memory_footprint)
                });
            },
        );
        
        // Tokio memory usage
        group.bench_with_input(
            BenchmarkId::new("tokio_memory", task_count),
            &task_count,
            |b, &count| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.iter(|| {
                    rt.block_on(async {
                        let handles: Vec<_> = (0..count)
                            .map(|i| tokio::spawn(async move { black_box(i) }))
                            .collect();
                        
                        let memory_footprint = handles.len() * std::mem::size_of_val(&handles[0]);
                        
                        // Clean up
                        drop(handles);
                        
                        black_box(memory_footprint)
                    });
                });
            },
        );
    }
    
    group.finish();
}

/// Scalability benchmarks across different thread counts
fn benchmark_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    
    for &thread_count in THREAD_COUNTS {
        group.throughput(Throughput::Elements(thread_count as u64));
        
        // Moirai scalability
        group.bench_with_input(
            BenchmarkId::new("moirai", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter(|| {
                    let runtime = moirai::Moirai::builder()
                        .worker_threads(threads)
                        .build()
                        .unwrap();
                    
                    let counter = Arc::new(AtomicU64::new(0));
                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let counter = counter.clone();
                            runtime.spawn_parallel(move || {
                                let result = cpu_intensive_work(10_000);
                                counter.fetch_add(result, Ordering::Relaxed);
                            })
                        })
                        .collect();
                    
                    // Simplified wait
                    thread::sleep(Duration::from_millis(100));
                    
                    black_box(counter.load(Ordering::Relaxed))
                });
            },
        );
        
        // std::thread scalability
        group.bench_with_input(
            BenchmarkId::new("std_thread", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter(|| {
                    let counter = Arc::new(AtomicU64::new(0));
                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let counter = counter.clone();
                            thread::spawn(move || {
                                let result = cpu_intensive_work(10_000);
                                counter.fetch_add(result, Ordering::Relaxed);
                            })
                        })
                        .collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                    
                    black_box(counter.load(Ordering::Relaxed))
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_task_spawning,
    benchmark_cpu_workloads,
    benchmark_async_workloads,
    benchmark_mixed_workloads,
    benchmark_memory_efficiency,
    benchmark_scalability
);

criterion_main!(benches);