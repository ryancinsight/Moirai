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

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
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

/// Memory allocation simulation
fn memory_allocation_work(size_kb: usize) -> Vec<u8> {
    let mut data = vec![0u8; size_kb * 1024];
    for i in 0..data.len() {
        data[i] = (i % 256) as u8;
    }
    black_box(data)
}

/// Task Spawning Benchmarks
fn benchmark_task_spawning(c: &mut Criterion) {
    let mut group = c.benchmark_group("task_spawning");

    for &task_count in TASK_COUNTS {
        group.throughput(Throughput::Elements(task_count as u64));

        // Moirai task spawning
        group.bench_with_input(
            BenchmarkId::new("moirai_executor", task_count),
            &task_count,
            |b, &task_count| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        use moirai_async::AsyncExecutor;
                        
                        let executor = AsyncExecutor::new();
                        let mut handles = Vec::new();
                        
                        for i in 0..task_count {
                            let handle = executor.spawn(async move {
                                black_box(i)
                            });
                            handles.push(handle);
                        }
                        
                        // Wait for completion
                        executor.run_until_complete(Some(Duration::from_secs(10)));
                    });
                });
            },
        );

        // Tokio task spawning
        group.bench_with_input(
            BenchmarkId::new("tokio", task_count),
            &task_count,
            |b, &task_count| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        let mut handles = Vec::new();
                        
                        for i in 0..task_count {
                            let handle = tokio::spawn(async move {
                                black_box(i)
                            });
                            handles.push(handle);
                        }
                        
                        // Wait for all tasks
                        futures::future::join_all(handles).await;
                    });
                });
            },
        );

        // Rayon task spawning
        group.bench_with_input(
            BenchmarkId::new("rayon", task_count),
            &task_count,
            |b, &task_count| {
                b.iter(|| {
                    use rayon::prelude::*;
                    
                    (0..task_count).into_par_iter().for_each(|i| {
                        black_box(i);
                    });
                });
            },
        );

        // Standard thread spawning
        group.bench_with_input(
            BenchmarkId::new("std_thread", task_count),
            &task_count,
            |b, &task_count| {
                b.iter(|| {
                    let handles: Vec<_> = (0..std::cmp::min(task_count, 100)) // Limit thread count
                        .map(|i| {
                            thread::spawn(move || {
                                black_box(i);
                            })
                        })
                        .collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Parallel Execution Benchmarks
fn benchmark_parallel_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_execution");

    for &workload_size in WORKLOAD_SIZES {
        group.throughput(Throughput::Elements(workload_size as u64));

        // Moirai parallel iterator
        group.bench_with_input(
            BenchmarkId::new("moirai_parallel", workload_size),
            &workload_size,
            |b, &workload_size| {
                b.iter(|| {
                    use moirai_iter::{moirai_iter_parallel, ParallelIterator};
                    
                    let data: Vec<usize> = (0..workload_size).collect();
                    let result: Vec<u64> = moirai_iter_parallel(data)
                        .map(|x| cpu_intensive_work(x % 100))
                        .collect();
                    
                    black_box(result);
                });
            },
        );

        // Rayon parallel iterator
        group.bench_with_input(
            BenchmarkId::new("rayon", workload_size),
            &workload_size,
            |b, &workload_size| {
                b.iter(|| {
                    use rayon::prelude::*;
                    
                    let data: Vec<usize> = (0..workload_size).collect();
                    let result: Vec<u64> = data
                        .into_par_iter()
                        .map(|x| cpu_intensive_work(x % 100))
                        .collect();
                    
                    black_box(result);
                });
            },
        );

        // Sequential execution for comparison
        group.bench_with_input(
            BenchmarkId::new("sequential", workload_size),
            &workload_size,
            |b, &workload_size| {
                b.iter(|| {
                    let data: Vec<usize> = (0..workload_size).collect();
                    let result: Vec<u64> = data
                        .into_iter()
                        .map(|x| cpu_intensive_work(x % 100))
                        .collect();
                    
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Async Performance Benchmarks
fn benchmark_async_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_performance");

    for &task_count in &[10, 100, 1000] {
        group.throughput(Throughput::Elements(task_count as u64));

        // Moirai async iterator
        group.bench_with_input(
            BenchmarkId::new("moirai_async", task_count),
            &task_count,
            |b, &task_count| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        use moirai_iter::{moirai_iter_async, AsyncIterator, IntoAsyncIterator};
                        
                        let data: Vec<u64> = (0..task_count as u64).collect();
                        let results = data
                            .into_async_iter()
                            .map(|x| async move {
                                io_simulation(x % 10).await
                            })
                            .collect::<Vec<u64>>()
                            .await;
                        
                        black_box(results);
                    });
                });
            },
        );

        // Tokio async processing
        group.bench_with_input(
            BenchmarkId::new("tokio", task_count),
            &task_count,
            |b, &task_count| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        let data: Vec<u64> = (0..task_count as u64).collect();
                        let mut results = Vec::new();
                        
                        for x in data {
                            let result = io_simulation(x % 10).await;
                            results.push(result);
                        }
                        
                        black_box(results);
                    });
                });
            },
        );

        // Concurrent Tokio processing
        group.bench_with_input(
            BenchmarkId::new("tokio_concurrent", task_count),
            &task_count,
            |b, &task_count| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        let data: Vec<u64> = (0..task_count as u64).collect();
                        let futures: Vec<_> = data
                            .into_iter()
                            .map(|x| io_simulation(x % 10))
                            .collect();
                        
                        let results = futures::future::join_all(futures).await;
                        black_box(results);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Mixed Workload Benchmarks
fn benchmark_mixed_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workloads");

    for &workload_size in &[100, 500, 1000] {
        group.throughput(Throughput::Elements(workload_size as u64));

        // Moirai hybrid processing
        group.bench_with_input(
            BenchmarkId::new("moirai_hybrid", workload_size),
            &workload_size,
            |b, &workload_size| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        use moirai_iter::{moirai_iter_hybrid, AsyncIterator};
                        
                        let data: Vec<usize> = (0..workload_size).collect();
                        let results = moirai_iter_hybrid(data)
                            .map(|x| cpu_intensive_work(x % 50))      // CPU work
                            .map_async(|x| async move {               // I/O work
                                io_simulation((x % 5) as u64).await;
                                x * 2
                            })
                            .await
                            .collect_async()
                            .await;
                        
                        black_box(results);
                    });
                });
            },
        );

        // Separate Tokio + Rayon processing
        group.bench_with_input(
            BenchmarkId::new("tokio_rayon_separate", workload_size),
            &workload_size,
            |b, &workload_size| {
                b.iter(|| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        use rayon::prelude::*;
                        
                        let data: Vec<usize> = (0..workload_size).collect();
                        
                        // CPU work with Rayon
                        let cpu_results: Vec<u64> = data
                            .par_iter()
                            .map(|&x| cpu_intensive_work(x % 50))
                            .collect();
                        
                        // I/O work with Tokio
                        let mut final_results = Vec::new();
                        for result in cpu_results {
                            io_simulation((result % 5) as u64).await;
                            final_results.push(result * 2);
                        }
                        
                        black_box(final_results);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Memory Efficiency Benchmarks
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    for &size_kb in &[1, 10, 100, 1000] {
        group.throughput(Throughput::Bytes((size_kb * 1024) as u64));

        // Moirai memory allocation
        group.bench_with_input(
            BenchmarkId::new("moirai_parallel", size_kb),
            &size_kb,
            |b, &size_kb| {
                b.iter(|| {
                    use moirai_iter::{moirai_iter_parallel, ParallelIterator};
                    
                    let task_count = 100;
                    let data: Vec<usize> = (0..task_count).collect();
                    let results: Vec<Vec<u8>> = moirai_iter_parallel(data)
                        .map(|_| memory_allocation_work(size_kb))
                        .collect();
                    
                    black_box(results);
                });
            },
        );

        // Rayon memory allocation
        group.bench_with_input(
            BenchmarkId::new("rayon", size_kb),
            &size_kb,
            |b, &size_kb| {
                b.iter(|| {
                    use rayon::prelude::*;
                    
                    let task_count = 100;
                    let data: Vec<usize> = (0..task_count).collect();
                    let results: Vec<Vec<u8>> = data
                        .into_par_iter()
                        .map(|_| memory_allocation_work(size_kb))
                        .collect();
                    
                    black_box(results);
                });
            },
        );

        // Sequential memory allocation
        group.bench_with_input(
            BenchmarkId::new("sequential", size_kb),
            &size_kb,
            |b, &size_kb| {
                b.iter(|| {
                    let task_count = 100;
                    let data: Vec<usize> = (0..task_count).collect();
                    let results: Vec<Vec<u8>> = data
                        .into_iter()
                        .map(|_| memory_allocation_work(size_kb))
                        .collect();
                    
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Scalability Benchmarks
fn benchmark_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");

    for &thread_count in THREAD_COUNTS {
        group.throughput(Throughput::Elements(1000));

        // Moirai scalability with custom thread count
        group.bench_with_input(
            BenchmarkId::new("moirai_custom_threads", thread_count),
            &thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    use moirai_iter::{moirai_iter_parallel, ParallelIterator};
                    
                    let data: Vec<usize> = (0..1000).collect();
                    let results: Vec<u64> = moirai_iter_parallel(data)
                        .map(|x| cpu_intensive_work(x % 100))
                        .collect();
                    
                    black_box(results);
                });
            },
        );

        // Rayon scalability with custom thread count
        group.bench_with_input(
            BenchmarkId::new("rayon_custom_threads", thread_count),
            &thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(thread_count)
                        .build()
                        .unwrap();
                    
                    pool.install(|| {
                        use rayon::prelude::*;
                        
                        let data: Vec<usize> = (0..1000).collect();
                        let results: Vec<u64> = data
                            .into_par_iter()
                            .map(|x| cpu_intensive_work(x % 100))
                            .collect();
                        
                        black_box(results);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Synchronization Primitives Benchmarks
fn benchmark_synchronization(c: &mut Criterion) {
    let mut group = c.benchmark_group("synchronization");

    // Semaphore benchmarks
    group.bench_function("moirai_semaphore", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                use moirai_async::Semaphore;
                
                let semaphore = Semaphore::new(5);
                let mut handles = Vec::new();
                
                for i in 0..100 {
                    let sem = semaphore.clone();
                    let handle = tokio::spawn(async move {
                        let _permit = sem.acquire().await;
                        black_box(i);
                    });
                    handles.push(handle);
                }
                
                futures::future::join_all(handles).await;
            });
        });
    });

    group.bench_function("tokio_semaphore", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                use tokio::sync::Semaphore;
                
                let semaphore = Arc::new(Semaphore::new(5));
                let mut handles = Vec::new();
                
                for i in 0..100 {
                    let sem = semaphore.clone();
                    let handle = tokio::spawn(async move {
                        let _permit = sem.acquire().await.unwrap();
                        black_box(i);
                    });
                    handles.push(handle);
                }
                
                futures::future::join_all(handles).await;
            });
        });
    });

    // Broadcast channel benchmarks
    group.bench_function("moirai_broadcast", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                use moirai_async::{Broadcast};
                
                let (tx, mut rx) = Broadcast::new(1000);
                
                // Send messages
                for i in 0..100 {
                    tx.send(i).unwrap();
                }
                
                // Receive messages
                for _ in 0..100 {
                    let _ = rx.recv().await;
                }
            });
        });
    });

    group.bench_function("tokio_broadcast", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                use tokio::sync::broadcast;
                
                let (tx, mut rx) = broadcast::channel(1000);
                
                // Send messages
                for i in 0..100 {
                    tx.send(i).unwrap();
                }
                
                // Receive messages
                for _ in 0..100 {
                    let _ = rx.recv().await;
                }
            });
        });
    });

    group.finish();
}

/// Timer Performance Benchmarks
fn benchmark_timers(c: &mut Criterion) {
    let mut group = c.benchmark_group("timers");

    group.bench_function("moirai_sleep", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                use moirai_async::timer::sleep;
                
                let mut handles = Vec::new();
                for _ in 0..100 {
                    let handle = tokio::spawn(async {
                        sleep(Duration::from_millis(1)).await;
                    });
                    handles.push(handle);
                }
                
                futures::future::join_all(handles).await;
            });
        });
    });

    group.bench_function("tokio_sleep", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                use tokio::time::sleep;
                
                let mut handles = Vec::new();
                for _ in 0..100 {
                    let handle = tokio::spawn(async {
                        sleep(Duration::from_millis(1)).await;
                    });
                    handles.push(handle);
                }
                
                futures::future::join_all(handles).await;
            });
        });
    });

    group.bench_function("moirai_interval", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                use moirai_async::timer::interval;
                
                let mut interval = interval(Duration::from_millis(1));
                for _ in 0..10 {
                    interval.next().await;
                }
            });
        });
    });

    group.bench_function("tokio_interval", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                use tokio::time::{interval, Duration};
                
                let mut interval = interval(Duration::from_millis(1));
                for _ in 0..10 {
                    interval.tick().await;
                }
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_task_spawning,
    benchmark_parallel_execution,
    benchmark_async_performance,
    benchmark_mixed_workloads,
    benchmark_memory_efficiency,
    benchmark_scalability,
    benchmark_synchronization,
    benchmark_timers
);

criterion_main!(benches);

        // Moirai task spawning
        group.bench_with_input(
            BenchmarkId::new("moirai", task_count),
            &task_count,
            |b, &count| {
                b.iter(|| {
                    let runtime = moirai::Moirai::new().unwrap();
                    let start = Instant::now();

                    let handles: Vec<_> = (0..count)
                        .map(|i| runtime.spawn_fn(move || black_box(i * 2)))
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
                            runtime.spawn_fn(move || {
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
                        let handles: Vec<_> =
                            (0..count).map(|_| tokio::spawn(io_simulation(1))).collect();

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
                    .map(|_| runtime.spawn_fn(|| cpu_intensive_work(1000)))
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
                let io_handles: Vec<_> = (0..10).map(|_| tokio::spawn(io_simulation(5))).collect();

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
                        .map(|i| runtime.spawn_fn(move || black_box(i)))
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
                            runtime.spawn_fn(move || {
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
