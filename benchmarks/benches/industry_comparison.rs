//! Current industry comparison benchmarks.
//!
//! This target compares Moirai's scoped unified scheduler against Tokio and
//! Rayon on workloads where the APIs perform the same value-preserving work.
//! Tokio and Rayon remain benchmark-only dependencies.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use moirai::Moirai;
use rayon::prelude::*;
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

const TASK_COUNTS: &[usize] = &[100, 1_000, 10_000];
const MAP_REDUCE_COUNTS: &[usize] = &[4_096, 32_768, 65_536];
const WORKER_THREADS: usize = 4;
const CPU_WORK: usize = 8;
const BENCHMARK_SAMPLE_SIZE: usize = 10;
const BENCHMARK_MEASUREMENT_SECONDS: u64 = 1;
const BENCHMARK_WARM_UP_MILLIS: u64 = 250;

fn expected_ready_sum(count: usize) -> usize {
    count * (count + 1) / 2
}

fn verify_ready_sum(sum: usize, count: usize) -> usize {
    assert_eq!(
        sum,
        expected_ready_sum(count),
        "ready-task benchmark must preserve computed values"
    );
    black_box(sum)
}

fn cpu_work(seed: usize) -> u64 {
    let mut value = black_box(seed as u64);
    for index in 0..CPU_WORK {
        value = value.wrapping_add(black_box(index as u64).wrapping_mul(31));
    }
    black_box(value)
}

fn expected_cpu_work_sum(work_items: usize) -> u64 {
    let work_items = work_items as u64;
    let per_task_offset = 31u64 * (CPU_WORK as u64) * ((CPU_WORK - 1) as u64) / 2;
    work_items * per_task_offset + work_items * (work_items - 1) / 2
}

fn verify_cpu_work_sum(sum: u64, work_items: usize) -> u64 {
    assert_eq!(
        sum,
        expected_cpu_work_sum(work_items),
        "CPU benchmark must preserve computed values"
    );
    black_box(sum)
}

fn moirai_indexed_map_reduce(moirai: &Moirai, work_items: usize) -> u64 {
    moirai
        .map_reduce_indexed(work_items, 0u64, cpu_work, u64::wrapping_add)
        .expect("Moirai indexed reduction completed")
}

fn rayon_par_iter_map_reduce(rayon: &rayon::ThreadPool, work_items: usize) -> u64 {
    rayon.install(|| (0..work_items).into_par_iter().map(cpu_work).sum())
}

fn benchmark_ready_task_spawning(c: &mut Criterion) {
    let mut group = c.benchmark_group("industry_ready_task_spawning");
    group.sample_size(BENCHMARK_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(BENCHMARK_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(BENCHMARK_WARM_UP_MILLIS));

    let moirai = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("Moirai runtime must start");

    let tokio = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(WORKER_THREADS)
        .enable_all()
        .build()
        .expect("Tokio runtime must start");

    let rayon = rayon::ThreadPoolBuilder::new()
        .num_threads(WORKER_THREADS)
        .build()
        .expect("Rayon pool must start");

    for &task_count in TASK_COUNTS {
        group.throughput(Throughput::Elements(task_count as u64));

        group.bench_with_input(
            BenchmarkId::new("moirai_scope", task_count),
            &task_count,
            |bench, &count| {
                bench.iter(|| {
                    let sum = AtomicU64::new(0);
                    moirai
                        .scope(|scope| {
                            for value in 0..count {
                                let sum = &sum;
                                scope.spawn(move |_| {
                                    sum.fetch_add(
                                        black_box(value.wrapping_add(1)) as u64,
                                        Ordering::Relaxed,
                                    );
                                })?;
                            }
                            Ok(())
                        })
                        .expect("Moirai scope completed");

                    verify_ready_sum(sum.load(Ordering::Relaxed) as usize, count)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tokio_spawn", task_count),
            &task_count,
            |bench, &count| {
                bench.iter(|| {
                    tokio.block_on(async {
                        let handles = (0..count)
                            .map(|value| {
                                tokio::spawn(async move { black_box(value.wrapping_add(1)) })
                            })
                            .collect::<Vec<_>>();

                        let mut sum = 0usize;
                        for handle in handles {
                            sum = sum.wrapping_add(handle.await.expect("Tokio task completed"));
                        }

                        verify_ready_sum(sum, count)
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rayon_scope", task_count),
            &task_count,
            |bench, &count| {
                bench.iter(|| {
                    let sum = AtomicU64::new(0);
                    rayon.scope(|scope| {
                        for value in 0..count {
                            let sum = &sum;
                            scope.spawn(move |_| {
                                sum.fetch_add(
                                    black_box(value.wrapping_add(1)) as u64,
                                    Ordering::Relaxed,
                                );
                            });
                        }
                    });

                    verify_ready_sum(sum.load(Ordering::Relaxed) as usize, count)
                });
            },
        );
    }

    group.finish();
    moirai.shutdown();
}

fn benchmark_official_rayon_map_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("official_rayon_map_reduce");
    group.sample_size(BENCHMARK_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(BENCHMARK_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(BENCHMARK_WARM_UP_MILLIS));

    let moirai = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("Moirai runtime must start");

    let rayon = rayon::ThreadPoolBuilder::new()
        .num_threads(WORKER_THREADS)
        .build()
        .expect("Rayon pool must start");

    for &work_items in MAP_REDUCE_COUNTS {
        group.throughput(Throughput::Elements(work_items as u64));

        group.bench_with_input(
            BenchmarkId::new("moirai_indexed_reduce", work_items),
            &work_items,
            |bench, &count| {
                bench
                    .iter(|| verify_cpu_work_sum(moirai_indexed_map_reduce(&moirai, count), count));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rayon_into_par_iter", work_items),
            &work_items,
            |bench, &count| {
                bench.iter(|| verify_cpu_work_sum(rayon_par_iter_map_reduce(&rayon, count), count));
            },
        );
    }

    group.finish();
    moirai.shutdown();
}

criterion_group!(
    benches,
    benchmark_ready_task_spawning,
    benchmark_official_rayon_map_reduce
);
criterion_main!(benches);
