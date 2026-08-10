//! Quick scheduler comparison for ready tasks.
//!
//! This benchmark isolates spawn, dispatch, execution, and join overhead for
//! small ready tasks across Moirai's unified scheduler, Tokio, and Rayon.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use crossbeam::channel::{bounded, TrySendError};
use moirai::Moirai;
use moirai_core::{error::ExecutorError, Priority};
use moirai_executor::{BlockingTask, ThreadScheduler};
use moirai_scheduler::{ChaseLevDeque, SharedEpochReclaim, SplitDeque};
use rayon::prelude::*;
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

const SATURATED_ADMISSION_CAPACITY: usize = 256;
const SATURATED_ADMISSION_WORKERS: usize = 1;

const TASKS_PER_ITERATION: usize = 256;
const SCALING_TASK_COUNTS: &[usize] = &[64, 256, 1_024];
const MIXED_TASKS_PER_CLASS: usize = 64;
const REAL_APP_RECORDS: usize = 64;
const REAL_APP_CHANNEL_RECORDS: usize = 8;
const REAL_APP_ANALYTICS_RECORDS: usize = 1_048_576;
const REAL_APP_CHANNEL_CAPACITY: usize = 128;
const DEQUE_RECLAIM_ITEMS: usize = 256;
const WORKER_THREADS: usize = 4;

fn expected_ready_sum(count: usize) -> usize {
    count.wrapping_mul(count.wrapping_add(1)) / 2
}

fn verify_ready_sum(sum: usize, count: usize) -> usize {
    assert_eq!(sum, expected_ready_sum(count));
    black_box(sum)
}

fn expected_mixed_sum(count: usize) -> usize {
    expected_ready_sum(count).wrapping_mul(3)
}

fn verify_mixed_sum(sum: usize, count: usize) -> usize {
    assert_eq!(sum, expected_mixed_sum(count));
    black_box(sum)
}

fn expected_real_app_sum(count: usize) -> usize {
    expected_ready_sum(count)
        .wrapping_mul(8)
        .wrapping_add(expected_ready_sum(REAL_APP_CHANNEL_RECORDS).wrapping_mul(5))
        .wrapping_add(expected_ready_sum(REAL_APP_ANALYTICS_RECORDS).wrapping_mul(3))
}

fn verify_real_app_sum(sum: usize, count: usize) -> usize {
    assert_eq!(sum, expected_real_app_sum(count));
    black_box(sum)
}

fn moirai_deque_deferred_reclaim_sum(count: usize) -> usize {
    let mut deque: ChaseLevDeque<usize> = ChaseLevDeque::new(2);
    for value in 0..count {
        deque.push(black_box(value.wrapping_add(1)));
    }

    let mut sum = 0usize;
    while let Some(value) = deque.pop() {
        sum = sum.wrapping_add(value);
    }
    sum
}

fn moirai_deque_shared_epoch_reclaim_sum(count: usize) -> usize {
    let mut deque: ChaseLevDeque<usize, SharedEpochReclaim> = ChaseLevDeque::new(2);
    for value in 0..count {
        deque.push(black_box(value.wrapping_add(1)));
    }

    assert!(
        deque.try_reclaim_shared(SharedEpochReclaim),
        "shared epoch reclaim must succeed without active guards"
    );

    let mut sum = 0usize;
    while let Some(value) = deque.pop() {
        sum = sum.wrapping_add(value);
    }
    sum
}

fn moirai_split_deque_sum(count: usize) -> usize {
    let deque: SplitDeque<usize> = SplitDeque::new();
    for value in 0..count {
        deque.push(black_box(value.wrapping_add(1)));
    }

    let mut sum = 0usize;
    while let Some(value) = deque.pop() {
        sum = sum.wrapping_add(value);
    }
    sum
}

struct MoiraiAdmissionFixture {
    scheduler: ThreadScheduler<SATURATED_ADMISSION_CAPACITY>,
    release: Option<crossbeam::channel::Sender<()>>,
}

impl Drop for MoiraiAdmissionFixture {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            let _ = release.try_send(());
        }
    }
}

fn prepare_moirai_saturated_admission() -> MoiraiAdmissionFixture {
    let scheduler = ThreadScheduler::<SATURATED_ADMISSION_CAPACITY>::new(
        SATURATED_ADMISSION_WORKERS,
        "benchmark-saturated-admission",
    )
    .expect("Moirai scheduler must start");
    let (started_tx, started_rx) = bounded(0);
    let (release_tx, release_rx) = bounded(1);
    scheduler
        .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
            started_tx.send(()).expect("blocker start must be observed");
            let _ = release_rx.recv();
        })
        .expect("blocker admission must succeed");
    started_rx.recv().expect("blocker must enter the lane");

    for _ in 0..SATURATED_ADMISSION_CAPACITY {
        scheduler
            .schedule::<BlockingTask, _>(Priority::Normal, None, |_| {})
            .expect("queue fill must succeed");
    }

    MoiraiAdmissionFixture {
        scheduler,
        release: Some(release_tx),
    }
}

fn prepare_crossbeam_saturated_admission() -> (
    crossbeam::channel::Sender<usize>,
    crossbeam::channel::Receiver<usize>,
) {
    let (sender, receiver) = bounded(SATURATED_ADMISSION_CAPACITY);
    for value in 0..SATURATED_ADMISSION_CAPACITY {
        sender.send(value).expect("bounded queue fill must succeed");
    }

    (sender, receiver)
}

fn benchmark_moirai_rejection(fixture: &MoiraiAdmissionFixture, iterations: u64) -> Duration {
    let started = Instant::now();
    for _ in 0..iterations {
        let rejection = fixture
            .scheduler
            .schedule::<BlockingTask, _>(Priority::Normal, None, |_| {})
            .expect_err("capacity-plus-one admission must be rejected");
        assert!(matches!(rejection, ExecutorError::ResourceExhausted(_)));
    }
    assert_eq!(
        fixture.scheduler.pending_tasks(),
        SATURATED_ADMISSION_CAPACITY
    );
    started.elapsed()
}

fn benchmark_crossbeam_rejection(
    sender: &crossbeam::channel::Sender<usize>,
    receiver: &crossbeam::channel::Receiver<usize>,
    iterations: u64,
) -> Duration {
    let started = Instant::now();
    for _ in 0..iterations {
        let rejection = sender
            .try_send(SATURATED_ADMISSION_CAPACITY)
            .expect_err("capacity-plus-one send must be rejected");
        assert!(matches!(rejection, TrySendError::Full(_)));
    }
    assert_eq!(receiver.len(), SATURATED_ADMISSION_CAPACITY);
    started.elapsed()
}

fn moirai_scope_sum(moirai: &Moirai, count: usize) -> usize {
    let sum = AtomicUsize::new(0);
    moirai
        .scope(|scope| {
            for value in 0..count {
                let sum = &sum;
                scope.spawn(move |_| {
                    sum.fetch_add(black_box(value.wrapping_add(1)), Ordering::Relaxed);
                })?;
            }
            Ok(())
        })
        .expect("Moirai scope must complete");

    sum.load(Ordering::Relaxed)
}

fn moirai_indexed_reduce_sum(moirai: &Moirai, count: usize) -> usize {
    moirai
        .map_reduce_indexed(
            count,
            0usize,
            |value| black_box(value.wrapping_add(1)),
            usize::wrapping_add,
        )
        .expect("Moirai indexed reduction must complete")
}

fn tokio_spawn_ready_sum(tokio: &tokio::runtime::Runtime, count: usize) -> usize {
    tokio.block_on(async {
        let handles = (0..count)
            .map(|value| tokio::spawn(async move { black_box(value.wrapping_add(1)) }))
            .collect::<Vec<_>>();

        let mut sum = 0usize;
        for handle in handles {
            sum = sum.wrapping_add(handle.await.expect("tokio task ok"));
        }

        sum
    })
}

fn rayon_scope_sum(rayon: &rayon::ThreadPool, count: usize) -> usize {
    let sum = AtomicUsize::new(0);
    rayon.scope(|scope| {
        for value in 0..count {
            let sum = &sum;
            scope.spawn(move |_| {
                sum.fetch_add(black_box(value.wrapping_add(1)), Ordering::Relaxed);
            });
        }
    });

    sum.load(Ordering::Relaxed)
}

fn rayon_indexed_sum(rayon: &rayon::ThreadPool, count: usize) -> usize {
    rayon.install(|| {
        (0..count)
            .into_par_iter()
            .map(|value| black_box(value.wrapping_add(1)))
            .sum()
    })
}

fn moirai_mixed_unified_sum(moirai: &Moirai, count: usize) -> usize {
    let async_handles = (0..count)
        .map(|value| moirai.spawn_async(async move { black_box(value.wrapping_add(1)) }))
        .collect::<Vec<_>>();

    let mut sum =
        moirai_scope_sum(moirai, count).wrapping_add(moirai_indexed_reduce_sum(moirai, count));
    for handle in async_handles {
        sum = sum.wrapping_add(
            handle
                .join()
                .expect("Moirai async mixed handle must be attached")
                .expect("Moirai async mixed task must complete"),
        );
    }

    sum
}

fn tokio_rayon_mixed_sum(
    tokio: &tokio::runtime::Runtime,
    rayon: &rayon::ThreadPool,
    count: usize,
) -> usize {
    tokio.block_on(async {
        let async_handles = (0..count)
            .map(|value| tokio::spawn(async move { black_box(value.wrapping_add(1)) }))
            .collect::<Vec<_>>();

        let mut sum = rayon_scope_sum(rayon, count).wrapping_add(rayon_indexed_sum(rayon, count));
        for handle in async_handles {
            sum = sum.wrapping_add(handle.await.expect("Tokio async mixed task must complete"));
        }

        sum
    })
}

fn moirai_channel_transfer_sum(count: usize, capacity: usize) -> usize {
    let (tx, rx) = spsc::<usize>(capacity);
    for value in 0..count {
        tx.send(black_box(value.wrapping_add(1).wrapping_mul(5)))
            .expect("Moirai channel must accept record");
    }

    let mut sum = 0usize;
    for _ in 0..count {
        let value = rx.recv().expect("Moirai channel must receive record");
        sum = sum.wrapping_add(value);
    }
    sum
}

fn moirai_real_app_pipeline_sum(moirai: &Moirai, count: usize) -> usize {
    let async_handles = (0..count)
        .map(|value| moirai.spawn_async(async move { black_box(value.wrapping_add(1)) }))
        .collect::<Vec<_>>();

    let scope_sum = AtomicUsize::new(0);
    moirai
        .scope(|scope| {
            for value in 0..count {
                let scope_sum = &scope_sum;
                scope.spawn(move |_| {
                    scope_sum.fetch_add(
                        black_box(value.wrapping_add(1).wrapping_mul(7)),
                        Ordering::Relaxed,
                    );
                })?;
            }
            Ok(())
        })
        .expect("Moirai real-app scoped work must complete");

    let mut sum = scope_sum.load(Ordering::Relaxed);
    sum = sum.wrapping_add(
        moirai
            .map_reduce_indexed(
                REAL_APP_ANALYTICS_RECORDS,
                0usize,
                |value| black_box(value.wrapping_add(1).wrapping_mul(3)),
                usize::wrapping_add,
            )
            .expect("Moirai real-app indexed reduction must complete"),
    );
    sum = sum.wrapping_add(moirai_channel_transfer_sum(
        REAL_APP_CHANNEL_RECORDS,
        REAL_APP_CHANNEL_CAPACITY,
    ));

    for handle in async_handles {
        sum = sum.wrapping_add(
            handle
                .join()
                .expect("Moirai real-app async handle must be attached")
                .expect("Moirai real-app async task must complete"),
        );
    }

    sum
}

async fn tokio_channel_transfer_sum(count: usize, capacity: usize) -> usize {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<usize>(capacity);
    for value in 0..count {
        tx.send(black_box(value.wrapping_add(1).wrapping_mul(5)))
            .await
            .expect("Tokio channel must accept record");
    }

    drop(tx);

    let mut sum = 0usize;
    while let Some(value) = rx.recv().await {
        sum = sum.wrapping_add(value);
    }
    sum
}

fn tokio_rayon_real_app_pipeline_sum(
    tokio: &tokio::runtime::Runtime,
    rayon: &rayon::ThreadPool,
    count: usize,
) -> usize {
    tokio.block_on(async {
        let async_handles = (0..count)
            .map(|value| tokio::spawn(async move { black_box(value.wrapping_add(1)) }))
            .collect::<Vec<_>>();

        let scope_sum = AtomicUsize::new(0);
        rayon.scope(|scope| {
            for value in 0..count {
                let scope_sum = &scope_sum;
                scope.spawn(move |_| {
                    scope_sum.fetch_add(
                        black_box(value.wrapping_add(1).wrapping_mul(7)),
                        Ordering::Relaxed,
                    );
                });
            }
        });

        let mut sum = scope_sum.load(Ordering::Relaxed);
        sum = sum.wrapping_add(rayon.install(|| {
            (0..REAL_APP_ANALYTICS_RECORDS)
                .into_par_iter()
                .map(|value| black_box(value.wrapping_add(1).wrapping_mul(3)))
                .sum::<usize>()
        }));
        sum = sum.wrapping_add(
            tokio_channel_transfer_sum(REAL_APP_CHANNEL_RECORDS, REAL_APP_CHANNEL_CAPACITY).await,
        );

        for handle in async_handles {
            sum = sum.wrapping_add(
                handle
                    .await
                    .expect("Tokio real-app async task must complete"),
            );
        }

        sum
    })
}

fn bench_saturated_admission(c: &mut Criterion) {
    let mut group = c.benchmark_group("saturated_admission");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(250));
    group.throughput(Throughput::Elements(1));

    let mut moirai = prepare_moirai_saturated_admission();
    group.bench_function("moirai_bounded_rejection", |b| {
        b.iter_custom(|iterations| benchmark_moirai_rejection(&moirai, iterations));
    });
    moirai
        .release
        .take()
        .expect("blocker release must be present")
        .send(())
        .expect("blocker release must succeed");
    moirai
        .scheduler
        .join()
        .expect("saturated scheduler must recover");
    assert_eq!(moirai.scheduler.pending_tasks(), 0);
    moirai.scheduler.shutdown();

    let (sender, receiver) = prepare_crossbeam_saturated_admission();
    group.bench_function("crossbeam_bounded_try_send", |b| {
        b.iter_custom(|iterations| benchmark_crossbeam_rejection(&sender, &receiver, iterations));
    });
    for _ in 0..SATURATED_ADMISSION_CAPACITY {
        receiver.recv().expect("bounded queue drain must succeed");
    }
    group.finish();
}

fn bench_ready_task_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("ready_task_schedule");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(250));

    let moirai = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("Moirai runtime must start");

    group.bench_function("moirai_scope", |b| {
        b.iter(|| {
            verify_ready_sum(
                moirai_scope_sum(&moirai, TASKS_PER_ITERATION),
                TASKS_PER_ITERATION,
            )
        });
    });

    let tokio = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(WORKER_THREADS)
        .enable_all()
        .build()
        .expect("Tokio runtime must start");

    group.bench_function("tokio_spawn_ready", |b| {
        b.iter(|| {
            verify_ready_sum(
                tokio_spawn_ready_sum(&tokio, TASKS_PER_ITERATION),
                TASKS_PER_ITERATION,
            )
        });
    });

    let rayon = rayon::ThreadPoolBuilder::new()
        .num_threads(WORKER_THREADS)
        .build()
        .expect("Rayon pool must start");

    group.bench_function("rayon_scope", |b| {
        b.iter(|| {
            verify_ready_sum(
                rayon_scope_sum(&rayon, TASKS_PER_ITERATION),
                TASKS_PER_ITERATION,
            )
        });
    });

    group.finish();
    moirai.shutdown();
}

fn bench_indexed_reduce_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexed_reduce_schedule");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(250));

    let moirai = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("Moirai runtime must start");

    group.bench_function("moirai_indexed_reduce", |b| {
        b.iter(|| {
            verify_ready_sum(
                moirai_indexed_reduce_sum(&moirai, TASKS_PER_ITERATION),
                TASKS_PER_ITERATION,
            )
        });
    });

    let rayon = rayon::ThreadPoolBuilder::new()
        .num_threads(WORKER_THREADS)
        .build()
        .expect("Rayon pool must start");

    group.bench_function("rayon_indexed", |b| {
        b.iter(|| {
            verify_ready_sum(
                rayon_indexed_sum(&rayon, TASKS_PER_ITERATION),
                TASKS_PER_ITERATION,
            )
        });
    });

    group.finish();
    moirai.shutdown();
}

fn bench_scoped_ready_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scoped_ready_scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(250));

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

    for &count in SCALING_TASK_COUNTS {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("moirai_scope", count),
            &count,
            |b, &count| {
                b.iter(|| verify_ready_sum(moirai_scope_sum(&moirai, count), count));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rayon_scope", count),
            &count,
            |b, &count| {
                b.iter(|| verify_ready_sum(rayon_scope_sum(&rayon, count), count));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tokio_spawn_ready", count),
            &count,
            |b, &count| {
                b.iter(|| verify_ready_sum(tokio_spawn_ready_sum(&tokio, count), count));
            },
        );
    }

    group.finish();
    moirai.shutdown();
}

fn bench_indexed_reduce_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexed_reduce_scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(250));

    let moirai = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("Moirai runtime must start");

    let rayon = rayon::ThreadPoolBuilder::new()
        .num_threads(WORKER_THREADS)
        .build()
        .expect("Rayon pool must start");

    for &count in SCALING_TASK_COUNTS {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("moirai_indexed_reduce", count),
            &count,
            |b, &count| {
                b.iter(|| verify_ready_sum(moirai_indexed_reduce_sum(&moirai, count), count));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rayon_indexed", count),
            &count,
            |b, &count| {
                b.iter(|| verify_ready_sum(rayon_indexed_sum(&rayon, count), count));
            },
        );
    }

    group.finish();
    moirai.shutdown();
}

fn bench_mixed_unified_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_unified_schedule");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(250));
    group.throughput(Throughput::Elements((MIXED_TASKS_PER_CLASS * 3) as u64));

    let moirai = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("Moirai runtime must start");

    group.bench_function("moirai_unified_mixed", |b| {
        b.iter(|| {
            verify_mixed_sum(
                moirai_mixed_unified_sum(&moirai, MIXED_TASKS_PER_CLASS),
                MIXED_TASKS_PER_CLASS,
            )
        });
    });

    let tokio = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(WORKER_THREADS)
        .enable_all()
        .build()
        .expect("Tokio runtime must start");

    let rayon = rayon::ThreadPoolBuilder::new()
        .num_threads(WORKER_THREADS)
        .build()
        .expect("Rayon pool must start");

    group.bench_function("tokio_rayon_mixed", |b| {
        b.iter(|| {
            verify_mixed_sum(
                tokio_rayon_mixed_sum(&tokio, &rayon, MIXED_TASKS_PER_CLASS),
                MIXED_TASKS_PER_CLASS,
            )
        });
    });

    group.finish();
    moirai.shutdown();
}

fn bench_real_application_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_application_mixed_workload");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(250));
    group.throughput(Throughput::Elements(
        (REAL_APP_RECORDS * 2 + REAL_APP_CHANNEL_RECORDS + REAL_APP_ANALYTICS_RECORDS) as u64,
    ));

    let moirai = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("Moirai runtime must start");

    group.bench_function("moirai_real_app_pipeline", |b| {
        b.iter(|| {
            verify_real_app_sum(
                moirai_real_app_pipeline_sum(&moirai, REAL_APP_RECORDS),
                REAL_APP_RECORDS,
            )
        });
    });

    let tokio = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(WORKER_THREADS)
        .enable_all()
        .build()
        .expect("Tokio runtime must start");

    let rayon = rayon::ThreadPoolBuilder::new()
        .num_threads(WORKER_THREADS)
        .build()
        .expect("Rayon pool must start");

    group.bench_function("tokio_rayon_real_app_pipeline", |b| {
        b.iter(|| {
            verify_real_app_sum(
                tokio_rayon_real_app_pipeline_sum(&tokio, &rayon, REAL_APP_RECORDS),
                REAL_APP_RECORDS,
            )
        });
    });

    group.finish();
    moirai.shutdown();
}

fn bench_standalone_deque_reclaim_policy(c: &mut Criterion) {
    let mut group = c.benchmark_group("standalone_deque_reclaim_policy");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(250));
    group.throughput(Throughput::Elements(DEQUE_RECLAIM_ITEMS as u64));

    group.bench_function("moirai_deferred_reclaim", |b| {
        b.iter(|| {
            verify_ready_sum(
                moirai_deque_deferred_reclaim_sum(DEQUE_RECLAIM_ITEMS),
                DEQUE_RECLAIM_ITEMS,
            )
        });
    });

    group.bench_function("moirai_shared_epoch_reclaim", |b| {
        b.iter(|| {
            verify_ready_sum(
                moirai_deque_shared_epoch_reclaim_sum(DEQUE_RECLAIM_ITEMS),
                DEQUE_RECLAIM_ITEMS,
            )
        });
    });

    group.bench_function("moirai_split_deque", |b| {
        b.iter(|| {
            verify_ready_sum(
                moirai_split_deque_sum(DEQUE_RECLAIM_ITEMS),
                DEQUE_RECLAIM_ITEMS,
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_saturated_admission,
    bench_ready_task_schedule,
    bench_indexed_reduce_schedule,
    bench_scoped_ready_scaling,
    bench_indexed_reduce_scaling,
    bench_mixed_unified_schedule,
    bench_real_application_mixed_workload,
    bench_standalone_deque_reclaim_policy
);
criterion_main!(benches);
