//! Public result-handle comparison for ready tasks.
//!
//! This benchmark isolates APIs that return a per-task result handle. Moirai
//! uses `Moirai::spawn_fn` plus `TaskHandle::join`; Tokio uses
//! `tokio::spawn` plus `JoinHandle::await`. Moirai async-ready rows compare
//! against the same ready Tokio `JoinHandle` baseline because Tokio's task API
//! is async-native and the equivalent ready future is identical. Rayon does not
//! expose a directly equivalent result handle, so its row is labeled as a scoped
//! completion baseline rather than a result-handle equivalent.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use moirai::Moirai;
use std::{
    future::Future,
    pin::Pin,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll},
    time::Duration,
};

const BENCHMARK_SAMPLE_SIZE: usize = 20;
const BENCHMARK_MEASUREMENT_SECONDS: u64 = 2;
const BENCHMARK_WARM_UP_MILLIS: u64 = 500;
const WORKER_THREADS: usize = 4;
const READY_VALUE: usize = 42;
const CAPTURE_WORDS: usize = 10;
const CAPTURED_READY_VALUE: usize = CAPTURE_WORDS;
const OVERSIZED_CAPTURE_WORDS: usize = 32;
const OVERSIZED_CAPTURED_READY_VALUE: usize = OVERSIZED_CAPTURE_WORDS;

#[derive(Default)]
struct WakeOnce {
    observed_pending: bool,
}

impl Future for WakeOnce {
    type Output = usize;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.observed_pending {
            Poll::Ready(black_box(READY_VALUE))
        } else {
            self.observed_pending = true;
            context.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

fn verify_ready_value(value: usize) -> usize {
    assert_eq!(value, READY_VALUE);
    black_box(value)
}

fn verify_captured_ready_value(value: usize) -> usize {
    assert_eq!(value, CAPTURED_READY_VALUE);
    black_box(value)
}

fn verify_oversized_captured_ready_value(value: usize) -> usize {
    assert_eq!(value, OVERSIZED_CAPTURED_READY_VALUE);
    black_box(value)
}

fn moirai_spawn_join_ready(moirai: &Moirai) -> usize {
    let handle = moirai.spawn_fn(|| black_box(READY_VALUE));
    let result = handle
        .join()
        .expect("Moirai task handle must be attached")
        .expect("Moirai ready task must not fail");

    verify_ready_value(result)
}

fn moirai_spawn_join_captured_ready(moirai: &Moirai) -> usize {
    let words = [1usize; CAPTURE_WORDS];
    let handle = moirai.spawn_fn(move || black_box(words.iter().copied().sum::<usize>()));
    let result = handle
        .join()
        .expect("Moirai captured task handle must be attached")
        .expect("Moirai captured ready task must not fail");

    verify_captured_ready_value(result)
}

fn tokio_spawn_join_ready(tokio: &tokio::runtime::Runtime) -> usize {
    tokio.block_on(async {
        let handle = tokio::spawn(async { black_box(READY_VALUE) });
        let result = handle.await.expect("Tokio ready task must not fail");

        verify_ready_value(result)
    })
}

fn tokio_spawn_join_captured_ready(tokio: &tokio::runtime::Runtime) -> usize {
    tokio.block_on(async {
        let words = [1usize; CAPTURE_WORDS];
        let handle = tokio::spawn(async move { black_box(words.iter().copied().sum::<usize>()) });
        let result = handle
            .await
            .expect("Tokio captured ready task must not fail");

        verify_captured_ready_value(result)
    })
}

fn moirai_spawn_join_oversized_captured_ready(moirai: &Moirai) -> usize {
    let words = [1usize; OVERSIZED_CAPTURE_WORDS];
    let handle = moirai.spawn_fn(move || black_box(words.iter().copied().sum::<usize>()));
    let result = handle
        .join()
        .expect("Moirai oversized captured task handle must be attached")
        .expect("Moirai oversized captured ready task must not fail");

    verify_oversized_captured_ready_value(result)
}

fn tokio_spawn_join_oversized_captured_ready(tokio: &tokio::runtime::Runtime) -> usize {
    tokio.block_on(async {
        let words = [1usize; OVERSIZED_CAPTURE_WORDS];
        let handle = tokio::spawn(async move { black_box(words.iter().copied().sum::<usize>()) });
        let result = handle
            .await
            .expect("Tokio oversized captured ready task must not fail");

        verify_oversized_captured_ready_value(result)
    })
}

fn moirai_spawn_async_ready(moirai: &Moirai) -> usize {
    let handle = moirai.spawn_async(async { black_box(READY_VALUE) });
    let result = handle
        .join()
        .expect("Moirai async task handle must be attached")
        .expect("Moirai async ready task must not fail");

    verify_ready_value(result)
}

fn moirai_spawn_async_wake_once(moirai: &Moirai) -> usize {
    let handle = moirai.spawn_async(WakeOnce::default());
    let result = handle
        .join()
        .expect("Moirai async task handle must be attached")
        .expect("Moirai wake-once task must not fail");

    verify_ready_value(result)
}

fn tokio_spawn_async_wake_once(tokio: &tokio::runtime::Runtime) -> usize {
    tokio.block_on(async {
        let handle = tokio::spawn(WakeOnce::default());
        let result = handle.await.expect("Tokio wake-once task must not fail");

        verify_ready_value(result)
    })
}

fn moirai_scope_single_ready(moirai: &Moirai) -> usize {
    let result = AtomicUsize::new(0);
    moirai
        .scope(|scope| {
            let result = &result;
            scope.spawn(move |_| {
                result.store(black_box(READY_VALUE), Ordering::Relaxed);
            })?;
            Ok(())
        })
        .expect("Moirai scope must complete");

    verify_ready_value(result.load(Ordering::Relaxed))
}

fn rayon_scope_single_ready(rayon: &rayon::ThreadPool) -> usize {
    let result = AtomicUsize::new(0);
    rayon.scope(|scope| {
        let result = &result;
        scope.spawn(move |_| {
            result.store(black_box(READY_VALUE), Ordering::Relaxed);
        });
    });

    verify_ready_value(result.load(Ordering::Relaxed))
}

fn benchmark_public_result_handles(c: &mut Criterion) {
    let mut group = c.benchmark_group("public_result_handle_ready");
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

    group.bench_function("moirai_spawn_join_ready", |bench| {
        bench.iter(|| moirai_spawn_join_ready(&moirai));
    });

    group.bench_function("tokio_spawn_join_ready", |bench| {
        bench.iter(|| tokio_spawn_join_ready(&tokio));
    });

    group.bench_function("moirai_spawn_join_captured_ready", |bench| {
        bench.iter(|| moirai_spawn_join_captured_ready(&moirai));
    });

    group.bench_function("tokio_spawn_join_captured_ready", |bench| {
        bench.iter(|| tokio_spawn_join_captured_ready(&tokio));
    });

    group.bench_function("moirai_spawn_join_oversized_captured_ready", |bench| {
        bench.iter(|| moirai_spawn_join_oversized_captured_ready(&moirai));
    });

    group.bench_function("tokio_spawn_join_oversized_captured_ready", |bench| {
        bench.iter(|| tokio_spawn_join_oversized_captured_ready(&tokio));
    });

    group.bench_function("moirai_spawn_async_ready", |bench| {
        bench.iter(|| moirai_spawn_async_ready(&moirai));
    });

    group.bench_function("moirai_spawn_async_wake_once", |bench| {
        bench.iter(|| moirai_spawn_async_wake_once(&moirai));
    });

    group.bench_function("tokio_spawn_async_wake_once", |bench| {
        bench.iter(|| tokio_spawn_async_wake_once(&tokio));
    });

    group.bench_function("moirai_scope_single_ready", |bench| {
        bench.iter(|| moirai_scope_single_ready(&moirai));
    });

    group.bench_function("rayon_scope_single_ready", |bench| {
        bench.iter(|| rayon_scope_single_ready(&rayon));
    });

    group.finish();
    moirai.shutdown();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(BENCHMARK_SAMPLE_SIZE)
        .measurement_time(Duration::from_secs(BENCHMARK_MEASUREMENT_SECONDS))
        .warm_up_time(Duration::from_millis(BENCHMARK_WARM_UP_MILLIS))
        .without_plots();
    targets = benchmark_public_result_handles
}

criterion_main!(benches);
