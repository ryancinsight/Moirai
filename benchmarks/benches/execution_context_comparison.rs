//! Owned execution-context iterator comparison against Rayon.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use futures::StreamExt;
use moirai_iter::{AsyncContext, ExecutionContext, MoiraiIterator, ParallelContext};
use rayon::prelude::*;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

const SAMPLE_SIZE: usize = 10;
const WARM_UP_MILLIS: u64 = 100;
const MEASUREMENT_MILLIS: u64 = 300;
const WORK_ITEMS: usize = 512;
const ASYNC_WORK_ITEMS: usize = 1_024;
const SPARSE_WORK_ITEMS: usize = 1_000;

fn source_data() -> Vec<u64> {
    (0..WORK_ITEMS as u64)
        .map(|value| value.wrapping_mul(13).wrapping_add(5))
        .collect()
}

fn moirai_parallel_context_map(context: &ParallelContext, data: Vec<u64>) -> u64 {
    context
        .execute_iter(data, |value| value.wrapping_mul(3).wrapping_add(1))
        .expect("parallel context map must complete")
        .into_iter()
        .sum()
}

fn rayon_owned_map(data: Vec<u64>) -> u64 {
    data.into_par_iter()
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .sum()
}

fn moirai_parallel_context_async_map(data: Vec<u64>) -> u64 {
    futures::executor::block_on(async {
        MoiraiIterator::parallel(data)
            .map_async(|value| async move { value.wrapping_mul(5).wrapping_add(3) })
            .await
            .collect()
            .await
            .into_iter()
            .sum()
    })
}

fn moirai_parallel_context_pending_async_map(data: Vec<u64>) -> u64 {
    futures::executor::block_on(async {
        MoiraiIterator::parallel(data)
            .map_async(|value| {
                let mut first_poll = true;
                futures::future::poll_fn(move |context| {
                    if first_poll {
                        first_poll = false;
                        context.waker().wake_by_ref();
                        Poll::Pending
                    } else {
                        Poll::Ready(value.wrapping_mul(5).wrapping_add(3))
                    }
                })
            })
            .await
            .collect()
            .await
            .into_iter()
            .sum()
    })
}

struct SparseWakeState {
    next_rank: AtomicUsize,
    wakers: Mutex<Vec<Option<Waker>>>,
}

struct SparseFuture {
    state: Arc<SparseWakeState>,
    rank: usize,
    value: u64,
}

impl Future for SparseFuture {
    type Output = u64;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.state.next_rank.load(Ordering::Acquire) != self.rank {
            self.state
                .wakers
                .lock()
                .expect("sparse benchmark waker registry must remain available")[self.rank] =
                Some(context.waker().clone());
            return Poll::Pending;
        }

        let next_rank = self.rank + 1;
        self.state.next_rank.store(next_rank, Ordering::Release);
        let next_waker = self
            .state
            .wakers
            .lock()
            .expect("sparse benchmark waker registry must remain available")
            .get_mut(next_rank)
            .and_then(Option::take);
        if let Some(waker) = next_waker {
            waker.wake();
        }
        Poll::Ready(self.value.wrapping_mul(5).wrapping_add(3))
    }
}

fn sparse_futures(data: Vec<u64>) -> impl Iterator<Item = SparseFuture> {
    let state = Arc::new(SparseWakeState {
        next_rank: AtomicUsize::new(0),
        wakers: Mutex::new(vec![None; SPARSE_WORK_ITEMS]),
    });
    data.into_iter()
        .enumerate()
        .map(move |(index, value)| SparseFuture {
            state: Arc::clone(&state),
            rank: SPARSE_WORK_ITEMS - index - 1,
            value,
        })
}

fn incumbent_sparse_pending_map(data: Vec<u64>) -> u64 {
    futures::executor::block_on(async {
        futures::stream::iter(sparse_futures(data))
            .buffered(SPARSE_WORK_ITEMS)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .sum()
    })
}

fn moirai_sparse_pending_map(data: Vec<u64>) -> u64 {
    let context =
        ExecutionContext::Async(AsyncContext::new().with_max_concurrent(SPARSE_WORK_ITEMS));
    futures::executor::block_on(async {
        MoiraiIterator::new(sparse_futures(data).collect(), context)
            .map_async(core::convert::identity)
            .await
            .collect()
            .await
            .into_iter()
            .sum()
    })
}

fn execution_context_comparison(c: &mut Criterion) {
    let data = source_data();
    let context = ParallelContext::with_chunk_size(WORK_ITEMS);

    assert_eq!(
        moirai_parallel_context_map(&context, data.clone()),
        rayon_owned_map(data.clone())
    );

    let mut group = c.benchmark_group("execution_context_owned_map");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| {
            black_box(moirai_parallel_context_map(
                &context,
                black_box(input.clone()),
            ))
        })
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_owned_map(black_box(input.clone()))))
    });
    group.finish();

    let async_data = (0..ASYNC_WORK_ITEMS as u64)
        .map(|value| value.wrapping_mul(19).wrapping_add(7))
        .collect::<Vec<_>>();
    let expected = async_data
        .iter()
        .copied()
        .map(|value| value.wrapping_mul(5).wrapping_add(3))
        .sum::<u64>();
    assert_eq!(
        moirai_parallel_context_async_map(async_data.clone()),
        expected
    );

    let mut group = c.benchmark_group("execution_context_parallel_async_map");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(
        BenchmarkId::new("moirai", ASYNC_WORK_ITEMS),
        &async_data,
        |b, input| {
            b.iter(|| black_box(moirai_parallel_context_async_map(black_box(input.clone()))))
        },
    );
    group.finish();

    assert_eq!(
        moirai_parallel_context_pending_async_map(async_data.clone()),
        expected
    );
    let mut group = c.benchmark_group("execution_context_parallel_pending_async_map");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(
        BenchmarkId::new("moirai", ASYNC_WORK_ITEMS),
        &async_data,
        |b, input| {
            b.iter(|| {
                black_box(moirai_parallel_context_pending_async_map(black_box(
                    input.clone(),
                )))
            })
        },
    );
    group.finish();

    let sparse_data = (0..SPARSE_WORK_ITEMS as u64)
        .map(|value| value.wrapping_mul(19).wrapping_add(7))
        .collect::<Vec<_>>();
    let sparse_expected = sparse_data
        .iter()
        .copied()
        .map(|value| value.wrapping_mul(5).wrapping_add(3))
        .sum::<u64>();
    assert_eq!(
        moirai_sparse_pending_map(sparse_data.clone()),
        sparse_expected
    );
    assert_eq!(
        incumbent_sparse_pending_map(sparse_data.clone()),
        sparse_expected
    );
    let mut group = c.benchmark_group("execution_context_sparse_pending_map");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(
        BenchmarkId::new("moirai", SPARSE_WORK_ITEMS),
        &sparse_data,
        |b, input| b.iter(|| black_box(moirai_sparse_pending_map(black_box(input.clone())))),
    );
    group.bench_with_input(
        BenchmarkId::new("futures-util", SPARSE_WORK_ITEMS),
        &sparse_data,
        |b, input| b.iter(|| black_box(incumbent_sparse_pending_map(black_box(input.clone())))),
    );
    group.finish();
}

criterion_group!(benches, execution_context_comparison);
criterion_main!(benches);
