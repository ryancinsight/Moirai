//! Async iterator comparison benchmarks against Tokio runtime fan-out.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_iter::{
    AsyncIterator as MoiraiAsyncIterator, AsyncParallelIterator as MoiraiAsyncParallelIterator,
    IntoAsyncIterator,
};
use std::task::Poll;
use std::time::Duration;
use tokio::runtime::Builder;
use tokio::task::JoinSet;

const SAMPLE_SIZE: usize = 30;
const MEASUREMENT_MILLIS: u64 = 750;
const WARM_UP_MILLIS: u64 = 250;
const WORK_ITEMS: usize = 32_768;
const DELAYED_WORK_ITEMS: usize = 8_192;
const BOUNDED_CONCURRENCY: usize = 256;
const WINDOW_TAKE: usize = WORK_ITEMS / 2;
const WINDOW_SKIP: usize = 512;

fn source_data() -> Vec<u64> {
    (0..WORK_ITEMS as u64).collect()
}

fn delayed_source_data() -> Vec<u64> {
    (0..DELAYED_WORK_ITEMS as u64).collect()
}

async fn pending_once<T>(value: T) -> T {
    let mut yielded = false;
    let mut value = Some(value);
    futures::future::poll_fn(move |cx| {
        if !yielded {
            yielded = true;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        Poll::Ready(value.take().expect("pending_once polled after completion"))
    })
    .await
}

fn moirai_ready_pipeline(data: Vec<u64>) -> Vec<u64> {
    data.into_async_iter()
        .map(|value| async move { value.wrapping_mul(3) })
        .filter(|value| {
            let value = *value;
            async move { value % 2 == 0 }
        })
        .into_vec()
}

fn moirai_take_skip_pipeline(data: Vec<u64>) -> Vec<u64> {
    data.into_async_iter()
        .map(|value| async move { value.wrapping_mul(5).wrapping_add(1) })
        .take(WINDOW_TAKE)
        .skip(WINDOW_SKIP)
        .into_vec()
}

fn moirai_enumerate_zip_pipeline(left: Vec<u64>, right: Vec<u64>) -> u64 {
    left.into_async_iter()
        .map(|value| async move { value.wrapping_mul(3) })
        .zip(
            right
                .into_async_iter()
                .map(|value| async move { value.wrapping_mul(7) }),
        )
        .enumerate()
        .into_vec()
        .into_iter()
        .fold(0_u64, |accumulator, (index, (left, right))| {
            accumulator.wrapping_add((index as u64) ^ left ^ right)
        })
}

async fn tokio_joinset_ready_pipeline(data: Vec<u64>) -> Vec<u64> {
    let mut tasks = JoinSet::new();
    for (index, value) in data.into_iter().enumerate() {
        tasks.spawn(async move {
            let mapped = value.wrapping_mul(3);
            (index, (mapped % 2 == 0).then_some(mapped))
        });
    }

    let mut indexed = Vec::with_capacity(WORK_ITEMS);
    while let Some(joined) = tasks.join_next().await {
        indexed.push(joined.expect("tokio benchmark task panicked"));
    }
    indexed.sort_unstable_by_key(|(index, _)| *index);
    indexed.into_iter().filter_map(|(_, value)| value).collect()
}

async fn tokio_joinset_take_skip_pipeline(data: Vec<u64>) -> Vec<u64> {
    let mut tasks = JoinSet::new();
    for (index, value) in data.into_iter().enumerate() {
        tasks.spawn(async move { (index, value.wrapping_mul(5).wrapping_add(1)) });
    }

    let mut indexed = Vec::with_capacity(WORK_ITEMS);
    while let Some(joined) = tasks.join_next().await {
        indexed.push(joined.expect("tokio benchmark task panicked"));
    }
    indexed.sort_unstable_by_key(|(index, _)| *index);
    indexed
        .into_iter()
        .map(|(_, value)| value)
        .take(WINDOW_TAKE)
        .skip(WINDOW_SKIP)
        .collect()
}

async fn tokio_joinset_enumerate_zip_pipeline(left: Vec<u64>, right: Vec<u64>) -> u64 {
    let mut left_tasks = JoinSet::new();
    for (index, value) in left.into_iter().enumerate() {
        left_tasks.spawn(async move { (index, value.wrapping_mul(3)) });
    }

    let mut left_values = Vec::with_capacity(WORK_ITEMS);
    while let Some(joined) = left_tasks.join_next().await {
        left_values.push(joined.expect("tokio left benchmark task panicked"));
    }
    left_values.sort_unstable_by_key(|(index, _)| *index);

    let mut right_tasks = JoinSet::new();
    for (index, value) in right.into_iter().enumerate() {
        right_tasks.spawn(async move { (index, value.wrapping_mul(7)) });
    }

    let mut right_values = Vec::with_capacity(WORK_ITEMS);
    while let Some(joined) = right_tasks.join_next().await {
        right_values.push(joined.expect("tokio right benchmark task panicked"));
    }
    right_values.sort_unstable_by_key(|(index, _)| *index);

    left_values
        .into_iter()
        .map(|(_, value)| value)
        .zip(right_values.into_iter().map(|(_, value)| value))
        .enumerate()
        .fold(0_u64, |accumulator, (index, (left, right))| {
            accumulator.wrapping_add((index as u64) ^ left ^ right)
        })
}

fn moirai_bounded_yield_pipeline(data: Vec<u64>) -> Vec<u64> {
    data.into_async_iter()
        .into_parallel()
        .par_map(BOUNDED_CONCURRENCY, |value| {
            pending_once(value.wrapping_mul(3))
        })
        .into_parallel()
        .par_filter(BOUNDED_CONCURRENCY, |value| {
            let value = *value;
            pending_once(value % 2 == 0)
        })
        .into_vec()
}

async fn tokio_bounded_yield_pipeline(data: Vec<u64>) -> Vec<u64> {
    let mut source = data.into_iter().enumerate();
    let mut tasks = JoinSet::new();
    for _ in 0..BOUNDED_CONCURRENCY {
        let Some((index, value)) = source.next() else {
            break;
        };
        tasks.spawn(async move { (index, pending_once(value.wrapping_mul(3)).await) });
    }

    let mut mapped = Vec::with_capacity(DELAYED_WORK_ITEMS);
    while let Some(joined) = tasks.join_next().await {
        mapped.push(joined.expect("tokio bounded map task panicked"));
        if let Some((index, value)) = source.next() {
            tasks.spawn(async move { (index, pending_once(value.wrapping_mul(3)).await) });
        }
    }
    mapped.sort_unstable_by_key(|(index, _)| *index);

    let mut source = mapped.into_iter().enumerate();
    let mut tasks = JoinSet::new();
    for _ in 0..BOUNDED_CONCURRENCY {
        let Some((filter_index, (_source_index, value))) = source.next() else {
            break;
        };
        tasks.spawn(async move { (filter_index, pending_once(value % 2 == 0).await, value) });
    }

    let mut filtered = Vec::with_capacity(DELAYED_WORK_ITEMS);
    while let Some(joined) = tasks.join_next().await {
        filtered.push(joined.expect("tokio bounded filter task panicked"));
        if let Some((filter_index, (_source_index, value))) = source.next() {
            tasks.spawn(async move { (filter_index, pending_once(value % 2 == 0).await, value) });
        }
    }
    filtered.sort_unstable_by_key(|(index, _, _)| *index);
    filtered
        .into_iter()
        .filter_map(|(_, keep, value)| keep.then_some(value))
        .collect()
}

fn async_iterator_comparison(c: &mut Criterion) {
    let data = source_data();
    let delayed_data = delayed_source_data();
    let runtime = Builder::new_multi_thread()
        .worker_threads(num_cpus::get().max(1))
        .enable_all()
        .build()
        .expect("tokio benchmark runtime must build");

    let moirai_expected = moirai_ready_pipeline(data.clone());
    let tokio_expected = runtime.block_on(tokio_joinset_ready_pipeline(data.clone()));
    assert_eq!(moirai_expected, tokio_expected);

    let mut group = c.benchmark_group("async_iterator_ready_pipeline");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_ready_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(
        BenchmarkId::new("tokio_joinset", WORK_ITEMS),
        &data,
        |b, input| {
            b.iter(|| {
                black_box(runtime.block_on(tokio_joinset_ready_pipeline(black_box(input.clone()))))
            })
        },
    );
    group.finish();

    let moirai_expected = moirai_take_skip_pipeline(data.clone());
    let tokio_expected = runtime.block_on(tokio_joinset_take_skip_pipeline(data.clone()));
    assert_eq!(moirai_expected, tokio_expected);

    let mut group = c.benchmark_group("async_iterator_take_skip_pipeline");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_take_skip_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(
        BenchmarkId::new("tokio_joinset", WORK_ITEMS),
        &data,
        |b, input| {
            b.iter(|| {
                black_box(
                    runtime.block_on(tokio_joinset_take_skip_pipeline(black_box(input.clone()))),
                )
            })
        },
    );
    group.finish();

    let moirai_expected = moirai_enumerate_zip_pipeline(data.clone(), data.clone());
    let tokio_expected = runtime.block_on(tokio_joinset_enumerate_zip_pipeline(
        data.clone(),
        data.clone(),
    ));
    assert_eq!(moirai_expected, tokio_expected);

    let mut group = c.benchmark_group("async_iterator_enumerate_zip_pipeline");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| {
            black_box(moirai_enumerate_zip_pipeline(
                black_box(input.clone()),
                black_box(input.clone()),
            ))
        })
    });
    group.bench_with_input(
        BenchmarkId::new("tokio_joinset", WORK_ITEMS),
        &data,
        |b, input| {
            b.iter(|| {
                black_box(runtime.block_on(tokio_joinset_enumerate_zip_pipeline(
                    black_box(input.clone()),
                    black_box(input.clone()),
                )))
            })
        },
    );
    group.finish();

    let moirai_expected = moirai_bounded_yield_pipeline(delayed_data.clone());
    let tokio_expected = runtime.block_on(tokio_bounded_yield_pipeline(delayed_data.clone()));
    assert_eq!(moirai_expected, tokio_expected);

    let mut group = c.benchmark_group("async_iterator_bounded_yield_pipeline");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(
        BenchmarkId::new("moirai", DELAYED_WORK_ITEMS),
        &delayed_data,
        |b, input| b.iter(|| black_box(moirai_bounded_yield_pipeline(black_box(input.clone())))),
    );
    group.bench_with_input(
        BenchmarkId::new("tokio_joinset", DELAYED_WORK_ITEMS),
        &delayed_data,
        |b, input| {
            b.iter(|| {
                black_box(runtime.block_on(tokio_bounded_yield_pipeline(black_box(input.clone()))))
            })
        },
    );
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_millis(MEASUREMENT_MILLIS))
        .warm_up_time(Duration::from_millis(WARM_UP_MILLIS))
        .without_plots();
    targets = async_iterator_comparison
}
criterion_main!(benches);
