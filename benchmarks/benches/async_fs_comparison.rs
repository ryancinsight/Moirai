//! Async file facade comparison benchmarks against Tokio fs.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use std::path::PathBuf;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::runtime::Builder;

const SAMPLE_SIZE: usize = 30;
const MEASUREMENT_MILLIS: u64 = 750;
const WARM_UP_MILLIS: u64 = 250;
const READ_BYTES: usize = 64 * 1024;
const APPEND_PREFIX: &[u8] = b"moirai-append-prefix:";

fn source_bytes() -> Vec<u8> {
    (0..READ_BYTES)
        .map(|index| (index as u8).wrapping_mul(31).wrapping_add(7))
        .collect()
}

fn benchmark_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "moirai_async_fs_comparison_{name}_{}_{}.bin",
        std::process::id(),
        READ_BYTES
    ))
}

fn prepare_file() -> (PathBuf, Vec<u8>) {
    let path = benchmark_path("source");
    let bytes = source_bytes();
    std::fs::write(&path, &bytes).expect("benchmark source file must be writable");
    (path, bytes)
}

fn moirai_read(runtime: &moirai::Moirai, path: &PathBuf) -> Vec<u8> {
    runtime
        .block_on(moirai_async::fs::read(path))
        .expect("moirai fs read must succeed")
}

fn tokio_read(runtime: &tokio::runtime::Runtime, path: &PathBuf) -> Vec<u8> {
    runtime
        .block_on(tokio::fs::read(path))
        .expect("tokio fs read must succeed")
}

fn moirai_write(runtime: &moirai::Moirai, path: &PathBuf, contents: &[u8]) {
    runtime
        .block_on(moirai_async::fs::write(path, contents))
        .expect("moirai fs write must succeed");
}

fn tokio_write(runtime: &tokio::runtime::Runtime, path: &PathBuf, contents: &[u8]) {
    runtime
        .block_on(tokio::fs::write(path, contents))
        .expect("tokio fs write must succeed");
}

fn moirai_append(runtime: &moirai::Moirai, path: &PathBuf, contents: &[u8]) {
    runtime
        .block_on(moirai_async::fs::append(path, contents))
        .expect("moirai fs append must succeed");
}

fn tokio_append(runtime: &tokio::runtime::Runtime, path: &PathBuf, contents: &[u8]) {
    runtime
        .block_on(async {
            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .await?;
            file.write_all(contents).await
        })
        .expect("tokio fs append must succeed");
}

fn moirai_copy(runtime: &moirai::Moirai, source: &PathBuf, dest: &PathBuf) -> u64 {
    let copied = runtime
        .block_on(moirai_async::fs::copy(source, dest))
        .expect("moirai fs copy must succeed");
    assert_eq!(copied, READ_BYTES as u64);
    copied
}

fn tokio_copy(runtime: &tokio::runtime::Runtime, source: &PathBuf, dest: &PathBuf) -> u64 {
    let copied = runtime
        .block_on(tokio::fs::copy(source, dest))
        .expect("tokio fs copy must succeed");
    assert_eq!(copied, READ_BYTES as u64);
    copied
}

fn assert_copied_bytes(path: &PathBuf, expected: &[u8]) {
    let actual = std::fs::read(path).expect("copied file must be readable");
    assert_eq!(actual, expected);
}

fn reset_append_file(path: &PathBuf) {
    std::fs::write(path, APPEND_PREFIX).expect("append benchmark prefix must be writable");
}

fn assert_appended_bytes(path: &PathBuf, appended: &[u8]) {
    let actual = std::fs::read(path).expect("appended file must be readable");
    assert_eq!(actual.len(), APPEND_PREFIX.len() + appended.len());
    assert_eq!(&actual[..APPEND_PREFIX.len()], APPEND_PREFIX);
    assert_eq!(&actual[APPEND_PREFIX.len()..], appended);
}

fn async_fs_comparison(c: &mut Criterion) {
    let (path, expected) = prepare_file();
    let moirai_write_path = benchmark_path("moirai-write");
    let tokio_write_path = benchmark_path("tokio-write");
    let moirai_append_path = benchmark_path("moirai-append");
    let tokio_append_path = benchmark_path("tokio-append");
    let moirai_copy_path = benchmark_path("moirai-copy");
    let tokio_copy_path = benchmark_path("tokio-copy");
    let runtime = Builder::new_multi_thread()
        .worker_threads(num_cpus::get().max(1))
        .enable_all()
        .build()
        .expect("tokio benchmark runtime must build");
    let moirai_runtime = moirai::Moirai::new().expect("moirai benchmark runtime must build");

    let moirai_expected = moirai_read(&moirai_runtime, &path);
    let tokio_expected = tokio_read(&runtime, &path);
    assert_eq!(moirai_expected, expected);
    assert_eq!(tokio_expected, expected);
    moirai_write(&moirai_runtime, &moirai_write_path, &expected);
    assert_copied_bytes(&moirai_write_path, &expected);
    tokio_write(&runtime, &tokio_write_path, &expected);
    assert_copied_bytes(&tokio_write_path, &expected);
    reset_append_file(&moirai_append_path);
    moirai_append(&moirai_runtime, &moirai_append_path, &expected);
    assert_appended_bytes(&moirai_append_path, &expected);
    reset_append_file(&tokio_append_path);
    tokio_append(&runtime, &tokio_append_path, &expected);
    assert_appended_bytes(&tokio_append_path, &expected);
    assert_eq!(
        moirai_copy(&moirai_runtime, &path, &moirai_copy_path),
        READ_BYTES as u64
    );
    assert_copied_bytes(&moirai_copy_path, &expected);
    assert_eq!(
        tokio_copy(&runtime, &path, &tokio_copy_path),
        READ_BYTES as u64
    );
    assert_copied_bytes(&tokio_copy_path, &expected);

    let mut group = c.benchmark_group("async_fs_read_to_end");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", READ_BYTES), &path, |b, input| {
        b.iter(|| black_box(moirai_read(black_box(&moirai_runtime), black_box(input))))
    });
    group.bench_with_input(BenchmarkId::new("tokio", READ_BYTES), &path, |b, input| {
        b.iter(|| black_box(tokio_read(&runtime, black_box(input))))
    });
    group.finish();

    let moirai_write_input = (&moirai_write_path, expected.as_slice());
    let tokio_write_input = (&tokio_write_path, expected.as_slice());
    let mut group = c.benchmark_group("async_fs_write_file");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(
        BenchmarkId::new("moirai", READ_BYTES),
        &moirai_write_input,
        |b, (dest, contents)| {
            b.iter(|| {
                moirai_write(
                    black_box(&moirai_runtime),
                    black_box(dest),
                    black_box(contents),
                )
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("tokio", READ_BYTES),
        &tokio_write_input,
        |b, (dest, contents)| {
            b.iter(|| tokio_write(&runtime, black_box(dest), black_box(contents)))
        },
    );
    group.finish();

    let moirai_append_input = (&moirai_append_path, expected.as_slice());
    let tokio_append_input = (&tokio_append_path, expected.as_slice());
    let mut group = c.benchmark_group("async_fs_append_file");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(
        BenchmarkId::new("moirai", READ_BYTES),
        &moirai_append_input,
        |b, input| {
            b.iter_batched(
                || {
                    reset_append_file(input.0);
                    *input
                },
                |(dest, contents)| {
                    moirai_append(
                        black_box(&moirai_runtime),
                        black_box(dest),
                        black_box(contents),
                    )
                },
                BatchSize::SmallInput,
            )
        },
    );
    group.bench_with_input(
        BenchmarkId::new("tokio", READ_BYTES),
        &tokio_append_input,
        |b, input| {
            b.iter_batched(
                || {
                    reset_append_file(input.0);
                    *input
                },
                |(dest, contents)| tokio_append(&runtime, black_box(dest), black_box(contents)),
                BatchSize::SmallInput,
            )
        },
    );
    group.finish();

    let moirai_copy_input = (&path, &moirai_copy_path);
    let tokio_copy_input = (&path, &tokio_copy_path);
    let mut group = c.benchmark_group("async_fs_copy_file");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(
        BenchmarkId::new("moirai", READ_BYTES),
        &moirai_copy_input,
        |b, (source, dest)| {
            b.iter(|| {
                black_box(moirai_copy(
                    black_box(&moirai_runtime),
                    black_box(source),
                    black_box(dest),
                ))
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("tokio", READ_BYTES),
        &tokio_copy_input,
        |b, (source, dest)| {
            b.iter(|| black_box(tokio_copy(&runtime, black_box(source), black_box(dest))))
        },
    );
    group.finish();

    std::fs::remove_file(&path).expect("benchmark source file cleanup must succeed");
    std::fs::remove_file(&moirai_write_path).expect("moirai written file cleanup must succeed");
    std::fs::remove_file(&tokio_write_path).expect("tokio written file cleanup must succeed");
    std::fs::remove_file(&moirai_append_path).expect("moirai appended file cleanup must succeed");
    std::fs::remove_file(&tokio_append_path).expect("tokio appended file cleanup must succeed");
    std::fs::remove_file(&moirai_copy_path).expect("moirai copied file cleanup must succeed");
    std::fs::remove_file(&tokio_copy_path).expect("tokio copied file cleanup must succeed");
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_millis(MEASUREMENT_MILLIS))
        .warm_up_time(Duration::from_millis(WARM_UP_MILLIS))
        .without_plots();
    targets = async_fs_comparison
}
criterion_main!(benches);
