//! Async file facade comparison benchmarks against Tokio fs.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::PathBuf;
use std::time::Duration;
use tokio::runtime::Builder;

const SAMPLE_SIZE: usize = 30;
const MEASUREMENT_MILLIS: u64 = 750;
const WARM_UP_MILLIS: u64 = 250;
const READ_BYTES: usize = 64 * 1024;

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

fn async_fs_comparison(c: &mut Criterion) {
    let (path, expected) = prepare_file();
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
