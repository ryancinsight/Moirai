//! Async directory facade comparison benchmarks against Tokio fs.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::runtime::Builder;

const SAMPLE_SIZE: usize = 30;
const MEASUREMENT_MILLIS: u64 = 750;
const WARM_UP_MILLIS: u64 = 250;
const DIR_UNITS: usize = 1;
const TREE_MARKER_BYTES: &[u8] = b"moirai-directory-tree-marker";

fn benchmark_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "moirai_async_fs_dir_comparison_{name}_{}_{}",
        std::process::id(),
        DIR_UNITS
    ))
}

fn remove_existing_tree(path: &PathBuf) {
    let _ = std::fs::remove_dir_all(path);
}

fn tree_leaf(root: &Path) -> PathBuf {
    root.join("alpha").join("beta").join("gamma")
}

fn assert_removed(path: &Path) {
    assert!(!path.exists());
}

fn moirai_create_remove_dir(runtime: &moirai::Moirai, path: &PathBuf) {
    runtime
        .block_on(async {
            moirai_async::fs::create_dir(path).await?;
            assert!(std::fs::metadata(path)?.is_dir());
            moirai_async::fs::remove_dir(path).await
        })
        .expect("moirai directory create/remove must succeed");
    assert_removed(path);
}

fn tokio_create_remove_dir(runtime: &tokio::runtime::Runtime, path: &PathBuf) {
    runtime
        .block_on(async {
            tokio::fs::create_dir(path).await?;
            assert!(std::fs::metadata(path)?.is_dir());
            tokio::fs::remove_dir(path).await
        })
        .expect("tokio directory create/remove must succeed");
    assert_removed(path);
}

fn write_tree_marker(leaf: &Path) -> io::Result<()> {
    let marker = leaf.join("marker.bin");
    std::fs::write(&marker, TREE_MARKER_BYTES)?;
    let actual = std::fs::read(marker)?;
    assert_eq!(actual, TREE_MARKER_BYTES);
    Ok(())
}

fn moirai_create_remove_dir_all(runtime: &moirai::Moirai, root: &PathBuf) {
    let leaf = tree_leaf(root);
    runtime
        .block_on(async {
            moirai_async::fs::create_dir_all(&leaf).await?;
            assert!(leaf.is_dir());
            write_tree_marker(&leaf)?;
            moirai_async::fs::remove_dir_all(root).await
        })
        .expect("moirai recursive directory create/remove must succeed");
    assert_removed(root);
}

fn tokio_create_remove_dir_all(runtime: &tokio::runtime::Runtime, root: &PathBuf) {
    let leaf = tree_leaf(root);
    runtime
        .block_on(async {
            tokio::fs::create_dir_all(&leaf).await?;
            assert!(leaf.is_dir());
            write_tree_marker(&leaf)?;
            tokio::fs::remove_dir_all(root).await
        })
        .expect("tokio recursive directory create/remove must succeed");
    assert_removed(root);
}

fn async_fs_dir_comparison(c: &mut Criterion) {
    let moirai_dir_path = benchmark_path("moirai-dir");
    let tokio_dir_path = benchmark_path("tokio-dir");
    let moirai_dir_all_path = benchmark_path("moirai-dir-all");
    let tokio_dir_all_path = benchmark_path("tokio-dir-all");
    let runtime = Builder::new_multi_thread()
        .worker_threads(num_cpus::get().max(1))
        .enable_all()
        .build()
        .expect("tokio benchmark runtime must build");
    let moirai_runtime = moirai::Moirai::new().expect("moirai benchmark runtime must build");

    remove_existing_tree(&moirai_dir_path);
    moirai_create_remove_dir(&moirai_runtime, &moirai_dir_path);
    remove_existing_tree(&tokio_dir_path);
    tokio_create_remove_dir(&runtime, &tokio_dir_path);
    remove_existing_tree(&moirai_dir_all_path);
    moirai_create_remove_dir_all(&moirai_runtime, &moirai_dir_all_path);
    remove_existing_tree(&tokio_dir_all_path);
    tokio_create_remove_dir_all(&runtime, &tokio_dir_all_path);

    let mut group = c.benchmark_group("async_fs_create_remove_dir");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(
        BenchmarkId::new("moirai", DIR_UNITS),
        &moirai_dir_path,
        |b, path| {
            b.iter_batched(
                || {
                    remove_existing_tree(path);
                    path
                },
                |path| moirai_create_remove_dir(black_box(&moirai_runtime), black_box(path)),
                BatchSize::PerIteration,
            )
        },
    );
    group.bench_with_input(
        BenchmarkId::new("tokio", DIR_UNITS),
        &tokio_dir_path,
        |b, path| {
            b.iter_batched(
                || {
                    remove_existing_tree(path);
                    path
                },
                |path| tokio_create_remove_dir(&runtime, black_box(path)),
                BatchSize::PerIteration,
            )
        },
    );
    group.finish();

    let mut group = c.benchmark_group("async_fs_create_remove_dir_all");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(
        BenchmarkId::new("moirai", DIR_UNITS),
        &moirai_dir_all_path,
        |b, path| {
            b.iter_batched(
                || {
                    remove_existing_tree(path);
                    path
                },
                |path| moirai_create_remove_dir_all(black_box(&moirai_runtime), black_box(path)),
                BatchSize::PerIteration,
            )
        },
    );
    group.bench_with_input(
        BenchmarkId::new("tokio", DIR_UNITS),
        &tokio_dir_all_path,
        |b, path| {
            b.iter_batched(
                || {
                    remove_existing_tree(path);
                    path
                },
                |path| tokio_create_remove_dir_all(&runtime, black_box(path)),
                BatchSize::PerIteration,
            )
        },
    );
    group.finish();

    remove_existing_tree(&moirai_dir_path);
    remove_existing_tree(&tokio_dir_path);
    remove_existing_tree(&moirai_dir_all_path);
    remove_existing_tree(&tokio_dir_all_path);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_millis(MEASUREMENT_MILLIS))
        .warm_up_time(Duration::from_millis(WARM_UP_MILLIS))
        .without_plots();
    targets = async_fs_dir_comparison
}
criterion_main!(benches);
