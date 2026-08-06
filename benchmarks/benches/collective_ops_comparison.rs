//! Criterion comparison: jagged `Vec<Vec<T>>` vs CSR-shaped `ChunkedVec<T>`
//! on the collective-operation traversal paths (ATLAS-ARCH-008).
//!
//! The pre-conversion jagged formulations are kept inline as baselines so the
//! traversal win of the flat-buffer layout is measurable head-to-head: one
//! contiguous allocation plus an offset table instead of a `Vec` per chunk.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_core::communication::{ChunkedVec, CollectiveOps};

/// Jagged baseline: the pre-conversion `scatter` formulation.
fn jagged_scatter<T: Clone>(data: &[T], num_participants: usize) -> Vec<Vec<T>> {
    let chunk_size = data.len().div_ceil(num_participants).max(1);
    data.chunks(chunk_size).map(<[T]>::to_vec).collect()
}

/// Jagged baseline: the pre-conversion `gather` formulation.
fn jagged_gather<T>(chunks: Vec<Vec<T>>) -> Vec<T> {
    chunks.into_iter().flatten().collect()
}

/// Jagged baseline: the pre-conversion `all_to_all` transpose.
fn jagged_all_to_all<T: Clone>(data: &[Vec<T>]) -> Vec<Vec<T>> {
    let n = data.len();
    let mut result = vec![Vec::new(); n];
    for row in data {
        for (j, item) in row.iter().enumerate() {
            if j < n {
                result[j].push(item.clone());
            }
        }
    }
    result
}

/// Traversal over the jagged layout: per-chunk allocations and pointer chase.
fn jagged_traverse_sum(chunks: &[Vec<u64>]) -> u64 {
    chunks
        .iter()
        .fold(0u64, |acc, chunk| acc + chunk.iter().sum::<u64>())
}

/// Traversal over the CSR layout: one contiguous buffer, sliced by offsets.
fn csr_traverse_sum(chunked: &ChunkedVec<u64>) -> u64 {
    chunked
        .chunks()
        .fold(0u64, |acc, chunk| acc + chunk.iter().sum::<u64>())
}

/// (participants, items) workload pairs.
fn pairs() -> Vec<(usize, usize)> {
    vec![(32, 4096), (128, 8192)]
}

pub fn collective_ops_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("collective_scatter");
    group.sample_size(10);
    for (participants, items) in pairs() {
        let data: Vec<u64> = (0..items as u64).collect();
        group.bench_with_input(
            BenchmarkId::new("jagged", participants),
            &data,
            |b, data| b.iter(|| black_box(jagged_scatter(data, participants))),
        );
        group.bench_with_input(BenchmarkId::new("csr", participants), &data, |b, data| {
            b.iter(|| black_box(CollectiveOps::scatter(data.clone(), participants)))
        });
    }
    group.finish();

    let mut group = c.benchmark_group("collective_gather");
    group.sample_size(10);
    for (participants, items) in pairs() {
        let data: Vec<u64> = (0..items as u64).collect();
        let jagged = jagged_scatter(&data, participants);
        let chunked = CollectiveOps::scatter(data, participants);
        group.bench_with_input(
            BenchmarkId::new("jagged", participants),
            &jagged,
            |b, chunks| b.iter(|| black_box(jagged_gather(chunks.clone()))),
        );
        group.bench_with_input(
            BenchmarkId::new("csr", participants),
            &chunked,
            |b, chunked| b.iter(|| black_box(CollectiveOps::gather(chunked.clone()))),
        );
    }
    group.finish();

    let mut group = c.benchmark_group("collective_traverse");
    group.sample_size(10);
    for (participants, items) in pairs() {
        let data: Vec<u64> = (0..items as u64).collect();
        let jagged = jagged_scatter(&data, participants);
        let chunked = CollectiveOps::scatter(data, participants);
        group.bench_with_input(
            BenchmarkId::new("jagged", participants),
            &jagged,
            |b, chunks| b.iter(|| black_box(jagged_traverse_sum(chunks))),
        );
        group.bench_with_input(
            BenchmarkId::new("csr", participants),
            &chunked,
            |b, chunked| b.iter(|| black_box(csr_traverse_sum(chunked))),
        );
    }
    group.finish();

    let mut group = c.benchmark_group("collective_all_to_all");
    group.sample_size(10);
    for (participants, items) in pairs() {
        let data: Vec<u64> = (0..items as u64).collect();
        let jagged = jagged_scatter(&data, participants);
        let chunked = CollectiveOps::scatter(data, participants);
        group.bench_with_input(
            BenchmarkId::new("jagged", participants),
            &jagged,
            |b, rows| b.iter(|| black_box(jagged_all_to_all(rows))),
        );
        group.bench_with_input(
            BenchmarkId::new("csr", participants),
            &chunked,
            |b, chunked| b.iter(|| black_box(CollectiveOps::all_to_all(chunked.clone()))),
        );
    }
    group.finish();
}

criterion_group!(benches, collective_ops_comparison);
criterion_main!(benches);
