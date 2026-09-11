//! Hybrid execution: parallel data-parallel work and async tasks composing on
//! the **one** unified scheduler (`moirai::global()` wraps the same executor
//! that `moirai-parallel` schedules on), within a single process.

#![cfg(all(feature = "parallel", feature = "async"))]

use moirai::ParallelSlice;

async fn identity_async(x: u64) -> u64 {
    x
}

/// A parallel worker drives async work on the same runtime — "asynchronous
/// threads within a parallel process".
#[test]
fn parallel_workers_drive_async_on_unified_runtime() {
    let data: Vec<u64> = (0..8_192).collect(); // above the adaptive threshold => parallel
    let out = data.par().map_collect(|&x| {
        // async work, driven on the shared unified scheduler from inside the
        // parallel worker task
        moirai::global().block_on(async move { identity_async(x).await.wrapping_mul(2) })
    });
    let expected: Vec<u64> = data.iter().map(|&x| x.wrapping_mul(2)).collect();
    assert_eq!(out, expected);
}

/// The other direction: an async context launches parallel compute on the same
/// runtime and gets the result back.
#[test]
fn async_context_runs_parallel_compute_on_same_runtime() {
    let total = moirai::global().block_on(async {
        let data: Vec<u64> = (0..100_000).collect();
        data.par().map_reduce(0u64, |&x| x, |a, b| a + b)
    });
    assert_eq!(total, (0..100_000u64).sum::<u64>());
}
