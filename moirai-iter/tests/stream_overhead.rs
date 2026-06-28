//! Per-item overhead instrument for `ConcurrentStreamExt::concurrent_map`.
//!
//! Profile-first evidence for two questions:
//!  1. How much does the distributed path (`limit > 1`) pay per item for the
//!     spawn + one-shot hop? This is the cost a future async-`TaskHandle`
//!     optimization (awaiting the spawned result directly, no one-shot) would
//!     remove.
//!  2. Does the `limit == 1` inline fast-path actually avoid that cost?
//!
//! Trivial item work (identity) isolates the dispatch overhead from real work.
//! `#[ignore]` — a timing instrument, not a correctness gate. Run:
//!   cargo test -p moirai-iter --release --test stream_overhead -- --ignored --nocapture

use std::time::Instant;

use futures::StreamExt;
use moirai_iter::stream::ConcurrentStreamExt;

const ITEMS: u64 = 50_000;

fn ns_per_item(limit: usize) -> f64 {
    let start = Instant::now();
    let sum: u64 = futures::executor::block_on(
        futures::stream::iter(0..ITEMS)
            // Identity item work: the only cost is the combinator's dispatch.
            .concurrent_map(limit, |x| async move { x })
            .fold(0u64, |acc, x| async move { acc + x }),
    );
    let elapsed = start.elapsed();
    assert_eq!(sum, (0..ITEMS).sum::<u64>());
    elapsed.as_nanos() as f64 / ITEMS as f64
}

#[test]
#[ignore = "timing instrument; run with --ignored --nocapture"]
fn concurrent_map_per_item_overhead() {
    eprintln!("== concurrent_map per-item dispatch overhead, identity work ({ITEMS} items) ==");
    for &limit in &[1usize, 8, 32] {
        // Best of 3 to discount scheduling jitter and one-off warmup.
        let mut best = f64::MAX;
        for _ in 0..3 {
            best = best.min(ns_per_item(limit));
        }
        let mode = if limit == 1 { "inline" } else { "spawn " };
        eprintln!("limit={limit:>3} [{mode}]: {best:>9.1} ns/item");
    }
    eprintln!("(inline vs spawn delta = per-item spawn + one-shot cost)");
}
