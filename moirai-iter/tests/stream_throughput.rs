//! Throughput instrument for `ConcurrentStreamExt::concurrent_map`.
//!
//! Demonstrates that the `limit` (bounded concurrency) actually scales
//! throughput for a latency-bound workload by dispatching items across the
//! unified scheduler's workers: N items each holding ~`HOLD` should take about
//! `N/limit * HOLD` wall-clock, not `N * HOLD`. `#[ignore]` (a timing
//! instrument, not a correctness gate; the bound itself is asserted in the unit
//! tests). Run:
//!   cargo test -p moirai-iter --release --test stream_throughput -- --ignored --nocapture

use std::time::{Duration, Instant};

use futures::StreamExt;
use moirai_iter::stream::ConcurrentStreamExt;

const ITEMS: u64 = 256;
const HOLD: Duration = Duration::from_millis(4);

fn run_at_limit(limit: usize) -> Duration {
    let start = Instant::now();
    let processed: u64 = futures::executor::block_on(
        futures::stream::iter(0..ITEMS)
            .concurrent_map(limit, |x| async move {
                // Stand-in for a latency-bound item (I/O wait): occupies the
                // worker for HOLD, so `limit` items overlap across workers.
                std::thread::sleep(HOLD);
                x
            })
            .fold(0u64, |acc, _| async move { acc + 1 }),
    );
    assert_eq!(processed, ITEMS);
    start.elapsed()
}

#[test]
#[ignore = "timing instrument; run with --ignored --nocapture"]
fn concurrent_map_throughput_scales_with_limit() {
    let serial = HOLD * ITEMS as u32;
    eprintln!(
        "== concurrent_map throughput ({ITEMS} items, {HOLD:?} each; serial floor ~{serial:?}) =="
    );
    for &limit in &[1usize, 4, 16, 64] {
        let elapsed = run_at_limit(limit);
        let ideal = serial / limit.min(ITEMS as usize) as u32;
        eprintln!(
            "limit={limit:>3}: {elapsed:>10?}  ({:>6.1} items/s, ideal ~{ideal:?})",
            ITEMS as f64 / elapsed.as_secs_f64()
        );
    }
}
