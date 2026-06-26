//! Evidence harness for the `ThreadScheduler` pre-park spin budget (`SPIN_LIMIT`).
//!
//! `SPIN_LIMIT` controls how long an idle worker busy-spins before parking
//! (default 131072, ~6 ms on x86). This measures whether that large spin buys
//! lower wake latency under intermittent load or merely burns idle CPU, by
//! comparing several budgets side by side (the budget is a const generic).
//!
//! `recv_timeout` guards every wait so a missed wakeup is *reported* (as a
//! `lost` count) instead of hanging the run. `#[ignore]`; run with:
//!   cargo test -p moirai-executor --release --test spin_budget_bench -- --ignored --nocapture

use std::sync::mpsc;
use std::time::{Duration, Instant};

use moirai_core::Priority;
use moirai_executor::schedule::{SyncTask, ThreadScheduler};

const WORKERS: usize = 8;
const WAIT: Duration = Duration::from_secs(2);

fn submit_and_wait<const SPIN: usize>(
    sched: &ThreadScheduler<256, SPIN>,
) -> Result<Duration, mpsc::RecvTimeoutError> {
    let (tx, rx) = mpsc::channel();
    let submit = Instant::now();
    sched
        .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
            let _ = tx.send(submit.elapsed());
        })
        .unwrap();
    rx.recv_timeout(WAIT)
}

/// Submit one task per idle `gap` and record submit -> start-of-execution
/// latency. Returns (median, p99, lost-wakeup count).
fn wake_latency<const SPIN: usize>(gap: Duration, iters: usize) -> (Duration, Duration, usize) {
    let sched = ThreadScheduler::<256, SPIN>::new_with_config(WORKERS, "spin-bench").unwrap();
    for _ in 0..(WORKERS * 4) {
        let _ = submit_and_wait(&sched);
    }

    let mut samples = Vec::with_capacity(iters);
    let mut lost = 0usize;
    for _ in 0..iters {
        std::thread::sleep(gap);
        match submit_and_wait(&sched) {
            Ok(latency) => samples.push(latency),
            Err(_) => lost += 1,
        }
    }
    sched.shutdown();

    if samples.is_empty() {
        return (WAIT, WAIT, lost);
    }
    samples.sort_unstable();
    (
        samples[samples.len() / 2],
        samples[samples.len() * 99 / 100],
        lost,
    )
}

/// Submit `tasks` with no gaps (sustained load) and time full drain. Under
/// sustained load workers never run out of work, so the spin budget should not
/// engage and throughput should be independent of `SPIN`.
fn sustained_drain<const SPIN: usize>(tasks: usize) -> Duration {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let sched = ThreadScheduler::<256, SPIN>::new_with_config(WORKERS, "spin-bench").unwrap();
    for _ in 0..(WORKERS * 4) {
        let _ = submit_and_wait(&sched);
    }

    let counter = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();
    for _ in 0..tasks {
        let counter = Arc::clone(&counter);
        sched
            .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                counter.fetch_add(1, Ordering::Relaxed);
            })
            .unwrap();
    }
    sched.join().unwrap();
    let elapsed = start.elapsed();
    assert_eq!(counter.load(Ordering::Relaxed), tasks);
    sched.shutdown();
    elapsed
}

#[test]
#[ignore = "timing instrument; run with --ignored --nocapture"]
fn spin_budget_wake_latency() {
    const ITERS: usize = 500;
    eprintln!("== wake latency / lost-wakeups under intermittent load ({WORKERS} workers) ==");
    for &gap_us in &[100u64, 1000] {
        let gap = Duration::from_micros(gap_us);
        for (label, result) in [
            ("131072", wake_latency::<131072>(gap, ITERS)),
            ("  8192", wake_latency::<8192>(gap, ITERS)),
            ("  1024", wake_latency::<1024>(gap, ITERS)),
            ("   128", wake_latency::<128>(gap, ITERS)),
        ] {
            eprintln!(
                "gap={gap_us:>4}us spin={label}: median {:>8?}  p99 {:>8?}  lost {}/{ITERS}",
                result.0, result.1, result.2
            );
        }
    }

    eprintln!("\n== sustained drain (spin should not engage; expect flat) ==");
    const TASKS: usize = 300_000;
    for trial in 0..3 {
        let t0 = sustained_drain::<131072>(TASKS);
        let t1 = sustained_drain::<8192>(TASKS);
        let t2 = sustained_drain::<1024>(TASKS);
        let t3 = sustained_drain::<128>(TASKS);
        eprintln!(
            "trial {trial}  131072: {t0:>10?} | 8192: {t1:>10?} | 1024: {t2:>10?} | 128: {t3:>10?}"
        );
    }
}
