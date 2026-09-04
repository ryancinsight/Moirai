//! End-to-end contract for worker idle hooks.
//!
//! A hook registered through the public API must run on executor worker
//! threads when they reach quiescence. This drives a parallel operation over
//! enough chunks to engage the shared pool's workers, then polls the hook
//! counter: shortly after the operation completes the workers exhaust their
//! spin budget, find no further work, and run their idle hooks right before
//! blocking — so the counter must advance without the test thread doing
//! anything besides waiting.

use core::sync::atomic::{AtomicUsize, Ordering};
use moirai_executor::schedule::register_idle_hook;
use moirai_parallel::{for_each_chunk_mut_with, Parallel};
use std::time::{Duration, Instant};

static HOOK_RUNS: AtomicUsize = AtomicUsize::new(0);

fn quiescence_counter_hook() {
    HOOK_RUNS.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn registered_hook_runs_on_worker_threads_at_quiescence() {
    let baseline = HOOK_RUNS.load(Ordering::SeqCst);
    register_idle_hook(quiescence_counter_hook);

    // Comfortably more chunks than any plausible pool width, so every worker
    // takes at least one and reaches its own park path afterwards.
    const CHUNK_LEN: usize = 256;
    const CHUNKS: usize = 128;
    let mut data: Vec<u64> = vec![0; CHUNK_LEN * CHUNKS];
    for_each_chunk_mut_with::<Parallel, _, _>(&mut data, CHUNK_LEN, |chunk| {
        for value in chunk.iter_mut() {
            *value += 1;
        }
    });
    let after_operation = HOOK_RUNS.load(Ordering::SeqCst);

    // Workers park on their own schedule; poll for the quiescence window
    // rather than racing a fixed sleep against pool teardown.
    let deadline = Instant::now() + Duration::from_secs(5);
    while HOOK_RUNS.load(Ordering::SeqCst) == after_operation && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }

    let observed = HOOK_RUNS.load(Ordering::SeqCst);
    assert!(
        observed > after_operation,
        "idle hook must fire on worker threads after the pool drains \
         (baseline {baseline}, after operation {after_operation}, final {observed})"
    );
    assert!(observed > baseline);
}
