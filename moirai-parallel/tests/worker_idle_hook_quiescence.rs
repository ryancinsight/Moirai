//! End-to-end contract for worker idle hooks.
//!
//! A hook registered through the public API must run on executor worker
//! threads when they reach quiescence. This drives a parallel operation over
//! enough chunks to engage the shared pool's workers. After the operation
//! completes, the workers exhaust their spin budget, find no further work, and
//! run their idle hooks right before blocking — so the condition variable must
//! be signalled without the test thread polling scheduler state.

use moirai_executor::schedule::register_idle_hook;
use moirai_parallel::{for_each_chunk_mut_with, Parallel};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::Duration;

static HOOK_SIGNAL: OnceLock<(Mutex<usize>, Condvar)> = OnceLock::new();

fn quiescence_counter_hook() {
    let (state, signal) = HOOK_SIGNAL.get_or_init(|| (Mutex::new(0), Condvar::new()));
    let mut fired = state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *fired += 1;
    signal.notify_all();
}

#[test]
fn registered_hook_runs_on_worker_threads_at_quiescence() {
    let (state, signal) = HOOK_SIGNAL.get_or_init(|| (Mutex::new(0), Condvar::new()));
    register_idle_hook(quiescence_counter_hook).expect("worker idle hook registry has capacity");
    let generation = *state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

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
    let fired = state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (fired, timeout) = signal
        .wait_timeout_while(fired, Duration::from_secs(5), |fired| *fired <= generation)
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert!(
        *fired > generation && !timeout.timed_out(),
        "idle hook must fire on worker threads after the pool drains"
    );
}
