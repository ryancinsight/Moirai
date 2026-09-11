//! End-to-end contract for worker idle hooks.
//!
//! A hook registered through the public API must run on executor worker
//! threads when they reach quiescence. This drives a parallel operation over
//! enough chunks to engage the shared pool's workers, then polls the hook
//! counter. A worker may reach quiescence while another chunk is still
//! running, so the callback can execute before the public operation returns;
//! the test records its baseline before submission and waits only for a
//! post-work callback.

use moirai_executor::schedule::register_idle_hook;
use moirai_parallel::{Parallel, for_each_chunk_mut_with};
use std::sync::{
    Condvar, Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

struct HookSignal {
    runs: Mutex<usize>,
    wake: Condvar,
    work_seen: AtomicBool,
}

impl HookSignal {
    fn new() -> Self {
        Self {
            runs: Mutex::new(0),
            wake: Condvar::new(),
            work_seen: AtomicBool::new(false),
        }
    }
}

static HOOK_SIGNAL: OnceLock<HookSignal> = OnceLock::new();

fn hook_signal() -> &'static HookSignal {
    HOOK_SIGNAL.get_or_init(HookSignal::new)
}

fn quiescence_counter_hook() {
    let signal = hook_signal();
    if !signal.work_seen.load(Ordering::Acquire) {
        return;
    }
    let mut runs = signal
        .runs
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *runs += 1;
    signal.wake.notify_all();
}

#[test]
fn registered_hook_runs_on_worker_threads_at_quiescence() {
    let signal = hook_signal();
    let baseline = *signal
        .runs
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    register_idle_hook(quiescence_counter_hook).expect("idle-hook registration capacity");

    // Comfortably more chunks than any plausible pool width, so every worker
    // takes at least one and reaches its own park path afterwards.
    const CHUNK_LEN: usize = 256;
    const CHUNKS: usize = 128;
    let mut data: Vec<u64> = vec![0; CHUNK_LEN * CHUNKS];
    signal.work_seen.store(false, Ordering::Release);
    for_each_chunk_mut_with::<Parallel, _, _>(&mut data, CHUNK_LEN, |chunk| {
        signal.work_seen.store(true, Ordering::Release);
        for value in chunk.iter_mut() {
            *value += 1;
        }
    });

    // Workers park on their own schedule; wait on the hook's condition
    // variable so the test does not poll or sleep while the pool drains.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut runs = signal
        .runs
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    while *runs == baseline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        let (next, _) = signal
            .wake
            .wait_timeout(runs, remaining)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        runs = next;
    }

    let observed = *runs;
    assert!(
        observed > baseline,
        "idle hook must fire on worker threads after work reaches quiescence \
         (baseline {baseline}, final {observed})"
    );
    assert!(observed > baseline);
}
