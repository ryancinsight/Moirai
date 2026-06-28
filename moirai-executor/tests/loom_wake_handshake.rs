//! loom exhaustive-interleaving model of the scheduler's park/wake handshake.
//!
//! A parking worker and a task producer race through a store-buffer (Dekker)
//! handshake. The worker marks itself idle then checks for work; the producer
//! publishes work then scans for an idle worker to wake. The danger is the
//! lost-wakeup outcome: the worker observes no work (and parks) while the
//! producer observes no idle worker (and does not wake it), stranding the task
//! until an unrelated future submission.
//!
//! The production handshake spans two files:
//!   * `schedule/runtime/scheduler/core.rs` — producer: `pending_tasks`
//!     `fetch_add(1, SeqCst)` then `idle_workers.claim_one(..)`.
//!   * `schedule/runtime/idle.rs` — `IdleBitset::set` (`fetch_or(SeqCst)`),
//!     `claim_one` (`load(SeqCst)` + `compare_exchange(SeqCst)`); a parking
//!     worker does `set(id)` then `pending_tasks.load(SeqCst)`.
//!
//! This models the *ordering protocol* — the four atomic accesses — over
//! loom-tracked atoms (loom cannot instrument the production `CacheAligned` /
//! `Box<[..]>` storage directly). loom enumerates every interleaving and asserts
//! the lost-wakeup outcome is unreachable.
//!
//! ## Why the model adds an explicit `SeqCst` fence
//!
//! Production relies on the C++/Rust memory model's guarantee that all `SeqCst`
//! operations share a single total order consistent with program order — which
//! forbids the store-buffer outcome (both sides reading the stale value) with
//! `SeqCst` RMW + `SeqCst` load alone, no fence required. On x86 the
//! `lock`-prefixed RMW is itself the StoreLoad barrier; on weaker targets the
//! `SeqCst` atomics carry the guarantee via their codegen.
//!
//! loom, however, does NOT fully model the SC total order for bare `SeqCst`
//! atomics in this store-buffer shape: it requires an explicit `fence(SeqCst)`
//! between the store and the load to represent the StoreLoad barrier (the same
//! reason `tests/loom_chase_lev.rs` models its protocol with `fence(SeqCst)`).
//! Without the fence loom reports a spurious lost wakeup that the language model
//! and real hardware do not permit; with it, loom confirms the handshake is
//! sound. The fence here is therefore a loom modeling device standing in for the
//! SC StoreLoad barrier that production gets from its `SeqCst` atoms — it is NOT
//! a prescription to add a fence to the production hot path (that would be a
//! redundant extra barrier).
//!
//! ## What this establishes (and the bound it places on the orderings)
//!
//! With the SC StoreLoad barrier present, no interleaving loses a wakeup. The
//! barrier is load-bearing: weakening any of the four accesses to
//! Acquire/Release removes the StoreLoad ordering, readmitting the store-buffer
//! outcome — the worker reads `pending == 0` and parks while the producer reads
//! the idle bit as clear and does not wake it. That is why all four production
//! accesses are `SeqCst`. Keep the orderings here in sync with `core.rs` /
//! `idle.rs`.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo test -p moirai-executor --test loom_wake_handshake --release`
//!
//! Under a normal build the `#![cfg(loom)]` gate makes this file empty, so it
//! never affects the standard test suite or pulls in the `loom` dependency.

#![cfg(loom)]

use loom::sync::atomic::{fence, AtomicU64, AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;

/// Worker id 0 occupies bit 0 of the (single-word) idle bitset.
const WORKER_BIT: u64 = 1;

/// The two shared atoms of the handshake: the task counter and the idle bitset.
struct Handshake {
    /// Mirrors `scheduler::core` `pending_tasks`.
    pending: AtomicUsize,
    /// Mirrors `idle::IdleBitset` (one word; worker 0 -> bit 0).
    idle: AtomicU64,
}

impl Handshake {
    fn new() -> Self {
        Self {
            pending: AtomicUsize::new(0),
            idle: AtomicU64::new(0),
        }
    }

    /// Producer half. Mirrors `core::schedule_with_class`: publish one task with
    /// a `SeqCst` increment, then `claim_one` the (single-worker) bitset with a
    /// `SeqCst` load + `SeqCst` CAS. Returns `true` if it claimed the worker and
    /// will therefore wake it.
    fn submit_and_try_wake(&self) -> bool {
        self.pending.fetch_add(1, Ordering::SeqCst);
        // Models the SC StoreLoad barrier between the publish and the idle scan
        // (see module docs: required for loom, implicit in production's SeqCst).
        fence(Ordering::SeqCst);

        let observed = self.idle.load(Ordering::SeqCst);
        if observed & WORKER_BIT != 0 {
            self.idle
                .compare_exchange(
                    observed,
                    observed & !WORKER_BIT,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
        } else {
            false
        }
    }

    /// Worker half. Mirrors a parking worker: `set(id)` (`fetch_or(SeqCst)`)
    /// then `pending_tasks.load(SeqCst)`. Returns `true` if it would park
    /// (observed no work, leaving its idle bit set for the producer to claim);
    /// `false` if it observed work and took it (clearing its own idle bit).
    fn register_idle_and_check(&self) -> bool {
        self.idle.fetch_or(WORKER_BIT, Ordering::SeqCst);
        // Models the SC StoreLoad barrier between marking idle and the work
        // check (see module docs).
        fence(Ordering::SeqCst);
        if self.pending.load(Ordering::SeqCst) == 0 {
            true
        } else {
            self.idle.fetch_and(!WORKER_BIT, Ordering::SeqCst);
            false
        }
    }
}

#[test]
fn wake_handshake_never_loses_a_wakeup() {
    loom::model(|| {
        let hs = Arc::new(Handshake::new());

        let producer_hs = hs.clone();
        let producer = thread::spawn(move || producer_hs.submit_and_try_wake());

        let parked = hs.register_idle_and_check();
        let woke = producer.join().unwrap();

        // The producer published exactly one task. It is serviced iff the worker
        // took it itself (`!parked`) or the producer claimed the worker to wake
        // it (`woke`). The conjunction `parked && !woke` is the lost wakeup: the
        // task is pending (>=1) yet nobody will run it. The SC StoreLoad barrier
        // across `pending` and `idle` must make this unreachable.
        assert!(
            !(parked && !woke),
            "lost wakeup: worker parked on pending==0 while the producer did not claim the idle worker"
        );
    });
}
