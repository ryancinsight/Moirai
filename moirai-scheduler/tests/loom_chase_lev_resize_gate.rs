//! Loom model of the Chase-Lev resize exclusion gate.
//!
//! Production storage uses raw pointers and a custom allocator that Loom cannot
//! instrument. This model isolates the storage-lifetime protocol: thieves enter
//! through one encoded `resize_gate`: bit zero is the owner claim and each
//! protected thief contributes two. The sole owner claims resize and waits for
//! the encoded access count to drain before republishing storage. The protocol
//! atomic uses the exact production `SeqCst` ordering. A model-only storage
//! generation uses the production
//! release-publish/acquire-observe ordering and must remain unchanged for the
//! lifetime of every protected access.
//!
//! A thief may speculatively increment the encoded access count after the owner
//! has observed only its claim bit. The value returned by that same RMW rejects
//! the attempt before it observes storage, so transient occupancy is not itself
//! overlap. An RMW after a completed resize is ordered after the owner's claim
//! release even though the encoded value returns to zero.
//!
//! A two-step protected region represents `steal_batch`: the production batch
//! holds one resize access guard across every bounded steal attempt. Two steps
//! are sufficient to model the ordering consequence of a multi-operation hold;
//! increasing the production batch limit does not add a new gate transition.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo nextest run -p moirai-scheduler --test loom_chase_lev_resize_gate --release`

#![cfg(loom)]

use loom::model::Builder;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::{mpsc, Arc};
use loom::thread;

#[path = "../src/deque/chase_lev/gate.rs"]
mod production_gate;
use production_gate::{ResizeGate, StealAccessGuard};

const MODEL_PREEMPTIONS: usize = 3;
const BATCH_STEPS: usize = 2;

struct ResizeGateModel {
    resize_gate: ResizeGate,
    storage_generation: AtomicUsize,
    republish_count: AtomicUsize,
}

struct AccessGuard<'a> {
    gate: &'a ResizeGateModel,
    _admission: StealAccessGuard<'a>,
    observed_generation: Option<usize>,
}

impl AccessGuard<'_> {
    fn protected_step(&mut self) {
        self.assert_generation_unchanged();
        thread::yield_now();
        self.assert_generation_unchanged();
    }

    fn assert_generation_unchanged(&mut self) {
        let generation = self.gate.storage_generation.load(Ordering::Acquire);
        if let Some(observed) = self.observed_generation {
            assert_eq!(
                generation,
                observed,
                "storage generation changed inside a protected access; gate={}",
                self.gate.resize_gate.state()
            );
        } else {
            self.observed_generation = Some(generation);
        }
    }
}

impl ResizeGateModel {
    fn new() -> Self {
        Self {
            resize_gate: ResizeGate::new(),
            storage_generation: AtomicUsize::new(0),
            republish_count: AtomicUsize::new(0),
        }
    }

    fn enter(&self) -> AccessGuard<'_> {
        self.enter_observed(|| {}, || {})
    }

    fn enter_observed(
        &self,
        mut before_attempt: impl FnMut(),
        mut on_backoff: impl FnMut(),
    ) -> AccessGuard<'_> {
        let admission = self.resize_gate.enter(&mut before_attempt, &mut on_backoff);
        AccessGuard {
            gate: self,
            _admission: admission,
            observed_generation: None,
        }
    }

    fn resize(&self) {
        self.resize_observed(|| {});
    }

    fn resize_observed(&self, after_claim: impl FnOnce()) {
        let claim = self.resize_gate.claim(after_claim);

        thread::yield_now();
        let generation = self.storage_generation.load(Ordering::Relaxed);
        self.storage_generation
            .store(generation + 1, Ordering::Release);
        self.republish_count.fetch_add(1, Ordering::SeqCst);
        drop(claim);
    }

    fn assert_idle(&self, expected_republishes: usize) {
        assert_eq!(self.resize_gate.state(), 0);
        assert_eq!(
            self.storage_generation.load(Ordering::Acquire),
            expected_republishes
        );
        assert_eq!(
            self.republish_count.load(Ordering::SeqCst),
            expected_republishes
        );
    }
}

fn model_builder() -> Builder {
    let mut builder = Builder::new();
    builder.preemption_bound = Some(MODEL_PREEMPTIONS);
    builder
}

#[test]
fn concurrent_entry_and_resize_never_overlap() {
    model_builder().check(|| {
        let gate = Arc::new(ResizeGateModel::new());

        let thief_gate = Arc::clone(&gate);
        let thief = thread::spawn(move || {
            let mut access = thief_gate.enter();
            access.protected_step();
        });

        let resize_gate = Arc::clone(&gate);
        let owner = thread::spawn(move || resize_gate.resize());

        thief.join().expect("thief model must terminate");
        owner.join().expect("resize model must terminate");
        gate.assert_idle(1);
    });
}

#[test]
fn entry_retries_after_observing_resize() {
    model_builder().check(|| {
        let gate = Arc::new(ResizeGateModel::new());

        let (claimed_tx, claimed_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let owner_gate = Arc::clone(&gate);
        let owner = thread::spawn(move || {
            owner_gate.resize_observed(|| {
                claimed_tx
                    .send(())
                    .expect("resize claim observer must remain connected");
                release_rx
                    .recv()
                    .expect("resize release observer must remain connected");
            });
        });
        claimed_rx
            .recv()
            .expect("owner must claim the resize gate before thief entry");

        let (backoff_tx, backoff_rx) = mpsc::channel();
        let thief_gate = Arc::clone(&gate);
        let thief = thread::spawn(move || {
            let mut first_backoff = Some(backoff_tx);
            let mut access = thief_gate.enter_observed(
                || {},
                || {
                    if let Some(sender) = first_backoff.take() {
                        sender
                            .send(())
                            .expect("backoff observer must remain connected");
                    }
                },
            );
            access.protected_step();
            drop(access);
        });

        backoff_rx
            .recv()
            .expect("thief must observe the claimed resize gate");
        release_tx
            .send(())
            .expect("claimed resize must remain connected");

        thief.join().expect("retrying thief must terminate");
        owner.join().expect("observed resize must terminate");
        gate.assert_idle(1);
    });
}

#[test]
fn entry_after_completed_resize_observes_published_generation() {
    model_builder().check(|| {
        let gate = Arc::new(ResizeGateModel::new());
        let phase = Arc::new(AtomicUsize::new(0));

        let owner_gate = Arc::clone(&gate);
        let owner_phase = Arc::clone(&phase);
        let owner = thread::spawn(move || {
            owner_gate.resize_observed(|| {
                owner_phase.store(1, Ordering::Relaxed);
                while owner_phase.load(Ordering::Relaxed) != 2 {
                    thread::yield_now();
                }
            });
        });

        let thief_gate = Arc::clone(&gate);
        let thief_phase = Arc::clone(&phase);
        let thief = thread::spawn(move || {
            while thief_phase.load(Ordering::Relaxed) != 1 {
                thread::yield_now();
            }
            let mut access =
                thief_gate.enter_observed(|| {}, || thief_phase.store(2, Ordering::Relaxed));
            access.protected_step();
            assert_eq!(
                access.observed_generation,
                Some(1),
                "admission after claim release must observe the published generation"
            );
        });

        thief
            .join()
            .expect("generation-checking thief must terminate");
        owner.join().expect("observed resize must terminate");
        gate.assert_idle(1);
    });
}

fn check_resize_waits_for_access<const STEPS: usize>() {
    model_builder().check(|| {
        let gate = Arc::new(ResizeGateModel::new());
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();

        let thief_gate = Arc::clone(&gate);
        let thief = thread::spawn(move || {
            let mut access = thief_gate.enter();
            entered_tx
                .send(())
                .expect("entry observer must remain connected");
            release_rx
                .recv()
                .expect("access release must remain connected");
            for _ in 0..STEPS {
                access.protected_step();
            }
        });

        entered_rx
            .recv()
            .expect("thief must hold the access region before resize");

        let (claimed_tx, claimed_rx) = mpsc::channel();
        let resize_gate = Arc::clone(&gate);
        let owner = thread::spawn(move || {
            resize_gate.resize_observed(|| {
                claimed_tx
                    .send(())
                    .expect("resize observer must remain connected");
            });
        });

        claimed_rx
            .recv()
            .expect("owner must claim resize before the access is released");
        release_tx
            .send(())
            .expect("held access must remain connected");

        thief.join().expect("held thief must terminate");
        owner.join().expect("waiting resize must terminate");
        gate.assert_idle(1);
    });
}

#[test]
fn resize_waits_for_single_access_region() {
    check_resize_waits_for_access::<1>();
}

#[test]
fn resize_waits_for_batch_access_region() {
    check_resize_waits_for_access::<BATCH_STEPS>();
}
