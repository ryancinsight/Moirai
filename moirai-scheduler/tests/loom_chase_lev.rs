//! Bounded Loom model of the Chase-Lev work-stealing deque's
//! steal/pop ordering protocol.
//!
//! The production [`ChaseLevDeque`] (`moirai-scheduler/src/deque/chase_lev.rs`)
//! backs its slots with a raw, custom-allocator `UnsafeCell` array that loom
//! cannot instrument. This file models the *ordering protocol* — the `bottom`
//! (owner) / `top` (thief) indices, the `SeqCst` separating fence, and the
//! read-before-CAS steal — using the EXACT atomic orderings of the production
//! code over a loom-tracked slot store. loom then enumerates every thread
//! interleaving and checks the core invariant: every pushed element is taken
//! exactly once (no loss, no double-take) across a concurrent pop + steal.
//!
//! This complements (does not replace) the real-code stress tests in
//! `tests/deque_concurrency.rs` and the hand proof in `chase_lev.rs`. Keep the
//! orderings here in sync with the production `push`/`pop`/`steal`: any change to
//! the Acquire/Release/SeqCst on `bottom`/`top` or the fence must be mirrored.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo test -p moirai-scheduler --test loom_chase_lev --release`
//!
//! Under a normal build the `#![cfg(loom)]` gate makes this file empty, so it
//! never affects the standard test suite or pulls in the `loom` dependency.

#![cfg(loom)]

use loom::cell::UnsafeCell;
use loom::sync::atomic::{fence, AtomicIsize, Ordering};
use loom::sync::Arc;

/// Fixed capacity large enough for the modelled scenario so no resize/wraparound
/// occurs (the resize path is covered separately by the stress tests).
const CAPACITY: usize = 4;
const MASK: isize = (CAPACITY as isize) - 1;

enum Steal {
    Success(usize),
    Empty,
    Retry,
}

struct ModelCore {
    bottom: AtomicIsize,
    top: AtomicIsize,
    slots: Vec<UnsafeCell<usize>>,
}

impl ModelCore {
    fn new() -> Self {
        let mut slots = Vec::with_capacity(CAPACITY);
        for _ in 0..CAPACITY {
            slots.push(UnsafeCell::new(0));
        }
        Self {
            bottom: AtomicIsize::new(0),
            top: AtomicIsize::new(0),
            slots,
        }
    }

    fn slot(&self, index: isize) -> &UnsafeCell<usize> {
        &self.slots[(index & MASK) as usize]
    }

    /// Owner-only. Mirrors `chase_lev::ChaseLevDeque::push` (single-array case).
    fn push(&self, item: usize) {
        let b = self.bottom.load(Ordering::Relaxed);
        self.slot(b).with_mut(|p| unsafe { *p = item });
        self.bottom.store(b.wrapping_add(1), Ordering::Release);
    }

    /// Owner-only. Mirrors the general (non-x86-fast-path) `pop`.
    fn pop(&self) -> Option<usize> {
        let b = self.bottom.load(Ordering::Relaxed).wrapping_sub(1);
        self.bottom.store(b, Ordering::Relaxed);
        fence(Ordering::SeqCst);
        let t = self.top.load(Ordering::Relaxed);

        if b.wrapping_sub(t) > 0 {
            // More than one element: no contention with thieves on this slot.
            return Some(self.slot(b).with(|p| unsafe { *p }));
        }
        if b.wrapping_sub(t) == 0 {
            // Last element: race a thief for it via the top CAS. Production uses
            // `compare_exchange_weak`; the strong form here models the same
            // correctness race (a lost CAS == a thief won), without the extra
            // spurious-failure interleavings that only re-enter the same path.
            let won = self
                .top
                .compare_exchange(t, t.wrapping_add(1), Ordering::SeqCst, Ordering::Relaxed)
                .is_ok();
            self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
            if won {
                return Some(self.slot(b).with(|p| unsafe { *p }));
            }
            return None;
        }
        self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
        None
    }

    /// Thief. Mirrors `chase_lev::ChaseLevDeque::steal`.
    fn steal(&self) -> Steal {
        let t = self.top.load(Ordering::Acquire);
        fence(Ordering::SeqCst);
        let b = self.bottom.load(Ordering::Acquire);

        if b.wrapping_sub(t) > 0 {
            // Speculative read-before-CAS. The value is `usize` (Copy), so a lost
            // CAS simply discards this copy (the production `mem::forget` is for
            // non-Copy `T`); the winner reads and owns the slot independently.
            let value = self.slot(t).with(|p| unsafe { *p });
            if self
                .top
                .compare_exchange(t, t.wrapping_add(1), Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return Steal::Success(value);
            }
            return Steal::Retry;
        }
        Steal::Empty
    }
}

struct ModelOwner {
    core: Arc<ModelCore>,
}

#[derive(Clone)]
struct ModelStealer {
    core: Arc<ModelCore>,
}

impl ModelOwner {
    fn new() -> (Self, ModelStealer) {
        let core = Arc::new(ModelCore::new());
        (
            Self {
                core: Arc::clone(&core),
            },
            ModelStealer { core },
        )
    }

    fn push(&mut self, item: usize) {
        self.core.push(item);
    }

    fn pop(&mut self) -> Option<usize> {
        self.core.pop()
    }
}

impl ModelStealer {
    fn steal(&self) -> Steal {
        self.core.steal()
    }
}

#[test]
fn chase_lev_pop_steal_take_each_element_exactly_once() {
    loom::model(|| {
        let (mut owner, stealer) = ModelOwner::new();
        // Owner pushes sequentially before spawning, so the pushes happen-before
        // all concurrent access (the owner is the sole pusher in Chase-Lev).
        owner.push(1);
        owner.push(2);

        let thief = loom::thread::spawn(move || {
            let mut taken = Vec::new();
            loop {
                match stealer.steal() {
                    Steal::Success(v) => taken.push(v),
                    Steal::Empty => break,
                    Steal::Retry => continue,
                }
            }
            taken
        });

        let mut owner_taken = Vec::new();
        while let Some(v) = owner.pop() {
            owner_taken.push(v);
        }

        let mut all = owner_taken;
        all.extend(thief.join().expect("thief model must terminate"));
        all.sort_unstable();

        // The invariant: across every interleaving the owner's pops and the
        // thief's steals together take {1, 2} exactly once each — never the same
        // element twice (double-take) and never zero times (lost element).
        assert_eq!(all, vec![1, 2], "every element must be taken exactly once");
    });
}
