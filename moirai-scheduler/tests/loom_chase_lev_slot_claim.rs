//! Bounded Loom model of the Chase-Lev deque's slot-claim protocol, with the
//! owner's fence-free pop fast path forced.
//!
//! The production deque (`moirai-scheduler/src/deque/chase_lev.rs`) arbitrates
//! every slot between the owner and the thieves with a per-slot state word: a
//! taker moves `state == index` to `!index` (the claim), moves the value out,
//! then either restores `index` (`publish`, the slot reads as ready again) or
//! moves it to `index + capacity` (`release`, the slot belongs to the next
//! lap). This file restates that protocol over loom-tracked atomics with the
//! orderings the production code uses, because the production slots are raw
//! allocator memory that loom cannot instrument.
//!
//! A state equal to `index` marks both "item present" and "slot free", so a
//! claim succeeds on a slot whose item the owner already moved out and
//! republished. What keeps a thief from taking such a slot is that it read
//! `bottom` before claiming: the owner's `SeqCst` fence in `pop` makes the
//! decremented `bottom` visible to any thief whose own fence follows. The
//! owner's x86 fast path skips that fence when `bottom - top >= MAX_BATCH_STEAL`.
//! Skipping it lets a thief hold a `bottom` older than the owner's pop, a delay
//! the x86-TSO abstract machine leaves unbounded, and claim the republished
//! slot. The model reaches that double take with the fast path forced and
//! shows that re-reading `bottom` after the claim, which the thief's claim
//! synchronizes with the owner's publish for, removes it.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo test -p moirai-scheduler --test loom_chase_lev_slot_claim --release`
//!
//! Under a normal build the `#![cfg(loom)]` gate makes this file empty.

#![cfg(loom)]

use loom::cell::UnsafeCell;
use loom::sync::Arc;
use loom::sync::atomic::{AtomicIsize, Ordering, fence};

const CAPACITY: isize = 4;

/// The production threshold is `MAX_BATCH_STEAL`; one is the smallest value
/// that still leaves the owner a slot to pop below the thieves' `top`.
const FAST_POP_THRESHOLD: isize = 1;

enum Steal {
    Success(usize),
    Empty,
    Retry,
}

/// Which owner and thief variants the model runs.
#[derive(Clone, Copy)]
struct Protocol {
    /// The owner skips the pop fence when `bottom - top >= FAST_POP_THRESHOLD`.
    fast_pop: bool,
    /// A thief re-reads `bottom` after winning the slot claim and gives the
    /// slot back when the deque no longer covers its index.
    revalidate_after_claim: bool,
}

struct ModelCore {
    protocol: Protocol,
    bottom: AtomicIsize,
    top: AtomicIsize,
    states: Vec<AtomicIsize>,
    slots: Vec<UnsafeCell<usize>>,
}

impl ModelCore {
    fn new(protocol: Protocol) -> Self {
        Self {
            protocol,
            bottom: AtomicIsize::new(0),
            top: AtomicIsize::new(0),
            states: (0..CAPACITY).map(AtomicIsize::new).collect(),
            slots: (0..CAPACITY).map(|_| UnsafeCell::new(0)).collect(),
        }
    }

    fn slot_of(index: isize) -> usize {
        (index & (CAPACITY - 1)) as usize
    }

    fn claim(&self, index: isize) -> bool {
        self.states[Self::slot_of(index)]
            .compare_exchange(index, !index, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    fn publish(&self, index: isize) {
        self.states[Self::slot_of(index)].store(index, Ordering::Release);
    }

    fn release(&self, index: isize) {
        self.states[Self::slot_of(index)].store(index + CAPACITY, Ordering::Release);
    }

    fn read(&self, index: isize) -> usize {
        self.slots[Self::slot_of(index)].with(|slot| {
            // SAFETY: the caller holds the slot claim, so no writer runs.
            unsafe { *slot }
        })
    }

    /// Owner-only. Mirrors `ChaseLevInner::push` for a deque that never grows.
    fn push(&self, item: usize) {
        let b = self.bottom.load(Ordering::Relaxed);
        assert!(self.claim(b), "the owner claims a free slot");
        self.slots[Self::slot_of(b)].with_mut(|slot| {
            // SAFETY: the claim makes the slot owner-exclusive.
            unsafe { *slot = item };
        });
        self.publish(b);
        self.bottom.store(b + 1, Ordering::Release);
    }

    /// Owner-only. Mirrors `ChaseLevInner::pop`.
    fn pop(&self) -> Option<usize> {
        let b = self.bottom.load(Ordering::Relaxed) - 1;
        self.bottom.store(b, Ordering::Relaxed);

        if self.protocol.fast_pop {
            let t = self.top.load(Ordering::Relaxed);
            if b - t >= FAST_POP_THRESHOLD {
                if self.claim(b) {
                    let item = self.read(b);
                    self.publish(b);
                    return Some(item);
                }
                self.bottom.store(b + 1, Ordering::Relaxed);
                return None;
            }
        }

        fence(Ordering::SeqCst);
        let t = self.top.load(Ordering::Relaxed);

        if b - t > 0 {
            if self.claim(b) {
                let item = self.read(b);
                self.publish(b);
                return Some(item);
            }
            self.bottom.store(b + 1, Ordering::Relaxed);
            return None;
        }

        if b - t == 0 {
            if !self.claim(t) {
                self.bottom.store(b + 1, Ordering::Relaxed);
                return None;
            }
            if self
                .top
                .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                self.bottom.store(b + 1, Ordering::Relaxed);
                let item = self.read(b);
                self.release(b);
                return Some(item);
            }
            self.publish(t);
            self.bottom.store(b + 1, Ordering::Relaxed);
            return None;
        }

        self.bottom.store(b + 1, Ordering::Relaxed);
        None
    }

    /// Thief. Mirrors `ChaseLevInner::steal_within_access`.
    fn steal(&self) -> Steal {
        let t = self.top.load(Ordering::Acquire);
        fence(Ordering::SeqCst);
        let b = self.bottom.load(Ordering::Acquire);

        if b - t <= 0 {
            return Steal::Empty;
        }
        if !self.claim(t) {
            return Steal::Retry;
        }
        if self.protocol.revalidate_after_claim && self.bottom.load(Ordering::Acquire) - t <= 0 {
            self.publish(t);
            return Steal::Retry;
        }
        if self
            .top
            .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            let item = self.read(t);
            self.release(t);
            return Steal::Success(item);
        }
        self.publish(t);
        Steal::Retry
    }
}

/// Every item of `items` is taken exactly once when the owner pops against
/// `thieves` threads that each steal until the deque reads empty.
fn run_pop_against_steals(
    protocol: Protocol,
    items: usize,
    thieves: usize,
    preemption_bound: Option<usize>,
) {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = preemption_bound;
    builder.check(move || {
        let core = Arc::new(ModelCore::new(protocol));
        // The pushes happen-before every concurrent access: the owner is the
        // only pusher and spawns the thieves afterwards.
        for item in 1..=items {
            core.push(item);
        }

        let handles: Vec<_> = (0..thieves)
            .map(|_| {
                let thief_core = Arc::clone(&core);
                loom::thread::spawn(move || {
                    let mut taken = Vec::new();
                    loop {
                        match thief_core.steal() {
                            Steal::Success(item) => taken.push(item),
                            Steal::Empty => break,
                            Steal::Retry => loom::thread::yield_now(),
                        }
                    }
                    taken
                })
            })
            .collect();

        let mut all = Vec::new();
        while let Some(item) = core.pop() {
            all.push(item);
        }
        for handle in handles {
            all.extend(handle.join().expect("the thief model terminates"));
        }
        // A thief can read the owner's transient `bottom - 1` as empty, and the
        // owner can lose a slot race to a thief that then gives the slot back,
        // so everyone may stop with an item still in the deque. The owner
        // drains what is left single-threaded; the item stays in the
        // structure, neither lost nor taken twice.
        while let Some(item) = core.pop() {
            all.push(item);
        }
        all.sort_unstable();
        assert!(
            all.windows(2).all(|pair| pair[0] != pair[1]),
            "an item was taken twice: {all:?}"
        );
        assert_eq!(
            all,
            (1..=items).collect::<Vec<_>>(),
            "an item was lost: {all:?}"
        );
    });
}

const FENCED: Protocol = Protocol {
    fast_pop: false,
    revalidate_after_claim: false,
};
const FENCE_FREE: Protocol = Protocol {
    fast_pop: true,
    revalidate_after_claim: false,
};
const FENCE_FREE_REVALIDATED: Protocol = Protocol {
    fast_pop: true,
    revalidate_after_claim: true,
};

#[test]
fn fenced_pop_takes_each_item_exactly_once() {
    run_pop_against_steals(FENCED, 2, 1, None);
}

#[test]
#[should_panic(expected = "an item was taken twice")]
fn fence_free_pop_without_revalidation_double_takes() {
    run_pop_against_steals(FENCE_FREE, 2, 1, None);
}

#[test]
fn fence_free_pop_with_revalidation_takes_each_item_exactly_once() {
    run_pop_against_steals(FENCE_FREE_REVALIDATED, 2, 1, None);
}

#[test]
fn fence_free_pop_with_revalidation_survives_two_thieves() {
    // Two thieves multiply the interleavings past exhaustive search, so this
    // case is bounded at two preemptions: evidence for interleavings up to
    // that depth, not a proof. The single-thief cases above are exhaustive.
    run_pop_against_steals(FENCE_FREE_REVALIDATED, 3, 2, Some(2));
}
