//! Loom model for retained-slot wake publication and claiming.
//!
//! Production wake tokens publish one bit with `fetch_or(Release)`. The stream
//! claims it with an Acquire load followed by `compare_exchange_weak` using
//! AcqRel/Acquire. This model completes generation one, inserts generation two
//! into the same slot, then races the old waker, replacement readiness, and the
//! single stream consumer. A stale wake may cause a permitted spurious poll,
//! but cannot erase replacement readiness, complete either generation twice,
//! or permit replacement before the old future terminates.
//!
//! Parent-task registration is deliberately delegated to
//! `futures::task::AtomicWaker`, whose upstream contract owns the concurrent
//! register/wake edge. The durable Moirai-owned ready bit modeled here remains
//! set even when several child wakes coalesce into one parent scheduling edge.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo nextest run -p moirai-iter --test loom_retained_ready_set --release`

#![cfg(loom)]

use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;

/// One raw owner count using the same `Arc` operations as the slot-waker
/// vtable. The pointer remains valid until this owner is dropped or consumed.
struct RawWakeOwner<T>(*const T);

impl<T> RawWakeOwner<T> {
    fn new(owner: Arc<T>) -> Self {
        Self(Arc::into_raw(owner))
    }

    fn with<R>(&self, operation: impl FnOnce(&T) -> R) -> R {
        // SAFETY: constructing this wrapper consumes one `Arc` strong count,
        // and `self` retains that count for the duration of the borrow.
        operation(unsafe { &*self.0 })
    }
}

impl<T> Clone for RawWakeOwner<T> {
    fn clone(&self) -> Self {
        // SAFETY: `self` owns a strong count, so the allocation remains valid
        // while this operation creates the returned owner's count.
        unsafe { Arc::increment_strong_count(self.0) };
        Self(self.0)
    }
}

impl<T> Drop for RawWakeOwner<T> {
    fn drop(&mut self) {
        // SAFETY: each wrapper owns exactly one strong count, released here.
        unsafe { Arc::decrement_strong_count(self.0) };
    }
}

// SAFETY: moving a raw owner transfers its unique strong-count obligation.
// Access to `T` remains subject to the same `Send + Sync` bounds as `Arc<T>`.
unsafe impl<T: Send + Sync> Send for RawWakeOwner<T> {}

// SAFETY: shared wrapper access exposes only `&T`, and the retained strong
// count keeps the allocation stable for every such borrow.
unsafe impl<T: Send + Sync> Sync for RawWakeOwner<T> {}

struct ModelWakeBlock {
    publications: Arc<AtomicUsize>,
    destructions: Arc<AtomicUsize>,
}

impl ModelWakeBlock {
    fn publish(&self) {
        self.publications.fetch_add(1, Ordering::Release);
    }
}

impl Drop for ModelWakeBlock {
    fn drop(&mut self) {
        self.destructions.fetch_add(1, Ordering::Release);
    }
}

fn take_one(word: &AtomicUsize) -> Option<usize> {
    let mut observed = word.load(Ordering::Acquire);
    while observed != 0 {
        let bit_index = observed.trailing_zeros() as usize;
        let bit = 1usize << bit_index;
        match word.compare_exchange_weak(
            observed,
            observed & !bit,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return Some(bit_index),
            Err(current) => observed = current,
        }
    }
    None
}

fn complete_generation(occupied: &AtomicBool, completions: &AtomicUsize, bit: usize) {
    assert!(
        occupied.swap(false, Ordering::AcqRel),
        "a slot generation completed twice"
    );
    let prior = completions.fetch_or(bit, Ordering::SeqCst);
    assert_eq!(prior & bit, 0, "a slot generation completed twice");
}

#[test]
fn stale_wake_cannot_erase_replacement_readiness() {
    loom::model(|| {
        let ready = Arc::new(AtomicUsize::new(0b1));
        let generation = Arc::new(AtomicUsize::new(1));
        let occupied = Arc::new(AtomicBool::new(true));
        let completions = Arc::new(AtomicUsize::new(0));
        let replacement_ready = Arc::new(AtomicBool::new(false));

        assert_eq!(take_one(&ready), Some(0));
        complete_generation(&occupied, &completions, 0b01);
        assert_eq!(completions.load(Ordering::SeqCst), 0b01);
        assert!(!occupied.load(Ordering::Acquire));

        generation.store(2, Ordering::Release);
        assert!(!occupied.swap(true, Ordering::AcqRel));

        let stale_ready = Arc::clone(&ready);
        let stale = thread::spawn(move || {
            stale_ready.fetch_or(0b1, Ordering::Release);
        });

        let current_ready = Arc::clone(&ready);
        let current_state = Arc::clone(&replacement_ready);
        let current = thread::spawn(move || {
            current_state.store(true, Ordering::Release);
            current_ready.fetch_or(0b1, Ordering::Release);
        });

        let consumer_ready = Arc::clone(&ready);
        let consumer_generation = Arc::clone(&generation);
        let consumer_occupied = Arc::clone(&occupied);
        let consumer_completions = Arc::clone(&completions);
        let consumer_current = Arc::clone(&replacement_ready);
        let consumer = thread::spawn(move || {
            for _ in 0..2 {
                if take_one(&consumer_ready).is_some()
                    && consumer_occupied.load(Ordering::Acquire)
                    && consumer_generation.load(Ordering::Acquire) == 2
                    && consumer_current.load(Ordering::Acquire)
                {
                    complete_generation(&consumer_occupied, &consumer_completions, 0b10);
                }
                thread::yield_now();
            }
        });

        stale.join().unwrap();
        current.join().unwrap();
        consumer.join().unwrap();

        while take_one(&ready).is_some() {
            if occupied.load(Ordering::Acquire) && replacement_ready.load(Ordering::Acquire) {
                complete_generation(&occupied, &completions, 0b10);
            }
        }
        assert_eq!(completions.load(Ordering::SeqCst), 0b11);
        assert!(!occupied.load(Ordering::Acquire));
        assert_eq!(ready.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn raw_wake_owners_reclaim_after_raced_clone_wake_and_cancel() {
    loom::model(|| {
        let publications = Arc::new(AtomicUsize::new(0));
        let destructions = Arc::new(AtomicUsize::new(0));
        let block = Arc::new(ModelWakeBlock {
            publications: Arc::clone(&publications),
            destructions: Arc::clone(&destructions),
        });
        let first = RawWakeOwner::new(Arc::clone(&block));
        let cancelled = first.clone();
        drop(block);

        let wake = thread::spawn(move || {
            let consuming_callback = first.clone();
            drop(first);
            thread::yield_now();
            consuming_callback.with(ModelWakeBlock::publish);
            drop(consuming_callback);
        });
        let cancel = thread::spawn(move || drop(cancelled));

        wake.join().unwrap();
        cancel.join().unwrap();

        assert_eq!(publications.load(Ordering::Acquire), 1);
        assert_eq!(destructions.load(Ordering::Acquire), 1);
    });
}
