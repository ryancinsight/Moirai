//! Allocation-stable wake routing for retained future slots.

use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use futures::task::{ArcWake, AtomicWaker};

const WORD_BITS: usize = usize::BITS as usize;

/// A bounded ready bitset and the parent stream waker.
pub(super) struct ReadySet {
    words: Box<[AtomicUsize]>,
    slot_count: usize,
    parent: AtomicWaker,
}

impl ReadySet {
    pub(super) fn new(slot_count: usize) -> Arc<Self> {
        let word_count = slot_count.div_ceil(WORD_BITS);
        Arc::new(Self {
            words: core::iter::repeat_with(|| AtomicUsize::new(0))
                .take(word_count)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            slot_count,
            parent: AtomicWaker::new(),
        })
    }

    pub(super) fn register(&self, waker: &core::task::Waker) {
        self.parent.register(waker);
    }

    pub(super) fn mark_ready(&self, index: usize) {
        self.set(index);
        self.parent.wake();
    }

    pub(super) fn set(&self, index: usize) {
        let word_index = index / WORD_BITS;
        let bit = 1usize << (index % WORD_BITS);
        self.words
            .get(word_index)
            .expect("invariant: retained wake index is in bounds")
            // Release publishes the wake before the parent observes the bit.
            .fetch_or(bit, Ordering::Release);
    }

    pub(super) fn take_one(&self, cursor: &mut usize) -> Option<usize> {
        let start = *cursor % self.slot_count;
        let index = self
            .take_from_range(start, self.slot_count)
            .or_else(|| self.take_from_range(0, start))?;
        *cursor = (index + 1) % self.slot_count;
        Some(index)
    }

    fn take_from_range(&self, start: usize, end: usize) -> Option<usize> {
        if start == end {
            return None;
        }

        let first_word = start / WORD_BITS;
        let last_word = (end - 1) / WORD_BITS;
        for word_index in first_word..=last_word {
            let word = self
                .words
                .get(word_index)
                .expect("invariant: retained wake word is in bounds");
            let lower_bit = if word_index == first_word {
                start % WORD_BITS
            } else {
                0
            };
            let upper_bit = if word_index == last_word {
                (end - 1) % WORD_BITS + 1
            } else {
                WORD_BITS
            };
            let lower_mask = usize::MAX << lower_bit;
            let upper_mask = if upper_bit == WORD_BITS {
                usize::MAX
            } else {
                (1usize << upper_bit) - 1
            };
            let range_mask = lower_mask & upper_mask;

            let mut observed = word.load(Ordering::Acquire);
            while observed & range_mask != 0 {
                let bit_index = (observed & range_mask).trailing_zeros() as usize;
                let bit = 1usize << bit_index;
                match word.compare_exchange_weak(
                    observed,
                    observed & !bit,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => return Some(word_index * WORD_BITS + bit_index),
                    Err(current) => observed = current,
                }
            }
        }
        None
    }
}

/// Stable wake identity for one future slot.
pub(super) struct WakeToken {
    index: usize,
    ready: Arc<ReadySet>,
}

impl WakeToken {
    pub(super) fn new(index: usize, ready: Arc<ReadySet>) -> Arc<Self> {
        Arc::new(Self { index, ready })
    }
}

impl ArcWake for WakeToken {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.ready.mark_ready(arc_self.index);
    }
}
