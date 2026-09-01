//! Allocation-stable wake routing for retained future slots.

use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::task::{RawWaker, RawWakerVTable, Waker};
use std::sync::{Arc, OnceLock};

use futures::task::AtomicWaker;

const WORD_BITS: usize = usize::BITS as usize;

/// A bounded ready bitset, stable slot tokens, and the parent stream waker.
pub(super) struct WakeBlock {
    words: Box<[AtomicUsize]>,
    slot_count: usize,
    parent: ParentWaker,
    tokens: OnceLock<Box<[WakeToken]>>,
}

enum ParentWaker {
    Root(AtomicWaker),
    Shared(Arc<WakeBlock>),
}

impl WakeBlock {
    pub(super) fn new_root(slot_count: usize) -> Arc<Self> {
        Self::new(slot_count, ParentWaker::Root(AtomicWaker::new()))
    }

    pub(super) fn new_child(slot_count: usize, root: Arc<Self>) -> Arc<Self> {
        Self::new(slot_count, ParentWaker::Shared(root))
    }

    fn new(slot_count: usize, parent: ParentWaker) -> Arc<Self> {
        let word_count = slot_count.div_ceil(WORD_BITS);
        let block = Arc::new(Self {
            words: core::iter::repeat_with(|| AtomicUsize::new(0))
                .take(word_count)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            slot_count,
            parent,
            tokens: OnceLock::new(),
        });
        let mut tokens = Box::<[WakeToken]>::new_uninit_slice(slot_count);
        let raw_owner = Arc::into_raw(Arc::clone(&block));
        let owner = NonNull::new(raw_owner.cast_mut())
            .expect("invariant: Arc allocation pointer is non-null");
        for (index, token) in tokens.iter_mut().enumerate() {
            token.write(WakeToken { index, owner });
        }
        // SAFETY: the loop writes every token exactly once before this slice is
        // exposed, and `enumerate` visits the complete allocated slice.
        let tokens = unsafe { tokens.assume_init() };
        let initialized = block.tokens.set(tokens).is_ok();
        // SAFETY: `raw_owner` was produced by the matching `Arc::into_raw`
        // above. Token construction is complete, so this temporary count is no
        // longer needed.
        unsafe { drop(Arc::from_raw(raw_owner)) };
        assert!(
            initialized,
            "invariant: wake tokens initialize exactly once"
        );
        block
    }

    pub(super) fn waker(block: &Arc<Self>, index: usize) -> Waker {
        let token = block
            .tokens
            .get()
            .expect("invariant: retained wake tokens are initialized")
            .get(index)
            .expect("invariant: retained wake token index is in bounds");
        debug_assert!(core::ptr::eq(
            token.owner.as_ptr().cast_const(),
            Arc::as_ptr(block)
        ));
        let raw_owner = Arc::into_raw(Arc::clone(block));
        debug_assert!(core::ptr::eq(token.owner.as_ptr().cast_const(), raw_owner));
        let data = NonNull::from(token).cast::<()>().as_ptr().cast_const();
        let raw = RawWaker::new(data, &WAKE_VTABLE);
        // SAFETY: `raw` carries the strong owner count created by
        // `Arc::into_raw`, and `WAKE_VTABLE` preserves that ownership contract
        // across clone, wake, and drop.
        unsafe { Waker::from_raw(raw) }
    }

    pub(super) fn register(&self, waker: &core::task::Waker) {
        match &self.parent {
            ParentWaker::Root(parent) => parent.register(waker),
            ParentWaker::Shared(root) => root.register(waker),
        }
    }

    pub(super) fn mark_ready(&self, index: usize) {
        self.set(index);
        match &self.parent {
            ParentWaker::Root(parent) => parent.wake(),
            ParentWaker::Shared(root) => root.wake_parent(),
        }
    }

    fn wake_parent(&self) {
        match &self.parent {
            ParentWaker::Root(parent) => parent.wake(),
            ParentWaker::Shared(root) => root.wake_parent(),
        }
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

/// Stable wake identity stored inline in one [`WakeBlock`] token allocation.
struct WakeToken {
    index: usize,
    owner: NonNull<WakeBlock>,
}

// SAFETY: `owner` comes from `Arc::into_raw` on this stable allocation. Every
// exposed raw waker owns a strong count on that same allocation, so the token
// cannot outlive or race destruction of its owner.
unsafe impl Send for WakeToken {}

// SAFETY: `index` and `owner` are immutable after unique construction. Shared
// access reaches only atomic ready words and `AtomicWaker` synchronization.
unsafe impl Sync for WakeToken {}

struct OwnedWakeBlock(NonNull<WakeBlock>);

impl Drop for OwnedWakeBlock {
    fn drop(&mut self) {
        // SAFETY: this guard represents exactly one strong count acquired for
        // a consuming raw-waker callback.
        unsafe { Arc::decrement_strong_count(self.0.as_ptr()) };
    }
}

unsafe fn clone_waker(data: *const ()) -> RawWaker {
    // SAFETY: the callback receives the stable token pointer from `waker`; the
    // source raw waker keeps its containing block alive throughout this call.
    let token = unsafe { &*data.cast::<WakeToken>() };
    // SAFETY: the source raw waker proves the owner's strong count is nonzero.
    // The returned raw waker owns this additional count.
    unsafe { Arc::increment_strong_count(token.owner.as_ptr()) };
    RawWaker::new(data, &WAKE_VTABLE)
}

unsafe fn wake(data: *const ()) {
    // SAFETY: the callback receives the stable token pointer from `waker` and
    // owns one strong count on its containing block.
    let token = unsafe { &*data.cast::<WakeToken>() };
    let owner = token.owner;
    let index = token.index;
    let _owned = OwnedWakeBlock(owner);
    // SAFETY: `_owned` keeps the allocation alive through readiness
    // publication and any parent wake it triggers.
    unsafe { owner.as_ref() }.mark_ready(index);
}

unsafe fn wake_by_ref(data: *const ()) {
    // SAFETY: the borrowed raw waker keeps the containing block alive for this
    // callback and its token pointer is stable within that block.
    let token = unsafe { &*data.cast::<WakeToken>() };
    // SAFETY: the caller's raw waker retains ownership through this borrowed
    // callback, so the block remains alive through parent notification.
    unsafe { token.owner.as_ref() }.mark_ready(token.index);
}

unsafe fn drop_waker(data: *const ()) {
    // SAFETY: the callback receives the stable token pointer from `waker`; its
    // raw waker owns exactly one strong count on the containing block.
    let token = unsafe { &*data.cast::<WakeToken>() };
    // SAFETY: consuming this raw waker releases its one owned strong count and
    // does not access the token afterward.
    unsafe { Arc::decrement_strong_count(token.owner.as_ptr()) };
}

const WAKE_VTABLE: RawWakerVTable = RawWakerVTable::new(clone_waker, wake, wake_by_ref, drop_waker);
