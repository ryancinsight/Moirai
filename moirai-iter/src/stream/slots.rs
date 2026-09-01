//! Retained bounded storage for in-flight futures.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use futures::stream::Stream;
use std::sync::Arc;

mod cell;
mod ordered;
mod unordered;
mod wake;

pub(crate) use ordered::retained_buffered;
pub(crate) use unordered::retained_unordered;

use wake::WakeBlock;

use cell::FutureSlot;

const VACANT_END: usize = usize::MAX;
const ORDER_END: usize = usize::MAX;
// One-slot geometric growth needs at most `usize::BITS + 1` blocks, including
// the final truncated block below `usize::MAX`; two words cover that bound.
const VACANT_BLOCK_WORDS: usize = 2;
const _: () = assert!(
    VACANT_BLOCK_WORDS * usize::BITS as usize > usize::BITS as usize,
    "vacancy bitset must cover every geometric slot block",
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SlotKey {
    block: usize,
    slot: usize,
    global: usize,
}

/// One independently pinned block in a lazily growing slot set.
struct SlotBlock<Fut> {
    slots: Pin<Box<[FutureSlot<Fut>]>>,
    wake: Arc<WakeBlock>,
    ready_cursor: usize,
    vacant_head: usize,
}

impl<Fut> SlotBlock<Fut> {
    fn new_root(len: usize) -> Self {
        Self::new(len, WakeBlock::new_root(len))
    }

    fn new_child(len: usize, root: Arc<WakeBlock>) -> Self {
        Self::new(len, WakeBlock::new_child(len, root))
    }

    fn new(len: usize, wake: Arc<WakeBlock>) -> Self {
        let slots = (0..len)
            .map(|index| {
                let next = if index + 1 == len {
                    VACANT_END
                } else {
                    index + 1
                };
                FutureSlot::empty(next)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            slots: Box::into_pin(slots),
            wake,
            ready_cursor: 0,
            vacant_head: 0,
        }
    }

    fn is_pollable(&self, index: usize) -> bool {
        self.slots
            .get(index)
            .expect("invariant: retained slot index is in bounds")
            .is_pollable()
    }

    fn slot_mut(
        slots: &mut Pin<Box<[FutureSlot<Fut>]>>,
        index: usize,
    ) -> Pin<&mut FutureSlot<Fut>> {
        // SAFETY: `slots` owns a pinned boxed slice and never exposes an
        // unpinned mutable reference. Selecting one element does not move it.
        let slots = unsafe { slots.as_mut().get_unchecked_mut() };
        let slot = slots
            .get_mut(index)
            .expect("invariant: retained slot index is in bounds");
        // SAFETY: the selected element remains inside the pinned boxed slice
        // for the returned borrow.
        unsafe { Pin::new_unchecked(slot) }
    }

    fn insert(&mut self, index: usize, future: Fut) {
        Self::slot_mut(&mut self.slots, index).insert(future);
        self.wake.set(index);
    }

    fn set_order_next(&mut self, index: usize, next: usize) {
        Self::slot_mut(&mut self.slots, index).set_order_next(next);
    }

    fn order_next(&self, index: usize) -> usize {
        self.slots
            .get(index)
            .expect("invariant: retained slot index is in bounds")
            .order_next()
    }

    fn mark_completed(&mut self, index: usize) {
        Self::slot_mut(&mut self.slots, index).mark_completed();
    }

    fn take_completed_next(&mut self, index: usize) -> Option<usize> {
        Self::slot_mut(&mut self.slots, index).take_completed_next()
    }

    fn take_ready(&mut self) -> Option<usize> {
        self.wake.take_one(&mut self.ready_cursor)
    }

    fn take_vacant(&mut self) -> Option<usize> {
        if self.vacant_head == VACANT_END {
            return None;
        }
        let index = self.vacant_head;
        self.vacant_head = Self::slot_mut(&mut self.slots, index).take_vacant_next();
        Some(index)
    }

    fn return_vacant(&mut self, index: usize) {
        let next = self.vacant_head;
        Self::slot_mut(&mut self.slots, index).return_to_vacant(next);
        self.vacant_head = index;
    }

    const fn has_vacant(&self) -> bool {
        self.vacant_head != VACANT_END
    }
}

impl<Fut> SlotBlock<Fut>
where
    Fut: Future,
{
    fn poll(&mut self, index: usize) -> Poll<Fut::Output> {
        let waker = WakeBlock::waker(&self.wake, index);
        let mut context = Context::from_waker(&waker);
        Self::slot_mut(&mut self.slots, index).poll(&mut context)
    }
}

/// Lazily segmented pinned storage reused for every bounded batch.
///
/// Each block remains at one address even when the block directory grows.
/// Exact-size sources allocate at most their reachable concurrency on first
/// admission. Streams without an upper bound grow geometrically only after
/// yielding another future, so a large configured ceiling is not an eager
/// reservation.
struct RetainedSlots<Fut> {
    first: Option<SlotBlock<Fut>>,
    overflow: Vec<SlotBlock<Fut>>,
    root: Option<Arc<WakeBlock>>,
    capacity: usize,
    limit: usize,
    initial_block_len: usize,
    vacant_blocks: [usize; VACANT_BLOCK_WORDS],
    ready_block_cursor: usize,
    #[cfg(test)]
    vacancy_word_probes: usize,
    #[cfg(test)]
    vacancy_head_probes: usize,
}

impl<Fut> RetainedSlots<Fut> {
    fn new(limit: usize, initial_block_len: usize, eager: bool) -> Self {
        debug_assert!(limit > 0, "retained future limit must be positive");
        debug_assert!(initial_block_len > 0, "initial slot block must be positive");
        debug_assert!(
            initial_block_len <= limit,
            "initial slot block cannot exceed the retained limit"
        );
        let mut slots = Self {
            first: None,
            overflow: Vec::new(),
            root: None,
            capacity: 0,
            limit,
            initial_block_len,
            vacant_blocks: [0; VACANT_BLOCK_WORDS],
            ready_block_cursor: 0,
            #[cfg(test)]
            vacancy_word_probes: 0,
            #[cfg(test)]
            vacancy_head_probes: 0,
        };
        if eager {
            slots.grow();
        }
        slots
    }

    fn limit(&self) -> usize {
        self.limit
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn register(&self, context: &Context<'_>) {
        if let Some(root) = &self.root {
            // `AtomicWaker` owns the register-versus-wake race. The Moirai
            // ready bit remains the durable event when concurrent wakes
            // coalesce into one parent scheduling edge.
            root.register(context.waker());
        }
    }

    fn block_count(&self) -> usize {
        usize::from(self.first.is_some()) + self.overflow.len()
    }

    fn block(&self, index: usize) -> &SlotBlock<Fut> {
        if index == 0 {
            self.first
                .as_ref()
                .expect("invariant: retained first block exists")
        } else {
            self.overflow
                .get(index - 1)
                .expect("invariant: retained overflow block index is in bounds")
        }
    }

    fn block_mut(&mut self, index: usize) -> &mut SlotBlock<Fut> {
        if index == 0 {
            self.first
                .as_mut()
                .expect("invariant: retained first block exists")
        } else {
            self.overflow
                .get_mut(index - 1)
                .expect("invariant: retained overflow block index is in bounds")
        }
    }

    fn block_start(&self, block: usize) -> usize {
        if block == 0 {
            return 0;
        }
        self.initial_block_len
            .checked_shl(
                u32::try_from(block - 1)
                    .expect("invariant: geometric block index fits a shift count"),
            )
            .expect("invariant: an existing geometric block start fits usize")
    }

    fn key_from_global(&self, global: usize) -> SlotKey {
        debug_assert!(global < self.capacity);
        let block = if global < self.initial_block_len {
            0
        } else {
            let quotient = global / self.initial_block_len;
            usize::try_from(quotient.ilog2())
                .expect("invariant: geometric block logarithm fits usize")
                + 1
        };
        let slot = global - self.block_start(block);
        debug_assert!(slot < self.block(block).slots.len());
        SlotKey {
            block,
            slot,
            global,
        }
    }

    fn grow(&mut self) {
        debug_assert!(self.capacity < self.limit, "retained slot limit is full");
        let remaining = self.limit - self.capacity;
        let block_len = if self.capacity == 0 {
            self.initial_block_len
        } else {
            self.capacity
        }
        .min(remaining);
        let block_index = self.block_count();
        if let Some(root) = &self.root {
            self.overflow
                .push(SlotBlock::new_child(block_len, Arc::clone(root)));
        } else {
            let block = SlotBlock::new_root(block_len);
            self.root = Some(Arc::clone(&block.wake));
            self.first = Some(block);
        }
        self.capacity += block_len;
        self.mark_block_vacant(block_index);
    }

    fn mark_block_vacant(&mut self, block: usize) {
        let word = block / usize::BITS as usize;
        let bit = block % usize::BITS as usize;
        let vacant_word = self
            .vacant_blocks
            .get_mut(word)
            .expect("invariant: geometric slot growth fits the vacancy bitset");
        *vacant_word |= 1usize << bit;
    }

    fn clear_block_vacant(&mut self, block: usize) {
        let word = block / usize::BITS as usize;
        let bit = block % usize::BITS as usize;
        let vacant_word = self
            .vacant_blocks
            .get_mut(word)
            .expect("invariant: geometric slot growth fits the vacancy bitset");
        *vacant_word &= !(1usize << bit);
    }

    fn first_vacant_block(&mut self) -> Option<usize> {
        for word_index in 0..VACANT_BLOCK_WORDS {
            #[cfg(test)]
            {
                self.vacancy_word_probes += 1;
            }
            let word = self.vacant_blocks[word_index];
            if word != 0 {
                return Some(word_index * usize::BITS as usize + word.trailing_zeros() as usize);
            }
        }
        None
    }

    fn take_vacant(&mut self) -> Option<SlotKey> {
        let block = self.first_vacant_block()?;
        #[cfg(test)]
        {
            self.vacancy_head_probes += 1;
        }
        let slot = self
            .block_mut(block)
            .take_vacant()
            .expect("invariant: marked vacancy block contains a vacant slot");
        if !self.block(block).has_vacant() {
            self.clear_block_vacant(block);
        }
        Some(SlotKey {
            block,
            slot,
            global: self.block_start(block) + slot,
        })
    }

    fn insert(&mut self, future: Fut) -> SlotKey {
        let key = self.take_vacant().unwrap_or_else(|| {
            self.grow();
            self.take_vacant()
                .expect("invariant: a grown slot block contains a vacancy")
        });
        self.block_mut(key.block).insert(key.slot, future);
        key
    }

    fn is_pollable(&self, key: SlotKey) -> bool {
        self.block(key.block).is_pollable(key.slot)
    }

    fn set_order_next(&mut self, key: SlotKey, next: usize) {
        self.block_mut(key.block).set_order_next(key.slot, next);
    }

    fn order_next(&self, key: SlotKey) -> usize {
        self.block(key.block).order_next(key.slot)
    }

    fn mark_completed(&mut self, key: SlotKey) {
        self.block_mut(key.block).mark_completed(key.slot);
    }

    fn take_completed_next(&mut self, key: SlotKey) -> Option<usize> {
        self.block_mut(key.block).take_completed_next(key.slot)
    }

    fn return_vacant(&mut self, key: SlotKey) {
        self.block_mut(key.block).return_vacant(key.slot);
        self.mark_block_vacant(key.block);
    }

    #[cfg(test)]
    const fn vacancy_probe_counts(&self) -> (usize, usize) {
        (self.vacancy_word_probes, self.vacancy_head_probes)
    }

    fn take_ready(&mut self) -> Option<SlotKey> {
        let block_count = self.block_count();
        if block_count == 0 {
            return None;
        }
        let start = self.ready_block_cursor % block_count;
        for offset in 0..block_count {
            let block = (start + offset) % block_count;
            if let Some(slot) = self.block_mut(block).take_ready() {
                self.ready_block_cursor = (block + 1) % block_count;
                return Some(SlotKey {
                    block,
                    slot,
                    global: self.block_start(block) + slot,
                });
            }
        }
        None
    }
}

impl<Fut> RetainedSlots<Fut>
where
    Fut: Future,
{
    fn poll(&mut self, key: SlotKey) -> Poll<Fut::Output> {
        self.block_mut(key.block).poll(key.slot)
    }
}

fn source_slot_plan<S>(source: &S, configured_limit: usize) -> (usize, usize, bool)
where
    S: Stream,
{
    let configured_limit = configured_limit.max(1);
    let (lower, upper) = source.size_hint();
    let limit = upper.map_or(configured_limit, |bound| configured_limit.min(bound.max(1)));
    let exact = upper == Some(lower);
    let initial_block_len = if exact { limit.min(lower.max(1)) } else { 1 };
    (limit, initial_block_len, exact && lower > 0)
}

#[cfg(test)]
mod tests;
