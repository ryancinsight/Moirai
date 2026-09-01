//! Retained bounded storage for in-flight futures.

use core::future::Future;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::ptr;
use core::task::{Context, Poll};

use futures::stream::Stream;
use futures::task::waker_ref;
use std::sync::Arc;

mod ordered;
mod unordered;
mod wake;

pub(crate) use ordered::retained_buffered;
pub(crate) use unordered::retained_unordered;

use wake::{ReadySet, WakeToken};

/// One stable-address future cell inside a pinned contiguous slab.
///
/// `occupied` is set before a future may be polled and cleared before that
/// future is dropped. This makes cancellation and unwinding drop each inserted
/// future at most once without moving a pinned value.
struct FutureSlot<Fut> {
    storage: MaybeUninit<Fut>,
    occupied: bool,
    output_index: usize,
}

impl<Fut> FutureSlot<Fut> {
    const fn empty() -> Self {
        Self {
            storage: MaybeUninit::uninit(),
            occupied: false,
            output_index: 0,
        }
    }

    const fn is_empty(&self) -> bool {
        !self.occupied
    }

    fn insert(self: Pin<&mut Self>, future: Fut, output_index: usize) {
        // SAFETY: the slot slab is already pinned. This method writes only an
        // empty slot and never moves an occupied future out of its address.
        let this = unsafe { self.get_unchecked_mut() };
        debug_assert!(!this.occupied, "retained future slot must be empty");
        this.storage.write(future);
        this.output_index = output_index;
        this.occupied = true;
    }
}

impl<Fut> FutureSlot<Fut>
where
    Fut: Future,
{
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<(usize, Fut::Output)> {
        // SAFETY: the slot remains pinned for this call. `storage` is
        // initialized exactly when `occupied` is true, and this method never
        // moves the future before dropping it in place after `Ready`.
        let this = unsafe { self.get_unchecked_mut() };
        debug_assert!(this.occupied, "retained future slot must be occupied");
        let future = this.storage.as_mut_ptr();
        // SAFETY: `future` points to an initialized value in a stable pinned
        // slab and remains at that address until it is dropped below.
        let poll = unsafe { Pin::new_unchecked(&mut *future) }.poll(context);
        match poll {
            Poll::Ready(output) => {
                let output_index = this.output_index;
                // Clear first so an unwinding destructor cannot be run twice by
                // `FutureSlot::drop`.
                this.occupied = false;
                // SAFETY: `future` is initialized and has not been moved. The
                // occupancy transition makes this its unique drop.
                unsafe { ptr::drop_in_place(future) };
                Poll::Ready((output_index, output))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<Fut> Drop for FutureSlot<Fut> {
    fn drop(&mut self) {
        if self.occupied {
            self.occupied = false;
            // SAFETY: an occupied slot contains one initialized future. The
            // slab has not moved it, and clearing first prevents a second drop
            // if the future destructor unwinds.
            unsafe { ptr::drop_in_place(self.storage.as_mut_ptr()) };
        }
    }
}

#[derive(Clone, Copy)]
struct SlotKey {
    block: usize,
    slot: usize,
}

/// One independently pinned block in a lazily growing slot set.
struct SlotBlock<Fut> {
    slots: Pin<Box<[FutureSlot<Fut>]>>,
    ready: Arc<ReadySet>,
    wake_tokens: Box<[Arc<WakeToken>]>,
    ready_cursor: usize,
}

impl<Fut> SlotBlock<Fut> {
    fn new_root(len: usize) -> Self {
        Self::new(len, ReadySet::new_root(len))
    }

    fn new_child(len: usize, root: Arc<ReadySet>) -> Self {
        Self::new(len, ReadySet::new_child(len, root))
    }

    fn new(len: usize, ready: Arc<ReadySet>) -> Self {
        let slots = core::iter::repeat_with(FutureSlot::empty)
            .take(len)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            slots: Box::into_pin(slots),
            wake_tokens: (0..len)
                .map(|index| WakeToken::new(index, Arc::clone(&ready)))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            ready,
            ready_cursor: 0,
        }
    }

    fn len(&self) -> usize {
        self.slots.len()
    }

    fn is_empty(&self, index: usize) -> bool {
        self.slots
            .get(index)
            .expect("invariant: retained slot index is in bounds")
            .is_empty()
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

    fn insert(&mut self, index: usize, future: Fut, output_index: usize) {
        Self::slot_mut(&mut self.slots, index).insert(future, output_index);
        self.ready.set(index);
    }

    fn take_ready(&mut self) -> Option<usize> {
        self.ready.take_one(&mut self.ready_cursor)
    }
}

impl<Fut> SlotBlock<Fut>
where
    Fut: Future,
{
    fn poll(&mut self, index: usize) -> Poll<(usize, Fut::Output)> {
        let token = self
            .wake_tokens
            .get(index)
            .expect("invariant: retained wake token index is in bounds");
        let waker = waker_ref(token);
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
    root: Option<Arc<ReadySet>>,
    capacity: usize,
    limit: usize,
    initial_block_len: usize,
    vacant_block_cursor: usize,
    vacant_slot_cursor: usize,
    ready_block_cursor: usize,
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
            vacant_block_cursor: 0,
            vacant_slot_cursor: 0,
            ready_block_cursor: 0,
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

    fn grow(&mut self) {
        debug_assert!(self.capacity < self.limit, "retained slot limit is full");
        let remaining = self.limit - self.capacity;
        let block_len = if self.capacity == 0 {
            self.initial_block_len
        } else {
            self.capacity
        }
        .min(remaining);
        if let Some(root) = &self.root {
            self.overflow
                .push(SlotBlock::new_child(block_len, Arc::clone(root)));
        } else {
            let block = SlotBlock::new_root(block_len);
            self.root = Some(Arc::clone(&block.ready));
            self.first = Some(block);
        }
        self.capacity += block_len;
    }

    fn advance_vacant_cursor(&mut self, key: SlotKey) {
        let block_len = self.block(key.block).len();
        if key.slot + 1 == block_len {
            self.vacant_block_cursor = (key.block + 1) % self.block_count();
            self.vacant_slot_cursor = 0;
        } else {
            self.vacant_block_cursor = key.block;
            self.vacant_slot_cursor = key.slot + 1;
        }
    }

    fn empty_slot(&mut self) -> Option<SlotKey> {
        let block_count = self.block_count();
        if block_count == 0 {
            return None;
        }

        let start_block = self.vacant_block_cursor % block_count;
        for block_offset in 0..block_count {
            let block = (start_block + block_offset) % block_count;
            let block_len = self.block(block).len();
            let start_slot = if block == start_block {
                self.vacant_slot_cursor.min(block_len - 1)
            } else {
                0
            };
            for slot_offset in 0..block_len {
                let slot = (start_slot + slot_offset) % block_len;
                if self.block(block).is_empty(slot) {
                    let key = SlotKey { block, slot };
                    self.advance_vacant_cursor(key);
                    return Some(key);
                }
            }
        }
        None
    }

    fn insert(&mut self, future: Fut, output_index: usize) {
        let key = self.empty_slot().unwrap_or_else(|| {
            self.grow();
            let key = SlotKey {
                block: self.block_count() - 1,
                slot: 0,
            };
            self.advance_vacant_cursor(key);
            key
        });
        self.block_mut(key.block)
            .insert(key.slot, future, output_index);
    }

    fn is_empty(&self, key: SlotKey) -> bool {
        self.block(key.block).is_empty(key.slot)
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
                return Some(SlotKey { block, slot });
            }
        }
        None
    }
}

impl<Fut> RetainedSlots<Fut>
where
    Fut: Future,
{
    fn poll(&mut self, key: SlotKey) -> Poll<(usize, Fut::Output)> {
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
