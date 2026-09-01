//! Retained bounded storage for in-flight futures.

use core::future::Future;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::ptr;
use core::task::{Context, Poll};

use futures::stream::Stream;
use futures::task::waker_ref;
use std::sync::Arc;

mod wake;

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

/// Pinned contiguous storage reused for every bounded batch.
struct RetainedSlots<Fut> {
    slots: Pin<Box<[FutureSlot<Fut>]>>,
    ready: Arc<ReadySet>,
    wake_tokens: Box<[Arc<WakeToken>]>,
    ready_cursor: usize,
}

impl<Fut> RetainedSlots<Fut> {
    fn new(limit: usize) -> Self {
        let ready = ReadySet::new(limit);
        let slots = core::iter::repeat_with(FutureSlot::empty)
            .take(limit)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            slots: Box::into_pin(slots),
            wake_tokens: (0..limit)
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

    fn register(&self, context: &Context<'_>) {
        self.ready.register(context.waker());
    }

    fn take_ready(&mut self) -> Option<usize> {
        self.ready.take_one(&mut self.ready_cursor)
    }
}

impl<Fut> RetainedSlots<Fut>
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

/// Ordered bounded stream whose future slab is retained and refilled.
pub(crate) struct RetainedBuffered<S>
where
    S: Stream,
    S::Item: Future,
{
    source: Pin<Box<S>>,
    slots: RetainedSlots<S::Item>,
    outputs: Box<[Option<<S::Item as Future>::Output>]>,
    next_insert: usize,
    next_yield: usize,
    buffered: usize,
    source_done: bool,
}

/// Buffer `source` in input order with storage proportional to `limit`.
pub(crate) fn retained_buffered<S>(source: S, limit: usize) -> RetainedBuffered<S>
where
    S: Stream,
    S::Item: Future,
{
    let limit = limit.max(1);
    let outputs = core::iter::repeat_with(|| None)
        .take(limit)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    RetainedBuffered {
        source: Box::pin(source),
        slots: RetainedSlots::new(limit),
        outputs,
        next_insert: 0,
        next_yield: 0,
        buffered: 0,
        source_done: false,
    }
}

impl<S> RetainedBuffered<S>
where
    S: Stream,
    S::Item: Future,
{
    fn fill(&mut self, context: &mut Context<'_>) {
        if self.source_done || self.buffered == self.slots.len() {
            return;
        }

        for slot_index in 0..self.slots.len() {
            if self.buffered == self.slots.len() || !self.slots.is_empty(slot_index) {
                continue;
            }
            match self.source.as_mut().poll_next(context) {
                Poll::Ready(Some(future)) => {
                    let output_index = self.next_insert;
                    self.next_insert = (self.next_insert + 1) % self.slots.len();
                    self.buffered += 1;
                    self.slots.insert(slot_index, future, output_index);
                }
                Poll::Ready(None) => {
                    self.source_done = true;
                    break;
                }
                Poll::Pending => break,
            }
        }
    }

    fn take_next_output(&mut self) -> Option<<S::Item as Future>::Output> {
        let output = self.outputs.get_mut(self.next_yield)?.take()?;
        self.next_yield = (self.next_yield + 1) % self.outputs.len();
        self.buffered -= 1;
        Some(output)
    }
}

impl<S> Stream for RetainedBuffered<S>
where
    S: Stream,
    S::Item: Future,
{
    type Item = <S::Item as Future>::Output;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // `source` and each future have independent pinned ownership. The
        // outer value contains no self-reference and may therefore be moved.
        let this = self.get_mut();
        this.slots.register(context);
        this.fill(context);

        if let Some(output) = this.take_next_output() {
            return Poll::Ready(Some(output));
        }

        for _ in 0..this.slots.len() {
            let Some(slot_index) = this.slots.take_ready() else {
                break;
            };
            if this.slots.is_empty(slot_index) {
                continue;
            }
            if let Poll::Ready((output_index, output)) = this.slots.poll(slot_index) {
                let destination = this
                    .outputs
                    .get_mut(output_index)
                    .expect("invariant: retained output index is in bounds");
                debug_assert!(destination.is_none(), "retained output slot must be empty");
                *destination = Some(output);
            }
        }

        if let Some(output) = this.take_next_output() {
            Poll::Ready(Some(output))
        } else if this.source_done && this.buffered == 0 {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }
}

/// Completion-order bounded stream used only when output order is irrelevant.
pub(crate) struct RetainedUnordered<S>
where
    S: Stream,
    S::Item: Future,
{
    source: Pin<Box<S>>,
    slots: RetainedSlots<S::Item>,
    active: usize,
    source_done: bool,
}

/// Buffer `source` without retaining a task node per item.
pub(crate) fn retained_unordered<S>(source: S, limit: usize) -> RetainedUnordered<S>
where
    S: Stream,
    S::Item: Future,
{
    RetainedUnordered {
        source: Box::pin(source),
        slots: RetainedSlots::new(limit.max(1)),
        active: 0,
        source_done: false,
    }
}

impl<S> RetainedUnordered<S>
where
    S: Stream,
    S::Item: Future,
{
    fn fill(&mut self, context: &mut Context<'_>) {
        if self.source_done || self.active == self.slots.len() {
            return;
        }

        for slot_index in 0..self.slots.len() {
            if self.active == self.slots.len() || !self.slots.is_empty(slot_index) {
                continue;
            }
            match self.source.as_mut().poll_next(context) {
                Poll::Ready(Some(future)) => {
                    self.active += 1;
                    self.slots.insert(slot_index, future, 0);
                }
                Poll::Ready(None) => {
                    self.source_done = true;
                    break;
                }
                Poll::Pending => break,
            }
        }
    }
}

impl<S> Stream for RetainedUnordered<S>
where
    S: Stream,
    S::Item: Future,
{
    type Item = <S::Item as Future>::Output;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // `source` and each future have independent pinned ownership. The
        // outer value contains no self-reference and may therefore be moved.
        let this = self.get_mut();
        this.slots.register(context);
        this.fill(context);

        for _ in 0..this.slots.len() {
            let Some(slot_index) = this.slots.take_ready() else {
                break;
            };
            if this.slots.is_empty(slot_index) {
                continue;
            }
            if let Poll::Ready((_, output)) = this.slots.poll(slot_index) {
                this.active -= 1;
                return Poll::Ready(Some(output));
            }
        }

        if this.source_done && this.active == 0 {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests;
