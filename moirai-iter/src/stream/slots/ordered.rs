//! Ordered retained-future stream execution.

use core::future::Future;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::task::{Context, Poll};

use futures::stream::Stream;

use super::{source_slot_plan, RetainedSlots, ORDER_END};

/// Ordered bounded stream whose future slabs are retained and refilled.
pub(crate) struct RetainedBuffered<S>
where
    S: Stream,
    S::Item: Future,
{
    source: Pin<Box<S>>,
    slots: RetainedSlots<S::Item>,
    outputs: Vec<MaybeUninit<<S::Item as Future>::Output>>,
    ordered_head: usize,
    ordered_tail: usize,
    buffered: usize,
    source_done: bool,
}

// Moving the stream moves only owning pointers and the output-vector header.
// In-flight futures remain in independently pinned boxes, and completed output
// elements are not exposed through a pinned reference.
impl<S> Unpin for RetainedBuffered<S>
where
    S: Stream,
    S::Item: Future,
{
}

struct CompletedOutputDropGuard<'a, S>
where
    S: Stream,
    S::Item: Future,
{
    stream: &'a mut RetainedBuffered<S>,
}

impl<S> Drop for CompletedOutputDropGuard<'_, S>
where
    S: Stream,
    S::Item: Future,
{
    fn drop(&mut self) {
        while self.stream.drop_next_completed_output() {}
    }
}

/// Buffer `source` in input order with storage proportional to reachable work.
pub(crate) fn retained_buffered<S>(source: S, limit: usize) -> RetainedBuffered<S>
where
    S: Stream,
    S::Item: Future,
{
    let (limit, initial_block_len, eager) = source_slot_plan(&source, limit);
    let slots = RetainedSlots::new(limit, initial_block_len, eager);
    let output_capacity = slots.capacity();
    let mut outputs = Vec::with_capacity(output_capacity);
    outputs.resize_with(output_capacity, MaybeUninit::uninit);
    RetainedBuffered {
        source: Box::pin(source),
        slots,
        outputs,
        ordered_head: ORDER_END,
        ordered_tail: ORDER_END,
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
        if self.source_done || self.buffered == self.slots.limit() {
            return;
        }

        while self.buffered < self.slots.limit() {
            match self.source.as_mut().poll_next(context) {
                Poll::Ready(Some(future)) => {
                    let slot = self.slots.insert(future);
                    if self.outputs.len() < self.slots.capacity() {
                        self.outputs
                            .resize_with(self.slots.capacity(), MaybeUninit::uninit);
                    }
                    if self.ordered_tail == ORDER_END {
                        debug_assert_eq!(self.ordered_head, ORDER_END);
                        self.ordered_head = slot.global;
                    } else {
                        let tail = self.slots.key_from_global(self.ordered_tail);
                        self.slots.set_order_next(tail, slot.global);
                    }
                    self.ordered_tail = slot.global;
                    self.buffered += 1;
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
        if self.ordered_head == ORDER_END {
            return None;
        }
        let slot = self.slots.key_from_global(self.ordered_head);
        let output = self
            .outputs
            .get_mut(slot.global)
            .expect("invariant: retained output slot is in bounds");
        let next = self.slots.take_completed_next(slot)?;
        self.ordered_head = next;
        if next == ORDER_END {
            self.ordered_tail = ORDER_END;
        }
        self.buffered -= 1;
        // SAFETY: `Completed` is published only after this physical output
        // cell is initialized. `take_completed_next` transitioned the slot to
        // `Detached`, making this the unique read before vacancy reuse.
        let output = unsafe { output.assume_init_read() };
        self.slots.return_vacant(slot);
        Some(output)
    }

    fn drop_next_completed_output(&mut self) -> bool {
        if self.ordered_head == ORDER_END {
            return false;
        }
        let global = self.ordered_head;
        let slot = self.slots.key_from_global(global);
        let next = self.slots.order_next(slot);
        self.ordered_head = next;
        if next == ORDER_END {
            self.ordered_tail = ORDER_END;
        }
        if self.slots.take_completed_next(slot).is_some() {
            let output = self
                .outputs
                .get_mut(global)
                .expect("invariant: completed output slot is in bounds");
            // SAFETY: `Completed` proves this cell was initialized, and the
            // transition to `Detached` plus the advanced head makes this its
            // unique drop even if the output destructor unwinds.
            unsafe { output.assume_init_drop() };
        }
        true
    }

    #[cfg(test)]
    pub(super) fn storage_capacities(&self) -> (usize, usize) {
        (self.slots.capacity(), self.outputs.len())
    }
}

impl<S> Drop for RetainedBuffered<S>
where
    S: Stream,
    S::Item: Future,
{
    fn drop(&mut self) {
        let guard = CompletedOutputDropGuard { stream: self };
        while guard.stream.drop_next_completed_output() {}
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
        let had_slots = this.slots.capacity() != 0;
        this.slots.register(context);
        this.fill(context);
        if !had_slots {
            this.slots.register(context);
        }

        if let Some(output) = this.take_next_output() {
            return Poll::Ready(Some(output));
        }

        for _ in 0..this.slots.capacity() {
            let Some(slot) = this.slots.take_ready() else {
                break;
            };
            if !this.slots.is_pollable(slot) {
                continue;
            }
            let destination = this
                .outputs
                .get_mut(slot.global)
                .expect("invariant: retained output index is in bounds");
            if let Poll::Ready(output) = this.slots.poll(slot) {
                destination.write(output);
                this.slots.mark_completed(slot);
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
