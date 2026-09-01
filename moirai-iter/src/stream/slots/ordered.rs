//! Ordered retained-future stream execution.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use futures::stream::Stream;

use super::{source_slot_plan, RetainedSlots};

/// Ordered bounded stream whose future slabs are retained and refilled.
pub(crate) struct RetainedBuffered<S>
where
    S: Stream,
    S::Item: Future,
{
    source: Pin<Box<S>>,
    slots: RetainedSlots<S::Item>,
    outputs: Vec<Option<<S::Item as Future>::Output>>,
    next_insert: usize,
    next_yield: usize,
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

/// Buffer `source` in input order with storage proportional to reachable work.
pub(crate) fn retained_buffered<S>(source: S, limit: usize) -> RetainedBuffered<S>
where
    S: Stream,
    S::Item: Future,
{
    let (limit, initial_block_len, eager) = source_slot_plan(&source, limit);
    let slots = RetainedSlots::new(limit, initial_block_len, eager);
    let output_capacity = slots.capacity();
    RetainedBuffered {
        source: Box::pin(source),
        slots,
        outputs: Vec::with_capacity(output_capacity),
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
        if self.source_done || self.buffered == self.slots.limit() {
            return;
        }

        while self.buffered < self.slots.limit() {
            match self.source.as_mut().poll_next(context) {
                Poll::Ready(Some(future)) => {
                    let output_index = self.next_insert;
                    self.next_insert = (self.next_insert + 1) % self.slots.limit();
                    self.buffered += 1;
                    self.slots.insert(future, output_index);
                    if self.outputs.capacity() < self.slots.capacity() {
                        self.outputs
                            .reserve_exact(self.slots.capacity() - self.outputs.capacity());
                    }
                    if output_index == self.outputs.len() {
                        self.outputs.push(None);
                    }
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
        self.next_yield = (self.next_yield + 1) % self.slots.limit();
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
            if this.slots.is_empty(slot) {
                continue;
            }
            if let Poll::Ready((output_index, output)) = this.slots.poll(slot) {
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
