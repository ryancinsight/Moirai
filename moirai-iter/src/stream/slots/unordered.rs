//! Completion-order retained-future stream execution.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use futures::stream::Stream;

use super::{source_slot_plan, RetainedSlots};

/// Completion-order bounded stream used when output order is irrelevant.
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
    let (limit, initial_block_len, eager) = source_slot_plan(&source, limit);
    RetainedUnordered {
        source: Box::pin(source),
        slots: RetainedSlots::new(limit, initial_block_len, eager),
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
        if self.source_done || self.active == self.slots.limit() {
            return;
        }

        while self.active < self.slots.limit() {
            match self.source.as_mut().poll_next(context) {
                Poll::Ready(Some(future)) => {
                    self.active += 1;
                    self.slots.insert(future, 0);
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
        let had_slots = this.slots.capacity() != 0;
        this.slots.register(context);
        this.fill(context);
        if !had_slots {
            this.slots.register(context);
        }

        for _ in 0..this.slots.capacity() {
            let Some(slot) = this.slots.take_ready() else {
                break;
            };
            if this.slots.is_empty(slot) {
                continue;
            }
            if let Poll::Ready((_, output)) = this.slots.poll(slot) {
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
