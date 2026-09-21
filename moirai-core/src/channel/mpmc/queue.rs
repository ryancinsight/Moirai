use crate::channel::error::CacheAligned;
use std::cell::UnsafeCell;
use std::cmp::Ordering as CmpOrdering;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) struct BoundedMpmcSlot<T> {
    sequence: AtomicUsize,
    value: UnsafeCell<MaybeUninit<T>>,
}

pub(super) struct BoundedMpmcQueue<T> {
    buffer: Box<[BoundedMpmcSlot<T>]>,
    mask: usize,
    capacity: usize,
    logical_capacity: usize,
    enqueue_pos: CacheAligned<AtomicUsize>,
    dequeue_pos: CacheAligned<AtomicUsize>,
}

/// Outcome of a successful [`BoundedMpmcQueue::try_push`].
///
/// A blocked receiver parks only after observing the ring empty, so only a push
/// that takes the ring from empty to non-empty can race that decision, and only
/// that push needs the notifier's Store→Load barrier before the waiter-counter
/// read. A push into an already-occupied ring cannot: whichever receiver next
/// calls [`try_pop`](BoundedMpmcQueue::try_pop) finds an item instead of
/// parking. Reporting the transition lets the notifier fence the former and
/// skip both the fence and the counter read on the latter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PushOutcome {
    /// The ring held no items when the slot was claimed: this push took it from
    /// empty to non-empty.
    BecameNonEmpty,
    /// The ring already held items, so no receiver can have parked on this
    /// push's account.
    AlreadyNonEmpty,
}

impl<T> BoundedMpmcQueue<T> {
    pub(super) fn new(requested_capacity: usize) -> Self {
        let logical_capacity = requested_capacity.max(1);
        let capacity = logical_capacity.next_power_of_two().max(2);
        let buffer = (0..capacity)
            .map(|index| BoundedMpmcSlot {
                sequence: AtomicUsize::new(index),
                value: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            buffer,
            mask: capacity - 1,
            capacity,
            logical_capacity,
            enqueue_pos: CacheAligned::new(AtomicUsize::new(0)),
            dequeue_pos: CacheAligned::new(AtomicUsize::new(0)),
        }
    }

    pub(super) fn try_push(&self, value: T) -> std::result::Result<PushOutcome, T> {
        let mut position = self.enqueue_pos.0.load(Ordering::Relaxed);

        loop {
            let slot = &self.buffer[position & self.mask];
            let sequence = slot.sequence.load(Ordering::Acquire);
            #[allow(clippy::cast_possible_wrap)]
            let difference = sequence.wrapping_sub(position) as isize;

            match difference.cmp(&0) {
                CmpOrdering::Equal => {
                    if position.wrapping_sub(self.dequeue_pos.0.load(Ordering::Acquire))
                        >= self.logical_capacity
                    {
                        return Err(value);
                    }

                    match self.enqueue_pos.0.compare_exchange_weak(
                        position,
                        position.wrapping_add(1),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            // Read the consumer cursor *before* publishing this
                            // slot: a consumer can only advance `dequeue_pos`
                            // through already-published slots, so it cannot pass
                            // `position` while this one is unpublished, and
                            // `dequeue_pos == position` is then exactly "the ring
                            // held no items".
                            let outcome = if self.dequeue_pos.0.load(Ordering::Acquire) == position
                            {
                                PushOutcome::BecameNonEmpty
                            } else {
                                PushOutcome::AlreadyNonEmpty
                            };
                            // SAFETY: winning the enqueue-position CAS grants
                            // exclusive right to fill this sequence slot; its
                            // value cell is uninit (fresh or drained) until
                            // this write publishes it.
                            unsafe {
                                (*slot.value.get()).write(value);
                            }
                            slot.sequence
                                .store(position.wrapping_add(1), Ordering::Release);
                            return Ok(outcome);
                        }
                        Err(observed) => position = observed,
                    }
                }
                CmpOrdering::Less => return Err(value),
                CmpOrdering::Greater => {
                    position = self.enqueue_pos.0.load(Ordering::Relaxed);
                }
            }
        }
    }

    pub(super) fn try_pop(&self) -> Option<T> {
        let mut position = self.dequeue_pos.0.load(Ordering::Relaxed);

        loop {
            let slot = &self.buffer[position & self.mask];
            let sequence = slot.sequence.load(Ordering::Acquire);
            #[allow(clippy::cast_possible_wrap)]
            let difference = sequence.wrapping_sub(position.wrapping_add(1)) as isize;

            match difference.cmp(&0) {
                CmpOrdering::Equal => {
                    match self.dequeue_pos.0.compare_exchange_weak(
                        position,
                        position.wrapping_add(1),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            let value = unsafe { (*slot.value.get()).assume_init_read() };
                            slot.sequence
                                .store(position.wrapping_add(self.capacity), Ordering::Release);
                            return Some(value);
                        }
                        Err(observed) => position = observed,
                    }
                }
                CmpOrdering::Less => return None,
                CmpOrdering::Greater => {
                    position = self.dequeue_pos.0.load(Ordering::Relaxed);
                }
            }
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.enqueue_pos.0.load(Ordering::Acquire) == self.dequeue_pos.0.load(Ordering::Acquire)
    }

    pub(super) fn is_full(&self) -> bool {
        self.enqueue_pos
            .0
            .load(Ordering::Acquire)
            .wrapping_sub(self.dequeue_pos.0.load(Ordering::Acquire))
            >= self.logical_capacity
    }

    pub(super) fn logical_capacity(&self) -> usize {
        self.logical_capacity
    }
}

impl<T> Drop for BoundedMpmcQueue<T> {
    fn drop(&mut self) {
        let dequeue_pos = *self.dequeue_pos.0.get_mut();
        let enqueue_pos = *self.enqueue_pos.0.get_mut();
        let len = enqueue_pos.wrapping_sub(dequeue_pos);

        for i in 0..len {
            let pos = dequeue_pos.wrapping_add(i);
            let slot = &mut self.buffer[pos & self.mask];
            let sequence = *slot.sequence.get_mut();
            if sequence == pos.wrapping_add(1) {
                // SAFETY: exclusive `&mut self` in drop; the published
                // sequence marks the cell initialized, and drain order visits
                // each published slot once.
                unsafe {
                    (*slot.value.get()).assume_init_drop();
                }
            }
        }
    }
}

// SAFETY: values move between threads through sequence-gated slots, so
// `T: Send` is required and sufficient; no references escape.
unsafe impl<T: Send> Send for BoundedMpmcQueue<T> {}
// SAFETY: all shared access is arbitrated by the position CAS protocol on
// atomics; stored values are touched only by the thread that owns their
// sequence claim, so `T: Send` suffices.
unsafe impl<T: Send> Sync for BoundedMpmcQueue<T> {}

#[cfg(test)]
mod tests {
    use super::{BoundedMpmcQueue, PushOutcome};

    /// The classification the notifier's fence hangs on: a push reports whether
    /// it took the ring from empty to non-empty.
    ///
    /// Misclassifying an empty→non-empty push as occupied would strand a parked
    /// receiver (its fence and counter read would be skipped), and the reverse
    /// misclassification would only cost an unnecessary fence — so the empty
    /// case, the occupied case, and the return to empty are all pinned.
    #[test]
    fn reports_the_empty_to_non_empty_transition() {
        let queue: BoundedMpmcQueue<u32> = BoundedMpmcQueue::new(4);

        assert_eq!(queue.try_push(1), Ok(PushOutcome::BecameNonEmpty));
        assert_eq!(queue.try_push(2), Ok(PushOutcome::AlreadyNonEmpty));
        assert_eq!(queue.try_push(3), Ok(PushOutcome::AlreadyNonEmpty));

        assert_eq!(queue.try_pop(), Some(1));
        assert_eq!(queue.try_pop(), Some(2));
        assert_eq!(queue.try_push(4), Ok(PushOutcome::AlreadyNonEmpty));

        assert_eq!(queue.try_pop(), Some(3));
        assert_eq!(queue.try_pop(), Some(4));
        // Drained: the next push is the empty→non-empty transition again.
        assert_eq!(queue.try_push(5), Ok(PushOutcome::BecameNonEmpty));
    }

    /// A full ring reports no transition and hands the value back.
    ///
    /// The notifier must not treat a rejected push as a transition: nothing was
    /// published, so there is nothing to wake a receiver for.
    #[test]
    fn a_rejected_push_reports_no_transition() {
        let queue: BoundedMpmcQueue<u32> = BoundedMpmcQueue::new(1);
        assert_eq!(queue.try_push(1), Ok(PushOutcome::BecameNonEmpty));
        assert_eq!(queue.try_push(2), Err(2));
    }
}
