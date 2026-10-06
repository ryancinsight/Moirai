//! Results of steal attempts: the single-item outcome and the batch container.

use std::mem::MaybeUninit;

pub(super) const MAX_BATCH_STEAL: usize = 16;

/// Outcome of a steal attempt against another worker's deque.
///
/// [`Empty`](Self::Empty) and [`Retry`](Self::Retry) are deliberately
/// distinct: the first is a fact about the victim, the second is a fact
/// about this attempt. Collapsing them would make a thief either spin on
/// a genuinely empty deque or abandon a victim that still has work.
#[derive(Debug, Clone, PartialEq)]
pub enum StealResult<T> {
    /// An item was taken from the victim.
    Success(T),
    /// The victim held no work; look elsewhere.
    Empty,
    /// The steal lost a race against the owner or another thief. The
    /// victim may still hold work, so retrying the same deque is
    /// worthwhile.
    Retry,
}

/// Allocation-free ownership container returned by a batch steal.
pub struct StolenBatch<T> {
    pub(super) items: [MaybeUninit<T>; MAX_BATCH_STEAL],
    pub(super) next: usize,
    pub(super) len: usize,
}

impl<T> Iterator for StolenBatch<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.len {
            return None;
        }
        let index = self.next;
        self.next += 1;
        // SAFETY: `[next, len)` is initialized and advancing `next` transfers
        // this slot exactly once.
        Some(unsafe { self.items[index].assume_init_read() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.next;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for StolenBatch<T> {}

impl<T> Drop for StolenBatch<T> {
    fn drop(&mut self) {
        for item in &mut self.items[self.next..self.len] {
            // SAFETY: `[next, len)` is the initialized, unconsumed tail.
            unsafe { item.assume_init_drop() };
        }
    }
}
