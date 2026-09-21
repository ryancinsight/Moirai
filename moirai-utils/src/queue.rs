//! Lock-free queues for high-performance data structures.
//!
//! This module provides an efficient bounded multi-producer multi-consumer
//! queue built on per-slot sequence numbers (the Vyukov algorithm). It is the
//! workspace's only implementation of that algorithm: the scheduler injector,
//! both executors' run queues, and `moirai-core`'s bounded MPMC channel all run
//! on it.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use core::cell::UnsafeCell;
use core::cmp::Ordering as CmpOrdering;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::cache::CacheAligned;

#[cfg(feature = "std")]
use std::boxed::Box;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

/// Default capacity for [`LockFreeQueue`]. Large enough to avoid backpressure
/// under normal scheduling load while bounding memory under adversarial
/// producer rates per the bounded-resource policy.
const DEFAULT_QUEUE_CAPACITY: usize = 65536;

/// Outcome of a successful [`LockFreeQueue::try_enqueue_outcome`].
///
/// A blocked receiver parks only after observing the ring empty, so only a push
/// that takes the ring from empty to non-empty can race that decision, and only
/// that push needs the notifier's Store→Load barrier before the waiter-counter
/// read. A push into an already-occupied ring cannot: whichever receiver next
/// calls [`try_dequeue`](LockFreeQueue::try_dequeue) finds an item instead of
/// parking. Reporting the transition lets the notifier fence the former and
/// skip both the fence and the counter read on the latter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnqueueOutcome {
    /// The ring held no items when the slot was claimed: this push took it from
    /// empty to non-empty.
    BecameNonEmpty,
    /// The ring already held items, so no receiver can have parked on this
    /// push's account.
    AlreadyNonEmpty,
}

/// A single slot in the bounded MPMC queue.
struct Slot<T> {
    /// Monotonic sequence number that distinguishes empty, full, and stale
    /// states without an ABA hazard.
    sequence: AtomicUsize,
    /// The slot's data. It is uninitialized exactly while the sequence number
    /// reports the slot empty, so occupancy needs no separate flag and no
    /// `Option` discriminant: the slot costs one machine word per payload.
    data: UnsafeCell<MaybeUninit<T>>,
}

/// A bounded, genuinely lock-free multi-producer multi-consumer queue.
///
/// This is an array-based MPMC queue using per-slot sequence numbers (the
/// Vyukov algorithm). Producers and consumers operate through independent
/// atomic head/tail cursors and never acquire a mutex or spinlock. The
/// sequence-number protocol eliminates the ABA problem without tagged
/// pointers or epoch-based reclamation: slots are reused in place, so no
/// node allocation or deallocation occurs during enqueue/dequeue.
///
/// # Capacity
///
/// The queue is bounded. [`LockFreeQueue::new`] creates a queue with
/// `DEFAULT_QUEUE_CAPACITY` usable slots. [`LockFreeQueue::with_capacity`]
/// accepts any request of one slot or more: the ring is sized to the next power
/// of two at least two, while [`capacity`](LockFreeQueue::capacity) keeps
/// reporting the request, so a non-power-of-two capacity bounds the queue
/// exactly and only the ring's unused tail is wasted. When the queue is full,
/// [`enqueue`] retries with exponential backoff (preserving the
/// unblocked-sender contract of the previous API), while [`try_enqueue`]
/// returns `Err(item)` immediately for callers that prefer explicit
/// backpressure.
///
/// # Memory safety
///
/// Each slot's `MaybeUninit<T>` is written by the producer — the only writer,
/// between `sequence == pos` and `sequence == pos + 1` — and moved out by the
/// consumer, which is the only reader, between `sequence == pos + 1` and
/// `sequence == pos + ring_len`. The sequence-number protocol therefore
/// guarantees that only one thread ever touches a slot's payload.
///
/// [`enqueue`]: LockFreeQueue::enqueue
/// [`try_enqueue`]: LockFreeQueue::try_enqueue
// No struct-level `repr(align)`: `head`/`tail` are `CacheAligned`, so the
// struct's alignment already equals `DESTRUCTIVE_INTERFERENCE_SIZE` and tracks
// the per-target table in `cache.rs` instead of pinning a second literal here.
pub struct LockFreeQueue<T> {
    buffer: Box<[Slot<T>]>,
    mask: usize,
    /// Slots in the ring: a power of two, at least two, so that a slot's empty
    /// and full generations never alias. Private because the sequence protocol
    /// addresses slots with it; callers reason in `capacity`.
    ring_len: usize,
    /// Usable slots: what `try_enqueue` accepts before reporting full.
    capacity: usize,
    head: CacheAligned<AtomicUsize>,
    tail: CacheAligned<AtomicUsize>,
}

// Safety: The sequence-number protocol ensures that each slot's data is
// accessed by at most one thread at a time: a producer writes between
// sequence == pos and sequence == pos+1; a consumer takes between
// sequence == pos+1 and sequence == pos+ring_len. The head and tail atomics
// are independently advanced via CAS, so no global lock is needed. T: Send
// is sufficient because ownership of the value transfers between threads
// through the slot, never shared concurrently.
unsafe impl<T: Send> Send for LockFreeQueue<T> {}
unsafe impl<T: Send> Sync for LockFreeQueue<T> {}

impl<T> LockFreeQueue<T> {
    /// Create a new queue with the default capacity.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_QUEUE_CAPACITY)
    }

    /// Create a new queue holding up to `capacity` items.
    ///
    /// The ring is the next power of two at least two, which the sequence
    /// protocol needs to tell a slot's empty generation from its full one; a
    /// one-slot request therefore gets a two-slot ring and still bounds the
    /// queue at one item.
    #[track_caller]
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let ring_len = capacity.next_power_of_two().max(2);

        #[cfg(feature = "std")]
        let buffer: Box<[Slot<T>]> = (0..ring_len)
            .map(|i| Slot {
                sequence: AtomicUsize::new(i),
                data: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect::<std::vec::Vec<_>>()
            .into_boxed_slice();

        #[cfg(not(feature = "std"))]
        let buffer: Box<[Slot<T>]> = (0..ring_len)
            .map(|i| Slot {
                sequence: AtomicUsize::new(i),
                data: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect::<alloc::vec::Vec<_>>()
            .into_boxed_slice();

        Self {
            buffer,
            mask: ring_len - 1,
            ring_len,
            capacity,
            head: CacheAligned::new(AtomicUsize::new(0)),
            tail: CacheAligned::new(AtomicUsize::new(0)),
        }
    }

    /// Try to enqueue an item without blocking.
    ///
    /// Returns `Ok(())` if the item was enqueued, or `Err(item)` if the
    /// queue is full. This is the lock-free fast path: no spinlock, no
    /// mutex, no retry loop.
    #[inline]
    pub fn try_enqueue(&self, item: T) -> Result<(), T> {
        self.try_enqueue_inner::<false>(item).map(|_| ())
    }

    /// Try to enqueue an item, reporting whether this push took the queue from
    /// empty to non-empty.
    ///
    /// Callers that gate a notification on the transition need this; see
    /// [`EnqueueOutcome`]. It costs one extra consumer-cursor read per push, so
    /// it is a separate entry point rather than the default.
    #[inline]
    pub fn try_enqueue_outcome(&self, item: T) -> Result<EnqueueOutcome, T> {
        self.try_enqueue_inner::<true>(item)
    }

    /// `REPORT_TRANSITION` is a const generic, so the consumer-cursor read the
    /// outcome needs is monomorphized away entirely for the callers that ignore
    /// it: the injector and run-queue pushes compile to the same code as a ring
    /// that never tracked the transition.
    #[inline]
    fn try_enqueue_inner<const REPORT_TRANSITION: bool>(
        &self,
        item: T,
    ) -> Result<EnqueueOutcome, T> {
        let mut pos = self.tail.load(Ordering::Relaxed);
        loop {
            let slot = &self.buffer[pos & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            #[allow(clippy::cast_possible_wrap)]
            let diff = seq.wrapping_sub(pos) as isize;

            match diff.cmp(&0) {
                CmpOrdering::Equal => {
                    // The ring can be larger than the requested capacity, so
                    // fullness is the request, not the ring size.
                    if pos.wrapping_sub(self.head.load(Ordering::Acquire)) >= self.capacity {
                        return Err(item);
                    }

                    match self.tail.compare_exchange_weak(
                        pos,
                        pos.wrapping_add(1),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            // Read the consumer cursor *before* publishing this
                            // slot: a consumer can only advance `head` through
                            // already-published slots, so it cannot pass `pos`
                            // while this one is unpublished, and
                            // `head == pos` is then exactly "the queue held no
                            // items".
                            let outcome =
                                if REPORT_TRANSITION && self.head.load(Ordering::Acquire) == pos {
                                    EnqueueOutcome::BecameNonEmpty
                                } else {
                                    EnqueueOutcome::AlreadyNonEmpty
                                };
                            // SAFETY: winning the tail CAS grants exclusive
                            // right to fill this slot's sequence generation; its
                            // payload cell is uninitialized (fresh or drained)
                            // until this write publishes it.
                            unsafe {
                                (*slot.data.get()).write(item);
                            }
                            slot.sequence.store(pos.wrapping_add(1), Ordering::Release);
                            return Ok(outcome);
                        }
                        Err(actual) => pos = actual,
                    }
                }
                // Queue is full: sequence lags behind tail, meaning all slots
                // between head and tail are occupied.
                CmpOrdering::Less => return Err(item),
                // Another producer advanced tail before us: reload and retry.
                CmpOrdering::Greater => pos = self.tail.load(Ordering::Relaxed),
            }
        }
    }

    /// Enqueue an item, retrying with exponential backoff if the queue is full.
    ///
    /// This preserves the unblocked-sender contract of the previous API: the
    /// call always eventually succeeds (assuming consumers make progress).
    /// The backoff path uses `core::hint::spin_loop` and, on std targets,
    /// `std::thread::yield_now` after heavy contention, but never acquires a
    /// global lock, so multiple producers can enqueue concurrently.
    #[inline]
    pub fn enqueue(&self, item: T) {
        let mut backoff: usize = 1;
        let mut item = Some(item);
        loop {
            match self.try_enqueue(item.take().expect("invariant: item present")) {
                Ok(()) => return,
                Err(returned) => {
                    item = Some(returned);
                    for _ in 0..backoff {
                        core::hint::spin_loop();
                    }
                    if backoff < 64 {
                        backoff = backoff.saturating_mul(2);
                    } else {
                        #[cfg(feature = "std")]
                        {
                            std::thread::yield_now();
                        }
                        backoff = 1;
                    }
                }
            }
        }
    }

    /// Try to dequeue an item from the front of the queue.
    /// Returns `None` if the queue is empty.
    ///
    /// This is the lock-free fast path: no spinlock, no mutex.
    #[inline]
    pub fn try_dequeue(&self) -> Option<T> {
        let mut pos = self.head.load(Ordering::Relaxed);
        loop {
            let slot = &self.buffer[pos & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            #[allow(clippy::cast_possible_wrap)]
            let diff = seq.wrapping_sub(pos.wrapping_add(1)) as isize;

            match diff.cmp(&0) {
                CmpOrdering::Equal => {
                    // Slot has data: try to claim it by advancing head.
                    match self.head.compare_exchange_weak(
                        pos,
                        pos.wrapping_add(1),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            // SAFETY: winning the head CAS means no other
                            // consumer can claim this slot, and the sequence
                            // == pos+1 invariant means the producer finished
                            // writing; that producer cannot write again until
                            // the store below opens the next generation.
                            let item = unsafe { (*slot.data.get()).assume_init_read() };
                            slot.sequence
                                .store(pos.wrapping_add(self.ring_len), Ordering::Release);
                            return Some(item);
                        }
                        Err(actual) => pos = actual,
                    }
                }
                // Queue is empty: sequence has not advanced past pos+1.
                CmpOrdering::Less => return None,
                // Another consumer advanced head before us: reload and retry.
                CmpOrdering::Greater => pos = self.head.load(Ordering::Relaxed),
            }
        }
    }

    /// Check if the queue is empty.
    ///
    /// This is a best-effort check: the queue may have items added or removed
    /// between this call and the next operation. It is safe to call
    /// concurrently with enqueue/dequeue.
    pub fn is_empty(&self) -> bool {
        self.tail.load(Ordering::Acquire) == self.head.load(Ordering::Acquire)
    }

    /// Check if the queue holds `capacity` items.
    ///
    /// Best-effort in the same sense as [`is_empty`](LockFreeQueue::is_empty).
    pub fn is_full(&self) -> bool {
        self.tail
            .load(Ordering::Acquire)
            .wrapping_sub(self.head.load(Ordering::Acquire))
            >= self.capacity
    }

    /// Number of items the queue accepts before `try_enqueue` reports full.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> Default for LockFreeQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for LockFreeQueue<T> {
    fn drop(&mut self) {
        // Exclusive access: walk the published range and drop what nobody took,
        // reading the sequence numbers without atomics.
        let head = *self.head.0.get_mut();
        let tail = *self.tail.0.get_mut();

        for pos in head..tail {
            let slot = &mut self.buffer[pos & self.mask];
            if *slot.sequence.get_mut() == pos.wrapping_add(1) {
                // SAFETY: the published sequence marks the payload
                // initialized, and each position is visited once.
                unsafe {
                    (*slot.data.get()).assume_init_drop();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::{mem::size_of, num::NonZeroUsize, sync::atomic::AtomicUsize};

    #[cfg(feature = "std")]
    use std::sync::Arc;

    #[cfg(not(feature = "std"))]
    use alloc::sync::Arc;

    #[test]
    fn slot_costs_one_machine_word_over_its_payload() {
        type RepresentativePayload = (NonZeroUsize, [usize; 16]);
        let word_size = size_of::<usize>();

        assert_eq!(size_of::<RepresentativePayload>(), 17 * word_size);
        assert_eq!(size_of::<Slot<RepresentativePayload>>(), 18 * word_size);

        // A payload without a niche is where the representation shows: an
        // `Option<T>` slot would have to store a discriminant word, and the
        // sequence number already says whether the slot is occupied.
        assert_eq!(size_of::<Option<[usize; 4]>>(), 5 * word_size);
        assert_eq!(size_of::<Slot<[usize; 4]>>(), 5 * word_size);
    }

    #[test]
    fn test_lock_free_queue_basic() {
        let queue = LockFreeQueue::<i32>::with_capacity(4);
        assert!(queue.is_empty());

        queue.enqueue(1);
        queue.enqueue(2);
        assert!(!queue.is_empty());

        assert_eq!(queue.try_dequeue(), Some(1));
        assert_eq!(queue.try_dequeue(), Some(2));
        assert_eq!(queue.try_dequeue(), None);
        assert!(queue.is_empty());
    }

    #[test]
    fn a_one_slot_request_gets_a_two_slot_ring_and_bounds_at_one() {
        let queue = LockFreeQueue::<i32>::with_capacity(1);
        assert_eq!(queue.capacity(), 1);
        assert!(!queue.is_full());

        assert_eq!(queue.try_enqueue(1), Ok(()));
        assert!(queue.is_full());
        assert_eq!(queue.try_enqueue(2), Err(2));

        assert_eq!(queue.try_dequeue(), Some(1));
        assert!(queue.is_empty());
        assert_eq!(queue.try_enqueue(2), Ok(()));
    }

    #[test]
    fn a_non_power_of_two_request_bounds_the_queue_exactly() {
        // Ring of four, capacity of three: the fourth push must be rejected
        // even though a slot is free.
        let queue = LockFreeQueue::<i32>::with_capacity(3);
        assert_eq!(queue.capacity(), 3);

        for i in 0..3 {
            assert_eq!(queue.try_enqueue(i), Ok(()), "push {i}");
        }
        assert!(queue.is_full());
        assert_eq!(queue.try_enqueue(3), Err(3));

        assert_eq!(queue.try_dequeue(), Some(0));
        assert!(!queue.is_full());
        assert_eq!(queue.try_enqueue(3), Ok(()));
    }

    /// The classification `moirai-core`'s notifier fence hangs on: a push
    /// reports whether it took the ring from empty to non-empty.
    ///
    /// Misclassifying an empty→non-empty push as occupied would strand a parked
    /// receiver (its fence and counter read would be skipped), and the reverse
    /// misclassification would only cost an unnecessary fence — so the empty
    /// case, the occupied case, and the return to empty are all pinned.
    #[test]
    fn reports_the_empty_to_non_empty_transition() {
        let queue: LockFreeQueue<u32> = LockFreeQueue::with_capacity(4);

        assert_eq!(
            queue.try_enqueue_outcome(1),
            Ok(EnqueueOutcome::BecameNonEmpty)
        );
        assert_eq!(
            queue.try_enqueue_outcome(2),
            Ok(EnqueueOutcome::AlreadyNonEmpty)
        );
        assert_eq!(
            queue.try_enqueue_outcome(3),
            Ok(EnqueueOutcome::AlreadyNonEmpty)
        );

        assert_eq!(queue.try_dequeue(), Some(1));
        assert_eq!(queue.try_dequeue(), Some(2));
        assert_eq!(
            queue.try_enqueue_outcome(4),
            Ok(EnqueueOutcome::AlreadyNonEmpty)
        );

        assert_eq!(queue.try_dequeue(), Some(3));
        assert_eq!(queue.try_dequeue(), Some(4));
        // Drained: the next push is the empty→non-empty transition again.
        assert_eq!(
            queue.try_enqueue_outcome(5),
            Ok(EnqueueOutcome::BecameNonEmpty)
        );
    }

    /// A full ring reports no transition and hands the value back.
    ///
    /// The notifier must not treat a rejected push as a transition: nothing was
    /// published, so there is nothing to wake a receiver for.
    #[test]
    fn a_rejected_push_reports_no_transition() {
        let queue: LockFreeQueue<u32> = LockFreeQueue::with_capacity(1);
        assert_eq!(
            queue.try_enqueue_outcome(1),
            Ok(EnqueueOutcome::BecameNonEmpty)
        );
        assert_eq!(queue.try_enqueue_outcome(2), Err(2));
    }

    #[test]
    fn test_lock_free_queue_wrap_around() {
        // Fill and drain multiple times to exercise slot reuse.
        let queue = LockFreeQueue::<i32>::with_capacity(4);
        for round in 0..16 {
            for i in 0..3 {
                queue.enqueue(round * 3 + i);
            }
            for i in 0..3 {
                assert_eq!(
                    queue.try_dequeue(),
                    Some(round * 3 + i),
                    "round {round}, item {i}"
                );
            }
            assert!(queue.try_dequeue().is_none(), "round {round} not empty");
        }
    }

    #[test]
    fn test_lock_free_queue_full_try_enqueue() {
        let queue = LockFreeQueue::<i32>::with_capacity(4);
        for i in 0..4 {
            queue.try_enqueue(i).unwrap();
        }
        // Now full.
        assert!(queue.try_enqueue(99).is_err());
        assert_eq!(queue.try_dequeue(), Some(0));
        // One slot freed.
        queue.try_enqueue(99).unwrap();
    }

    #[test]
    fn test_lock_free_queue_drop_runs_destructors() {
        struct DropCounter {
            counter: Arc<AtomicUsize>,
        }
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        {
            let queue = LockFreeQueue::<DropCounter>::with_capacity(4);
            for _ in 0..3 {
                queue.enqueue(DropCounter {
                    counter: Arc::clone(&counter),
                });
            }
            // Drop the queue without draining: destructors must run.
        }
        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_lock_free_queue_concurrent_mpmc() {
        use std::thread;

        let queue = Arc::new(LockFreeQueue::<i32>::with_capacity(1024));
        let num_producers = 4;
        let num_consumers = 4;
        let items_per_producer = 1000;
        let total_items = num_producers * items_per_producer;

        let mut handles = Vec::new();

        for p in 0..num_producers {
            let q = Arc::clone(&queue);
            handles.push(thread::spawn(move || {
                for i in 0..items_per_producer {
                    q.enqueue((p * items_per_producer + i) as i32);
                }
            }));
        }

        let consumed = Arc::new(AtomicUsize::new(0));
        for _ in 0..num_consumers {
            let q = Arc::clone(&queue);
            let c = Arc::clone(&consumed);
            handles.push(thread::spawn(move || {
                while c.load(Ordering::Relaxed) < total_items {
                    if q.try_dequeue().is_some() {
                        c.fetch_add(1, Ordering::Relaxed);
                    } else {
                        std::thread::yield_now();
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(consumed.load(Ordering::Relaxed), total_items);
    }
}
