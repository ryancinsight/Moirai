//! The ring's contract: the transition classification the notifier's fence
//! hangs on, the capacity bound, drop safety, and a concurrent MPMC run.

use super::ring::Slot;
use super::*;
use core::{
    mem::size_of,
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

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
