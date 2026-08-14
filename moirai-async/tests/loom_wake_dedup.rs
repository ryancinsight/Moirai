//! Loom model of the async executor's `is_queued` wake-deduplication protocol.
//!
//! The production queue publishes task ownership through each slot's
//! Release/Acquire sequence. `is_queued` is a separate one-bit protocol: the
//! consumer clears it after removing the current entry, and a waker atomically
//! claims the false state before adding one replacement entry. The flag carries
//! no task data, so both flag operations use Relaxed ordering; the atomic
//! read-modify-write still gives the two state transitions one modification
//! order.
//!
//! This model covers the race between dequeue/clear and wake/swap. It asserts
//! that the waker contributes either no replacement (the consumer owns the
//! existing entry) or exactly one replacement (the waker observes the clear),
//! never a duplicate or a lost queue entry. The abstract queue counter uses
//! SeqCst only to make the model's post-join accounting observable; it is not a
//! claim about the production queue's slot ordering, which is tested by its
//! own Release/Acquire protocol models.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo nextest run -p moirai-async --test loom_wake_dedup --release`
//!
//! Under a normal build the `#![cfg(loom)]` gate makes this file empty.

#![cfg(loom)]

use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;

struct WakeDedup {
    is_queued: AtomicBool,
    queue_entries: AtomicUsize,
}

impl WakeDedup {
    fn new() -> Self {
        Self {
            is_queued: AtomicBool::new(true),
            queue_entries: AtomicUsize::new(1),
        }
    }

    fn dequeue_and_clear(&self) {
        assert_eq!(
            self.queue_entries.fetch_sub(1, Ordering::SeqCst),
            1,
            "the modeled consumer owns the initial queue entry"
        );
        self.is_queued.store(false, Ordering::Relaxed);
    }

    fn wake_and_maybe_enqueue(&self) -> bool {
        let claimed = !self.is_queued.swap(true, Ordering::Relaxed);
        if claimed {
            self.queue_entries.fetch_add(1, Ordering::SeqCst);
        }
        claimed
    }
}

#[test]
fn dequeue_clear_and_wake_swap_never_duplicate_or_lose_entry() {
    loom::model(|| {
        let protocol = Arc::new(WakeDedup::new());

        let consumer_protocol = Arc::clone(&protocol);
        let consumer = thread::spawn(move || consumer_protocol.dequeue_and_clear());

        let waker_protocol = Arc::clone(&protocol);
        let waker = thread::spawn(move || waker_protocol.wake_and_maybe_enqueue());

        consumer.join().unwrap();
        let claimed = waker.join().unwrap();

        let entries = protocol.queue_entries.load(Ordering::SeqCst);
        assert!(
            entries <= 1,
            "wake deduplication produced {entries} queue entries"
        );
        let expected_entries = if claimed { 1 } else { 0 };
        assert_eq!(
            entries, expected_entries,
            "the wake claim and queue entry count diverged"
        );
    });
}
