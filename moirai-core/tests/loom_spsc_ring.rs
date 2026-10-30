//! Loom model of the SPSC ring's publication and reclamation orderings.
//!
//! The production ring owns its slots behind `UnsafeCell`: the producer writes
//! a slot before releasing `head`, and the consumer acquires `head` before
//! reading that slot. The consumer then releases `tail`, which the producer
//! acquires before reusing a slot. This model keeps those two release/acquire
//! edges, exercises wrap-around with a capacity-two ring, and checks that a
//! close published after the final head never overtakes that value.
//!
//! The model uses three messages and a preemption bound of four. The message
//! count forces one producer retry and one slot reuse; the bound is the finite
//! interleaving budget for this regression and is intentionally recorded here.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo nextest run -p moirai-core --test loom_spsc_ring --release`

#![cfg(loom)]

use loom::cell::UnsafeCell;
use loom::sync::Arc;
use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use loom::thread;

const CAPACITY: usize = 2;
const MESSAGE_COUNT: usize = 3;

/// A capacity-two SPSC ring with the production publication edges.
struct Ring {
    slots: [UnsafeCell<usize>; CAPACITY],
    head: AtomicUsize,
    tail: AtomicUsize,
    closed: AtomicBool,
}

impl Ring {
    fn new() -> Self {
        Self {
            slots: [UnsafeCell::new(0), UnsafeCell::new(0)],
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
        }
    }

    fn try_send(&self, value: usize) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= CAPACITY {
            return false;
        }

        let slot = self
            .slots
            .get(head & (CAPACITY - 1))
            .expect("invariant: masked head selects a ring slot");
        slot.with_mut(|pointer| {
            // SAFETY: only the producer writes this slot, and the acquired
            // tail proves the consumer has released its previous ownership.
            unsafe { *pointer = value };
        });
        self.head.store(head.wrapping_add(1), Ordering::Release);
        true
    }

    fn try_recv(&self) -> Option<usize> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail == head {
            return None;
        }

        let slot = self
            .slots
            .get(tail & (CAPACITY - 1))
            .expect("invariant: masked tail selects a ring slot");
        let value = slot.with(|pointer| {
            // SAFETY: the acquired head publishes this initialized slot, and
            // only the consumer reads it before releasing the next tail.
            unsafe { *pointer }
        });
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }

    fn try_recv_or_closed(&self) -> Receive {
        if let Some(value) = self.try_recv() {
            return Receive::Value(value);
        }
        if !self.closed.load(Ordering::Acquire) {
            return Receive::Empty;
        }

        self.try_recv().map_or(Receive::Closed, Receive::Value)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Receive {
    Value(usize),
    Empty,
    Closed,
}

#[test]
fn publication_and_reclamation_preserve_fifo_values() {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(4);
    builder.check(|| {
        let ring = Arc::new(Ring::new());
        let producer_ring = Arc::clone(&ring);
        let producer = thread::spawn(move || {
            for value in 1..=MESSAGE_COUNT {
                let value = value * 10;
                while !producer_ring.try_send(value) {
                    thread::yield_now();
                }
            }
        });

        let consumer_ring = Arc::clone(&ring);
        let consumer = thread::spawn(move || {
            let mut values = Vec::with_capacity(MESSAGE_COUNT);
            while values.len() < MESSAGE_COUNT {
                if let Some(value) = consumer_ring.try_recv() {
                    values.push(value);
                } else {
                    thread::yield_now();
                }
            }
            values
        });

        producer.join().expect("invariant: producer completes");
        let values = consumer.join().expect("invariant: consumer completes");
        assert_eq!(values, [10, 20, 30]);
    });
}

#[test]
fn closure_cannot_overtake_the_final_publication() {
    // Three preemptions cover consumer-before-publication, producer publication
    // plus close, and the consumer's ordered retry after acquiring the close.
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);
    builder.check(|| {
        let ring = Arc::new(Ring::new());
        let producer_ring = Arc::clone(&ring);
        let producer = thread::spawn(move || {
            assert!(producer_ring.try_send(37));
            producer_ring.close();
        });

        let consumer_ring = Arc::clone(&ring);
        let consumer = thread::spawn(move || {
            let mut values = Vec::new();
            loop {
                match consumer_ring.try_recv_or_closed() {
                    Receive::Value(value) => values.push(value),
                    Receive::Empty => thread::yield_now(),
                    Receive::Closed => break,
                }
            }
            values
        });

        producer.join().expect("invariant: producer completes");
        let values = consumer.join().expect("invariant: consumer completes");
        assert_eq!(values, [37]);
    });
}
