//! Loom model of connection-pool reservation accounting.
//!
//! `ConnectionPool` serializes admission checks and increments with its
//! `active_connections` mutex. Cancellation and successful admission commit
//! release one already-held reservation outside that mutex. The reservation
//! counter therefore needs atomicity and modification order, not a global
//! synchronization edge: it carries no connection payload, and a concurrent
//! release can only make an admission snapshot conservatively larger.
//!
//! This model exhausts two serialized admission attempts racing one paired
//! cancellation. It asserts that every successful increment observed the
//! capacity boundary and that a paired release never underflows the counter.
//! The abstract counter uses the production Relaxed orderings.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo nextest run -p moirai-async --test loom_connection_pool --release`
//!
//! Under a normal build the `#![cfg(loom)]` gate makes this file empty.

#![cfg(loom)]

use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;

const CAPACITY: usize = 2;

struct ReservationModel {
    admission: Mutex<()>,
    reserved: AtomicUsize,
}

impl ReservationModel {
    fn new() -> Self {
        Self {
            admission: Mutex::new(()),
            // One outstanding reservation is paired with the cancellation
            // thread. This is the smallest state that exercises release races.
            reserved: AtomicUsize::new(1),
        }
    }

    fn try_reserve(&self) -> bool {
        let _admission = self.admission.lock().unwrap();
        if self.reserved.load(Ordering::Relaxed) >= CAPACITY {
            return false;
        }

        let previous = self.reserved.fetch_add(1, Ordering::Relaxed);
        assert!(
            previous < CAPACITY,
            "admission increment crossed the capacity boundary"
        );
        true
    }

    fn release_reservation(&self) {
        let previous = self.reserved.fetch_sub(1, Ordering::Relaxed);
        assert!(previous > 0, "reservation release underflowed the counter");
    }
}

#[test]
fn reservation_accounting_preserves_capacity_and_pairing() {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(4);
    builder.check(|| {
        let model = Arc::new(ReservationModel::new());

        let first_model = Arc::clone(&model);
        let first = thread::spawn(move || first_model.try_reserve());

        let second_model = Arc::clone(&model);
        let second = thread::spawn(move || second_model.try_reserve());

        let cancellation_model = Arc::clone(&model);
        let cancellation = thread::spawn(move || cancellation_model.release_reservation());

        let first_granted = if first.join().unwrap() { 1 } else { 0 };
        let second_granted = if second.join().unwrap() { 1 } else { 0 };
        let granted = first_granted + second_granted;
        cancellation.join().unwrap();

        let reserved = model.reserved.load(Ordering::Relaxed);
        assert!(reserved <= CAPACITY, "reservation count exceeded capacity");
        assert_eq!(reserved, granted, "paired reservation accounting diverged");
    });
}
