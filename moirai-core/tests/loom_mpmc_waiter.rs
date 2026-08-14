//! loom exhaustive-interleaving model of the bounded MPMC channel's
//! waiter-count protocol.
//!
//! # The protocol
//!
//! `MpmcChannel`'s bounded path keeps items in a lock-free ring
//! (`BoundedMpmcQueue`) but parks blocked threads on a `Condvar` guarded by the
//! channel `Mutex`. Signalling that condvar means taking the mutex, so the hot
//! path elides it with a counter: `sender_waiter_count` /
//! `receiver_waiter_count` record how many threads are parked, and a thread
//! that pushes or pops only takes the mutex to notify when the counter is
//! non-zero.
//!
//! The counter and the ring are separate locations, and the *ring is not
//! covered by the mutex*, so the two sides form a store-buffer (Dekker) pair:
//!
//! | | waiter (`recv_bounded` slow path) | notifier (`send_bounded` fast path) |
//! |---|---|---|
//! | first | `receiver_waiter_count.fetch_add` | `queue.try_push` |
//! | then  | `queue.try_pop` (re-check)        | `receiver_waiter_count.load` |
//!
//! The failure this models is the lost wakeup: the notifier reads zero waiters
//! and skips the notify while the waiter reads an empty queue and parks — on a
//! channel that now holds an item. `Condvar::wait` has no timeout, so the
//! waiter stays parked until an unrelated send or the channel's close.
//!
//! # What this file establishes
//!
//! [`waiter_registers_before_rechecking_the_queue`] is the shipped shape and
//! must pass: registering before the re-check is what makes the two sequences
//! above a genuine Dekker pair rather than a plain race. Before this model
//! existed, `recv_bounded` had them inverted (re-check, then register), which
//! no ordering strength can rescue — [`inverted_registration_loses_the_wakeup`]
//! reproduces that as a reachable failure so the regression cannot return
//! silently.
//!
//! # Why the model carries an explicit `SeqCst` fence
//!
//! Same modeling device, and same caveat, as `moirai-executor`'s
//! `loom_wake_handshake.rs`: loom does not reconstruct the SC total order for
//! bare `SeqCst` atomics in a store-buffer shape, and needs a `fence(SeqCst)`
//! to stand in for the Store→Load barrier. Unlike that file, the fence here is
//! **not** purely a modeling artifact — the production notifier reaches its
//! `SeqCst` load through a plain release store in `try_push`, not through a
//! `SeqCst` RMW, so the barrier it represents is the one production relies on
//! the `SeqCst` pair to supply. [`notifier_without_the_store_load_barrier`]
//! measures exactly how much the barrier is doing.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo test -p moirai-core --test loom_mpmc_waiter --release`
//!
//! Under a normal build the `#![cfg(loom)]` gate makes this file empty, so it
//! never affects the standard test suite or pulls in the `loom` dependency.

#![cfg(loom)]

use loom::sync::atomic::{fence, AtomicUsize, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;

/// Ring-slot occupancy, standing in for `BoundedMpmcQueue`'s slot sequence.
///
/// The real queue publishes a pushed value with `sequence.store(_, Release)`
/// and reads it with `sequence.load(Acquire)`; the model keeps those exact
/// orderings on a single slot, which is the whole ordering content of the
/// one-slot case.
struct Ring {
    slot: AtomicUsize,
}

impl Ring {
    fn new() -> Self {
        Self {
            slot: AtomicUsize::new(0),
        }
    }

    fn try_push(&self) -> bool {
        if self.slot.load(Ordering::Relaxed) == 0 {
            self.slot.store(1, Ordering::Release);
            true
        } else {
            false
        }
    }

    fn try_pop(&self) -> bool {
        if self.slot.load(Ordering::Acquire) == 1 {
            self.slot.store(0, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    fn is_occupied(&self) -> bool {
        self.slot.load(Ordering::Relaxed) == 1
    }
}

/// Shared state for one modeled channel: the ring, the receiver waiter count,
/// and the mutex the condvar would be signalled under.
struct Channel {
    ring: Ring,
    receiver_waiters: AtomicUsize,
    /// Set when the notifier decides to signal. `Condvar` itself is not
    /// modeled: what matters is only whether the signal is issued at all.
    notified: AtomicUsize,
    /// Set when the receiver commits to parking.
    parked: AtomicUsize,
    lock: Mutex<()>,
}

impl Channel {
    fn new() -> Self {
        Self {
            ring: Ring::new(),
            receiver_waiters: AtomicUsize::new(0),
            notified: AtomicUsize::new(0),
            parked: AtomicUsize::new(0),
            lock: Mutex::new(()),
        }
    }

    /// `send_bounded`'s lock-free fast path: push, then consult the counter.
    ///
    /// `store_load_barrier` selects whether the Store→Load edge between the
    /// push and the counter read is present.
    fn notifier(&self, store_load_barrier: bool) {
        assert!(self.ring.try_push(), "the modeled ring starts empty");

        // Production must spell this out: `try_push` ends in a plain release
        // store and the counter read is a `SeqCst` *load*, which on x86-64 is
        // an ordinary `mov` and orders nothing against the preceding store.
        // See `MpmcChannel::send_bounded`.
        if store_load_barrier {
            fence(Ordering::SeqCst);
        }

        if self.receiver_waiters.load(Ordering::SeqCst) > 0 {
            let _guard = self.lock.lock().unwrap();
            self.notified.store(1, Ordering::Relaxed);
        }
    }

    /// `recv_bounded`'s slow path in the shipped order: take the mutex,
    /// register, re-check the ring, park only if it is still empty.
    fn waiter_registers_first(&self, store_load_barrier: bool) {
        if self.ring.try_pop() {
            return;
        }

        let guard = self.lock.lock().unwrap();

        self.receiver_waiters.fetch_add(1, Ordering::SeqCst);

        // In production this barrier is supplied by the `SeqCst` RMW above,
        // which lowers to a `lock`-prefixed instruction and is therefore a
        // full StoreLoad barrier on x86-64 (and `dmb ish` on aarch64). loom
        // does not model a bare `SeqCst` RMW as an SC fence for a subsequent
        // non-SC load, so the fence is spelled out — the same modeling device
        // documented in `loom_wake_handshake.rs`.
        if store_load_barrier {
            fence(Ordering::SeqCst);
        }

        if self.ring.try_pop() {
            self.receiver_waiters.fetch_sub(1, Ordering::Relaxed);
            return;
        }

        // `Condvar::wait(guard)` would run here, atomically releasing the
        // mutex. Committing to the park is the observable act.
        self.parked.store(1, Ordering::Relaxed);
        drop(guard);
    }

    /// The pre-fix order: re-check the ring, *then* register.
    ///
    /// Retained as an executable counter-example — see the module docs.
    fn waiter_rechecks_first(&self) {
        if self.ring.try_pop() {
            return;
        }

        let guard = self.lock.lock().unwrap();

        if self.ring.try_pop() {
            return;
        }

        self.receiver_waiters.fetch_add(1, Ordering::SeqCst);
        self.parked.store(1, Ordering::Relaxed);
        drop(guard);
    }

    /// A wakeup is lost when a receiver is parked on a ring that holds an item
    /// and no notify was issued.
    fn lost_wakeup(&self) -> bool {
        self.parked.load(Ordering::Relaxed) == 1
            && self.ring.is_occupied()
            && self.notified.load(Ordering::Relaxed) == 0
    }
}

fn model(notifier_barrier: bool, waiter_barrier: bool, register_first: bool) -> bool {
    let lost = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observed = std::sync::Arc::clone(&lost);

    loom::model(move || {
        let channel = Arc::new(Channel::new());

        let notifier_side = Arc::clone(&channel);
        let notifier = thread::spawn(move || notifier_side.notifier(notifier_barrier));

        let waiter_side = Arc::clone(&channel);
        let waiter = thread::spawn(move || {
            if register_first {
                waiter_side.waiter_registers_first(waiter_barrier);
            } else {
                waiter_side.waiter_rechecks_first();
            }
        });

        notifier.join().unwrap();
        waiter.join().unwrap();

        if channel.lost_wakeup() {
            observed.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });

    lost.load(std::sync::atomic::Ordering::Relaxed)
}

/// The shipped protocol: with the Store->Load barrier present on both halves of
/// the Dekker pair, no interleaving strands the receiver.
#[test]
fn waiter_registers_before_rechecking_the_queue() {
    assert!(
        !model(true, true, true),
        "register-before-recheck with both Store->Load barriers must admit no lost wakeup"
    );
}

/// The pre-fix protocol is broken independently of memory ordering: the window
/// between the failed re-check and the registration is a plain race, and no
/// barrier closes it. `recv_bounded` had exactly this shape.
#[test]
fn inverted_registration_loses_the_wakeup() {
    assert!(
        model(true, true, false),
        "recheck-before-register must remain reachable as a lost wakeup, or this          model no longer covers the regression it was written for"
    );
}

/// The notifier half is the one production has to spell out.
///
/// The waiter registers with a `SeqCst` RMW, which is a real StoreLoad barrier
/// on the targets moirai builds for. The notifier has no such thing: `try_push`
/// ends in a release store and the counter read is a `SeqCst` load, which
/// orders nothing against it. This is therefore the *as-written* production
/// shape, and it must stay unreachable — which is why `send_bounded`,
/// `recv_bounded`, `try_send` and `try_recv` carry an explicit
/// `fence(SeqCst)` before consulting the counter.
#[test]
fn notifier_without_the_store_load_barrier_loses_the_wakeup() {
    assert!(
        model(false, true, true),
        "an unfenced notifier must be reachable as a lost wakeup, or the fences in          the production notify paths are unnecessary and should be deleted"
    );
}

/// Symmetric counter-example: the waiter's barrier is load-bearing too, so a
/// future rewrite that demotes the registration below `SeqCst` reopens the hole.
#[test]
fn waiter_without_the_store_load_barrier_loses_the_wakeup() {
    assert!(
        model(true, false, true),
        "an unfenced waiter must be reachable as a lost wakeup"
    );
}
