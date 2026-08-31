#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use super::queue::BoundedMpmcQueue;
use super::recv::MpmcReceiver;
use super::send::MpmcSender;
use super::{MpmcState, MPMC_BLOCK_SPINS};
use crate::channel::error::{Channel, ChannelError, Result};
use crate::channel::CHANNEL_STORE_LOAD_ORDER;
use std::collections::VecDeque;
use std::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

mod roles;

/// Multi-Producer Multi-Consumer channel with bounded capacity
/// Uses mutex-based implementation for simplicity and correctness
pub struct MpmcChannel<T> {
    pub(super) state: Arc<(Mutex<MpmcState<T>>, Condvar, Condvar)>,
    pub(super) bounded: Option<Arc<BoundedMpmcQueue<T>>>,
    pub(super) closed: Arc<AtomicBool>,
    pub(super) sender_waiter_count: Arc<AtomicUsize>,
    pub(super) receiver_waiter_count: Arc<AtomicUsize>,
}

/// Slots preallocated for an unbounded channel's mutex-guarded deque.
///
/// The unbounded path is the only one that stores items in `MpmcState::queue`,
/// and it has no capacity to size that allocation from. One `VecDeque` growth
/// step reallocates and moves every element while the channel mutex is held,
/// so the first few sends are paid for up front; 16 covers the short-burst
/// case without committing memory a mostly-idle channel never uses.
const UNBOUNDED_PREALLOCATED_SLOTS: usize = 16;

impl<T> MpmcChannel<T> {
    /// Create a new MPMC channel with optional capacity
    pub fn new(capacity: Option<usize>) -> Self {
        let state = MpmcState {
            // Deliberately asymmetric, and not the inversion it looks like: a
            // bounded channel stores nothing here. Every `Channel` method
            // dispatches on `self.bounded` first, and `bounded` is `Some`
            // exactly when `capacity` is, so for a bounded channel this deque
            // stays empty for life and preallocating `capacity` slots for it
            // would reserve a second copy of the ring that is never written.
            // Items live in the lock-free `BoundedMpmcQueue`, which allocates
            // its ring in `BoundedMpmcQueue::new` below.
            queue: if capacity.is_some() {
                VecDeque::new()
            } else {
                VecDeque::with_capacity(UNBOUNDED_PREALLOCATED_SLOTS)
            },
            capacity,
            closed: false,
            sender_count: 0,
            receiver_count: 0,
        };

        let bounded = capacity.map(BoundedMpmcQueue::new).map(Arc::new);

        Self {
            state: Arc::new((Mutex::new(state), Condvar::new(), Condvar::new())),
            bounded,
            closed: Arc::new(AtomicBool::new(false)),
            sender_waiter_count: Arc::new(AtomicUsize::new(0)),
            receiver_waiter_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Create an unbounded channel
    pub fn unbounded() -> Self {
        Self::new(None)
    }

    /// Create a bounded channel with given capacity
    pub fn bounded(capacity: usize) -> Self {
        Self::new(Some(capacity))
    }

    /// Create a channel pair for ergonomic usage
    pub fn channel(capacity: Option<usize>) -> (MpmcSender<T>, MpmcReceiver<T>) {
        let channel = Arc::new(Self::new(capacity));
        let (mutex, _, _) = &*channel.state;

        {
            let mut state = mutex.lock().unwrap();
            state.sender_count = 1;
            state.receiver_count = 1;
        }

        (
            MpmcSender {
                channel: channel.clone(),
            },
            MpmcReceiver { channel },
        )
    }

    fn send_bounded(&self, queue: &BoundedMpmcQueue<T>, mut value: T) -> Result<()>
    where
        T: Send,
    {
        let mut spin_count = 0;

        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(ChannelError::Closed);
            }

            match queue.try_push(value) {
                Ok(()) => {
                    // SeqCst, load-bearing: this is the notifier half of a
                    // store-buffer (Dekker) pair with `recv_bounded`'s
                    // registration. Here the queue write precedes the counter
                    // read; there the counter write precedes the queue read.
                    // If either side could reorder Store→Load, this side reads
                    // "no waiters" while that side reads "still empty", and a
                    // receiver parks forever on an item that is already
                    // queued. Acquire is insufficient — it orders Load→Load
                    // and Load→Store, never Store→Load. The queue is
                    // lock-free, so the channel mutex orders neither side.
                    //
                    // The waiter half gets that barrier free from its `SeqCst`
                    // RMW; this half does not — `try_push` ends in a plain
                    // release store and a `SeqCst` load is an ordinary `mov`
                    // on x86-64 — so the fence is explicit and load-bearing.
                    // `tests/loom_mpmc_waiter.rs`
                    // (`notifier_without_the_store_load_barrier_loses_the_wakeup`)
                    // enumerates the interleaving it rules out.
                    //
                    // Cost, measured: `bounded_channel_matrix/moirai_mpmc`
                    // regresses 13–64% at 4 and 8 producers relative to the
                    // unfenced (and unsound) code it replaces. The way out is
                    // not to drop the barrier but to need it less often — a
                    // receiver only parks after observing the ring empty, so
                    // only a push that takes the ring from empty to non-empty
                    // can race a registration. Threading that transition out
                    // of `try_push` and fencing only then is the follow-up.
                    fence(CHANNEL_STORE_LOAD_ORDER);
                    if self.receiver_waiter_count.load(CHANNEL_STORE_LOAD_ORDER) > 0 {
                        let (mutex, _, not_empty) = &*self.state;
                        let _guard = mutex.lock().unwrap();
                        not_empty.notify_one();
                    }
                    return Ok(());
                }
                Err(returned) => {
                    value = returned;
                }
            }

            if spin_count < MPMC_BLOCK_SPINS {
                for _ in 0..(1 << spin_count) {
                    std::hint::spin_loop();
                }
                spin_count += 1;
                continue;
            }

            // Fallback to condvar wait to prevent CPU contention and busy-looping
            let (mutex, not_full, _) = &*self.state;
            let mut guard = mutex.lock().unwrap();

            if self.closed.load(Ordering::Acquire) || guard.closed {
                return Err(ChannelError::Closed);
            }

            // Register *before* the re-check, never after: the counter must be
            // visible to any receiver that frees a slot from here on, or that
            // receiver reads zero and skips the notify while this thread goes
            // on to park. SeqCst, load-bearing: this is the waiter half of the
            // Dekker pair described above, and the `fetch_add` is also the
            // Store→Load barrier separating it from the `try_push` below.
            self.sender_waiter_count
                .fetch_add(1, CHANNEL_STORE_LOAD_ORDER);

            match queue.try_push(value) {
                Ok(()) => {
                    // Relaxed: deregistration. No happens-before edge is
                    // needed — the counter only ever gates a `notify_one`, so
                    // a receiver still reading the pre-decrement value takes
                    // the mutex and signals a condvar nobody waits on. A
                    // spurious notify is free; a missed one is a hang, and
                    // only the increment above can be missed.
                    self.sender_waiter_count.fetch_sub(1, Ordering::Relaxed);
                    drop(guard);
                    // Relaxed: unlike the fast path above, this push happened
                    // while holding the channel mutex, and a receiver
                    // registers (and re-checks the queue) while holding that
                    // same mutex. Either it registered before this thread took
                    // the lock — then its `fetch_add` happens-before this load
                    // through the mutex and the load observes it — or it takes
                    // the lock afterwards, and its own re-check finds the item
                    // this thread just pushed. Neither branch parks.
                    if self.receiver_waiter_count.load(Ordering::Relaxed) > 0 {
                        let (_, _, not_empty) = &*self.state;
                        not_empty.notify_one();
                    }
                    return Ok(());
                }
                Err(returned) => {
                    value = returned;
                }
            }

            guard = not_full.wait(guard).unwrap();
            // Relaxed: deregistration, as above.
            self.sender_waiter_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn recv_bounded(&self, queue: &BoundedMpmcQueue<T>) -> Result<T>
    where
        T: Send,
    {
        let mut spin_count = 0;

        loop {
            if let Some(value) = queue.try_pop() {
                // Notifier half of the Dekker pair, mirroring `send_bounded`:
                // lock-free queue write, barrier, then the waiter count. Full
                // reasoning at that site.
                fence(CHANNEL_STORE_LOAD_ORDER);
                if self.sender_waiter_count.load(CHANNEL_STORE_LOAD_ORDER) > 0 {
                    let (mutex, not_full, _) = &*self.state;
                    let _guard = mutex.lock().unwrap();
                    not_full.notify_one();
                }
                return Ok(value);
            }

            if self.closed.load(Ordering::Acquire) {
                if queue.is_empty() {
                    return Err(ChannelError::Closed);
                }
                std::hint::spin_loop();
                continue;
            }

            if spin_count < MPMC_BLOCK_SPINS {
                for _ in 0..(1 << spin_count) {
                    std::hint::spin_loop();
                }
                spin_count += 1;
                continue;
            }

            // Fallback to condvar wait to prevent CPU contention and busy-looping
            let (mutex, _, not_empty) = &*self.state;
            let mut guard = mutex.lock().unwrap();

            // Register *before* the re-check below. Previously the order was
            // inverted here (re-check, then register) while `send_bounded`
            // registered first, and the asymmetry was a lost wakeup: a
            // producer could push and read `receiver_waiter_count == 0`
            // between this thread's failed `try_pop` and its `fetch_add`,
            // skip the notify, and leave this thread parked on a queue that
            // already holds its item. Registering first makes the producer's
            // counter read and this thread's queue read a Dekker pair that
            // `SeqCst` closes.
            //
            // SeqCst, load-bearing: waiter half of the pair, and the
            // Store→Load barrier before the `try_pop` that follows.
            self.receiver_waiter_count
                .fetch_add(1, CHANNEL_STORE_LOAD_ORDER);

            if let Some(value) = queue.try_pop() {
                // Relaxed: deregistration (see `send_bounded`).
                self.receiver_waiter_count.fetch_sub(1, Ordering::Relaxed);
                // Relaxed: this pop and the matching sender registration both
                // happen under the channel mutex, which supplies the edge —
                // a sender that registered earlier is visible through the
                // lock, and one that registers later re-checks the queue slot
                // this pop just freed.
                if self.sender_waiter_count.load(Ordering::Relaxed) > 0 {
                    let (_, not_full, _) = &*self.state;
                    not_full.notify_one();
                }
                drop(guard);
                return Ok(value);
            }

            if self.closed.load(Ordering::Acquire) || guard.closed {
                // Relaxed: deregistration on the error exit.
                self.receiver_waiter_count.fetch_sub(1, Ordering::Relaxed);
                return Err(ChannelError::Closed);
            }

            guard = not_empty.wait(guard).unwrap();
            // Relaxed: deregistration (see `send_bounded`).
            self.receiver_waiter_count.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl<T: Send> Channel<T> for MpmcChannel<T> {
    fn send(&self, value: T) -> Result<()> {
        if let Some(queue) = &self.bounded {
            return self.send_bounded(queue, value);
        }

        let (mutex, not_full, not_empty) = &*self.state;
        let mut guard = mutex.lock().unwrap();
        let mut spin_count = 0;

        while !guard.closed && guard.capacity.is_some_and(|cap| guard.queue.len() >= cap) {
            if spin_count < MPMC_BLOCK_SPINS {
                drop(guard);
                for _ in 0..(1 << spin_count) {
                    std::hint::spin_loop();
                }
                spin_count += 1;
                guard = mutex.lock().unwrap();
            } else {
                self.sender_waiter_count.fetch_add(1, Ordering::AcqRel);
                guard = not_full.wait(guard).unwrap();
                self.sender_waiter_count.fetch_sub(1, Ordering::AcqRel);
            }
        }

        if guard.closed {
            return Err(ChannelError::Closed);
        }

        guard.queue.push_back(value);
        drop(guard);

        if self.receiver_waiter_count.load(Ordering::Acquire) > 0 {
            not_empty.notify_one();
        }
        Ok(())
    }

    fn try_send(&self, value: T) -> Result<()> {
        if let Some(queue) = &self.bounded {
            if self.closed.load(Ordering::Acquire) {
                return Err(ChannelError::Closed);
            }
            queue.try_push(value).map_err(|_| ChannelError::Full)?;
            // SeqCst, load-bearing: lock-free push followed by the waiter
            // read — the notifier half of the Dekker pair (see
            // `send_bounded`). The mutex is not held here, so nothing else
            // orders this Store→Load.
            // Notifier half of the Dekker pair: the lock-free queue op
            // above needs a StoreLoad barrier before the waiter count is
            // read. Full reasoning in `send_bounded`.
            fence(CHANNEL_STORE_LOAD_ORDER);
            if self.receiver_waiter_count.load(CHANNEL_STORE_LOAD_ORDER) > 0 {
                let (mutex, _, not_empty) = &*self.state;
                let _guard = mutex.lock().unwrap();
                not_empty.notify_one();
            }
            return Ok(());
        }

        let (mutex, _, not_empty) = &*self.state;
        let mut guard = mutex.lock().unwrap();

        if guard.closed {
            return Err(ChannelError::Closed);
        }

        if guard.capacity.is_some_and(|cap| guard.queue.len() >= cap) {
            return Err(ChannelError::Full);
        }

        guard.queue.push_back(value);
        drop(guard);

        if self.receiver_waiter_count.load(Ordering::Acquire) > 0 {
            not_empty.notify_one();
        }
        Ok(())
    }

    fn recv(&self) -> Result<T> {
        if let Some(queue) = &self.bounded {
            return self.recv_bounded(queue);
        }

        let (mutex, not_full, not_empty) = &*self.state;
        let mut guard = mutex.lock().unwrap();
        let mut spin_count = 0;

        while guard.queue.is_empty() && !guard.closed {
            if spin_count < MPMC_BLOCK_SPINS {
                drop(guard);
                for _ in 0..(1 << spin_count) {
                    std::hint::spin_loop();
                }
                spin_count += 1;
                guard = mutex.lock().unwrap();
            } else {
                self.receiver_waiter_count.fetch_add(1, Ordering::AcqRel);
                guard = not_empty.wait(guard).unwrap();
                self.receiver_waiter_count.fetch_sub(1, Ordering::AcqRel);
            }
        }

        if let Some(value) = guard.queue.pop_front() {
            drop(guard);

            if self.sender_waiter_count.load(Ordering::Acquire) > 0 {
                not_full.notify_one();
            }
            Ok(value)
        } else {
            Err(ChannelError::Closed)
        }
    }

    fn try_recv(&self) -> Result<T> {
        if let Some(queue) = &self.bounded {
            if let Some(value) = queue.try_pop() {
                // SeqCst, load-bearing: lock-free pop followed by the waiter
                // read — notifier half of the Dekker pair (see
                // `send_bounded`).
                // Notifier half of the Dekker pair: the lock-free queue op
                // above needs a StoreLoad barrier before the waiter count is
                // read. Full reasoning in `send_bounded`.
                fence(CHANNEL_STORE_LOAD_ORDER);
                if self.sender_waiter_count.load(CHANNEL_STORE_LOAD_ORDER) > 0 {
                    let (mutex, not_full, _) = &*self.state;
                    let _guard = mutex.lock().unwrap();
                    not_full.notify_one();
                }
                return Ok(value);
            }
            if self.closed.load(Ordering::Acquire) {
                return Err(ChannelError::Closed);
            }
            return Err(ChannelError::Empty);
        }

        let (mutex, not_full, _) = &*self.state;
        let mut guard = mutex.lock().unwrap();

        if let Some(value) = guard.queue.pop_front() {
            drop(guard);

            if self.sender_waiter_count.load(Ordering::Acquire) > 0 {
                not_full.notify_one();
            }
            Ok(value)
        } else if guard.closed {
            Err(ChannelError::Closed)
        } else {
            Err(ChannelError::Empty)
        }
    }

    fn is_empty(&self) -> bool {
        if let Some(queue) = &self.bounded {
            return queue.is_empty();
        }

        let (mutex, _, _) = &*self.state;
        let guard = mutex.lock().unwrap();
        guard.queue.is_empty()
    }

    fn is_full(&self) -> bool {
        if let Some(queue) = &self.bounded {
            return queue.is_full();
        }

        let (mutex, _, _) = &*self.state;
        let guard = mutex.lock().unwrap();
        guard.capacity.is_some_and(|cap| guard.queue.len() >= cap)
    }

    fn capacity(&self) -> Option<usize> {
        if let Some(queue) = &self.bounded {
            return Some(queue.logical_capacity());
        }

        let (mutex, _, _) = &*self.state;
        let guard = mutex.lock().unwrap();
        guard.capacity
    }
}
