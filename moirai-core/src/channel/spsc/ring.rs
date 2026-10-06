//! The bounded ring's channel wrapper: the policy layer over
//! [`RingBuffer`](crate::communication::RingBuffer).
//!
//! The storage, the two cursors, the publication algebra, and the drain on drop
//! live on [`RingBuffer`] itself, which this type composes. What is left here is
//! what is *discipline* rather than mechanism: the cached-index layer the halves
//! drive, the `closed` flag, and the spin-then-yield schedule. One Lamport
//! protocol, one implementation (ADR-016 item 3).
//!
//! Nothing here decides *who* may send or receive. That is the job of the
//! wrappers in [`shared`](super::shared) and [`borrowed`](super::borrowed), which
//! is why [`SpscChannel`] stays crate-private: its methods take `&self` and its
//! `Sync` impl lets `&SpscChannel` cross threads, so exposing it would let safe
//! code drive two producers into one slot (ADR-024).

use crate::channel::error::{Channel, ChannelError, Result};
use crate::communication::RingBuffer;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

/// Exponential-backoff spin rounds (`1 << round` spin-loop hints per round,
/// ~63 total hints) before a blocked send/recv falls back to
/// `thread::yield_now`. Tuned for this channel's yield-based slow path;
/// intentionally local rather than crate-wide because MPMC uses a larger budget
/// matched to its condvar fallback.
const SPSC_BLOCK_SPINS: usize = 6;

/// Lock-free single-producer/single-consumer channel.
///
/// Deliberately crate-private. `Channel::send`/`recv` take `&self`, and the
/// `Sync` impl below lets `&SpscChannel` cross threads, so exposing the bare
/// channel would let safe code drive two producers into the same slot. The
/// discipline is enforced instead by [`SpscSender`]/[`SpscReceiver`], which are
/// neither `Clone` nor `Sync`; reach them through `channel::spsc`.
pub(crate) struct SpscChannel<T> {
    /// The ring: storage, cursors, and the publication algebra.
    ring: RingBuffer<T>,
    /// Channel state.
    ///
    /// Written by whichever half drops first, so a peer blocked in `send` or
    /// `recv` stops waiting; read on every operation.
    pub(super) closed: AtomicBool,
}

// SAFETY: the ring hands a value from one half to the other — the producer writes
// a slot then releases its cursor, the consumer acquires that cursor before
// reading the slot — so no slot is accessed by two threads at once and `T: Send`
// is the exact bound for moving values across the pair.
//
// `Sync` here is what lets one `Arc<SpscChannel>` back both halves, and it is
// sound only because the halves impose the one-of-each discipline. That is why
// the type is crate-private: on the bare channel, `&self` methods plus `Sync`
// would let any number of threads produce at once. It is also why the ring itself
// withholds `Sync` and asserts only `Send` — the discipline is this wrapper's to
// make, so this wrapper is where the assertion belongs.
unsafe impl<T: Send> Send for SpscChannel<T> {}
unsafe impl<T: Send> Sync for SpscChannel<T> {}

impl<T> SpscChannel<T> {
    /// Create a new SPSC channel with given capacity (rounded up to power of 2)
    pub fn new(capacity: usize) -> Self {
        Self {
            // `.max(2)` is this channel's bound, not the ring's: the sequence
            // protocol needs at least two slots to tell a slot's empty
            // generation from its full one.
            ring: RingBuffer::new(capacity.next_power_of_two().max(2)),
            closed: AtomicBool::new(false),
        }
    }

    /// The producer and consumer counters, in that order, read without
    /// synchronization. See [`RingBuffer::indices`] for why `Relaxed` is right
    /// here and nowhere else.
    pub(super) fn indices(&self) -> (usize, usize) {
        self.ring.indices()
    }
}

/// One step of the spin-then-yield schedule shared by the blocking paths.
#[inline]
fn back_off(spin: &mut usize) {
    if *spin < SPSC_BLOCK_SPINS {
        for _ in 0..(1 << *spin) {
            std::hint::spin_loop();
        }
        *spin += 1;
    } else {
        std::thread::yield_now();
    }
}

/// Retry `attempt` on the spin-then-yield schedule until it resolves to
/// something other than a transiently full or empty queue.
pub(super) fn blocking<F, R>(mut attempt: F) -> Result<R>
where
    F: FnMut() -> Result<R>,
{
    let mut spin = 0;
    loop {
        match attempt() {
            Err(ChannelError::Full | ChannelError::Empty) => back_off(&mut spin),
            other => return other,
        }
    }
}

impl<T: Send> SpscChannel<T> {
    pub(super) fn try_send_cached(&self, value: T, cached_tail: &Cell<usize>) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(ChannelError::Closed);
        }

        let head = self.ring.producer_relaxed();
        if !self.ring.has_room(head, cached_tail) {
            return Err(ChannelError::Full);
        }

        // SAFETY: `has_room` established that `head` is at or beyond the
        // consumer's cursor, and only this half writes slots.
        unsafe { self.ring.produce_at(head, value) };
        Ok(())
    }

    pub(super) fn try_recv_cached(&self, cached_head: &Cell<usize>) -> Result<T> {
        let tail = self.ring.consumer_relaxed();

        if !self.ring.has_value(tail, cached_head) {
            if self.closed.load(Ordering::Acquire) {
                // The sender publishes the element before it publishes closure,
                // so re-read the producer cursor after observing `closed`: the
                // check above may have preceded both releases and seen an empty
                // ring.
                let published = self.ring.producer_acquire();
                cached_head.set(published);
                if tail == published {
                    return Err(ChannelError::Closed);
                }
            } else {
                return Err(ChannelError::Empty);
            }
        }

        // SAFETY: `has_value` established (or the re-check confirmed) that `tail`
        // is behind the published producer cursor, and only this half advances it.
        Ok(unsafe { self.ring.consume_at(tail) })
    }

    /// Blocking send.
    ///
    /// Written as its own loop rather than through [`Self::blocking`] because
    /// `value` must survive a failed attempt: the cached send takes it by value,
    /// so a closure would move it on the first iteration. Here it stays owned by
    /// this frame and is moved exactly once, when a slot is claimed.
    pub(super) fn send_cached(&self, value: T, cached_tail: &Cell<usize>) -> Result<()> {
        let mut spin = 0;
        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(ChannelError::Closed);
            }

            let head = self.ring.producer_relaxed();
            if self.ring.has_room(head, cached_tail) {
                // SAFETY: as `try_send_cached` — the slot is past the consumer's
                // cursor and only this half writes slots.
                unsafe { self.ring.produce_at(head, value) };
                return Ok(());
            }

            back_off(&mut spin);
        }
    }
}

impl<T: Send> Channel<T> for SpscChannel<T> {
    /// Blocking send.
    ///
    /// The uncached `Channel` surface drives the *same* cached primitives the
    /// halves do, with the cache seeded from a fresh peer-cursor load. Seeding is
    /// what makes this exact: the value is the one the uncached path would have
    /// read anyway, so the first check is the check it always was, and
    /// [`RingBuffer::has_room`]'s one-sided staleness covers the rest of a
    /// blocking call exactly as it covers `SpscSender::send`. The slot write and
    /// the spin-then-yield schedule therefore exist once each, not per entry
    /// point, and no caller can reach a weaker variant of either.
    fn send(&self, value: T) -> Result<()> {
        let cached_tail = Cell::new(self.ring.consumer_acquire());
        self.send_cached(value, &cached_tail)
    }

    fn try_send(&self, value: T) -> Result<()> {
        let cached_tail = Cell::new(self.ring.consumer_acquire());
        self.try_send_cached(value, &cached_tail)
    }

    fn recv(&self) -> Result<T> {
        let cached_head = Cell::new(self.ring.producer_acquire());
        blocking(|| self.try_recv_cached(&cached_head))
    }

    fn try_recv(&self) -> Result<T> {
        let cached_head = Cell::new(self.ring.producer_acquire());
        self.try_recv_cached(&cached_head)
    }

    fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    fn is_full(&self) -> bool {
        self.ring.is_full()
    }

    fn capacity(&self) -> Option<usize> {
        Some(self.ring.capacity())
    }
}
