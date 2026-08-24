//! The bounded ring itself: storage, the two counters, and the cached-index
//! primitives both half-pair flavours drive.
//!
//! Nothing here decides *who* may send or receive. That is the job of the
//! wrappers in [`shared`](super::shared) and [`borrowed`](super::borrowed),
//! which is why [`SpscChannel`] stays crate-private: its methods take `&self`
//! and its `Sync` impl lets `&SpscChannel` cross threads, so exposing it would
//! let safe code drive two producers into one slot (ADR-024).

use crate::channel::error::{CacheAligned, Channel, ChannelError, Result};
use std::cell::{Cell, UnsafeCell};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Exponential-backoff spin rounds (`1 << round` spin-loop hints per round,
/// ~63 total hints) before a blocked send/recv falls back to
/// `thread::yield_now`. Tuned for this channel's yield-based slow path;
/// intentionally local rather than crate-wide because MPMC uses a larger
/// budget matched to its condvar fallback.
const SPSC_BLOCK_SPINS: usize = 6;

/// Lock-free single-producer/single-consumer channel.
///
/// Deliberately crate-private. `Channel::send`/`recv` take `&self`, and the
/// `Sync` impl below lets `&SpscChannel` cross threads, so exposing the bare
/// channel would let safe code drive two producers into the same slot. The
/// discipline is enforced instead by [`SpscSender`]/[`SpscReceiver`], which are
/// neither `Clone` nor `Sync`; reach them through `channel::spsc`.
pub(crate) struct SpscChannel<T> {
    /// Ring buffer for messages (owns the `T` storage)
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Capacity mask for fast modulo
    mask: usize,
    /// Producer position (cache-aligned)
    head: CacheAligned<AtomicUsize>,
    /// Consumer position (cache-aligned)
    tail: CacheAligned<AtomicUsize>,
    /// Channel state.
    ///
    /// Written by whichever half drops first, so a peer blocked in `send` or
    /// `recv` stops waiting; read on every operation.
    pub(super) closed: AtomicBool,
}

// SAFETY: the buffer is only ever touched by one producer and one consumer,
// each owning its own index — the producer writes a slot then releases `head`,
// the consumer acquires `head` before reading it — so no slot is accessed by
// two threads at once and `T: Send` is the exact bound for handing values
// across the pair.
//
// `Sync` here is what lets one `Arc<SpscChannel>` back both halves, and it is
// sound only because the halves impose the one-of-each discipline. That is why
// the type is crate-private: on the bare channel, `&self` methods plus `Sync`
// would let any number of threads produce at once.
unsafe impl<T: Send> Send for SpscChannel<T> {}
unsafe impl<T: Send> Sync for SpscChannel<T> {}

impl<T> SpscChannel<T> {
    /// Create a new SPSC channel with given capacity (rounded up to power of 2)
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two().max(2);
        let buffer = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            buffer,
            mask: capacity - 1,
            head: CacheAligned::new(AtomicUsize::new(0)),
            tail: CacheAligned::new(AtomicUsize::new(0)),
            closed: AtomicBool::new(false),
        }
    }
}

impl<T> SpscChannel<T> {
    /// The producer and consumer counters, in that order, read without
    /// synchronization.
    ///
    /// `Relaxed` is correct only because every caller holds the ring
    /// exclusively: [`SpscRing`](super::SpscRing)'s methods take `&self` or
    /// `&mut self`, and its halves borrow it, so no half can exist — and
    /// therefore no other thread can be advancing either counter — while this
    /// runs. Reading these from a live half would need the acquire loads the
    /// send and receive paths use.
    pub(super) fn indices(&self) -> (usize, usize) {
        (
            self.head.0.load(Ordering::Relaxed),
            self.tail.0.load(Ordering::Relaxed),
        )
    }
}

impl<T> Drop for SpscChannel<T> {
    fn drop(&mut self) {
        let tail = self.tail.0.load(Ordering::Relaxed);
        let head = self.head.0.load(Ordering::Relaxed);
        let len = head.wrapping_sub(tail);
        for i in 0..len {
            // SAFETY: exclusive `&mut self` in drop; indices span exactly
            // the live window tail..head, each holding a value that has not
            // been consumed, dropped once here.
            unsafe {
                let slot = &mut *self.buffer[(tail.wrapping_add(i)) & self.mask].get();
                slot.assume_init_drop();
            }
        }
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
    /// Room for one more value, consulting `cached_tail` before the consumer's
    /// real index.
    ///
    /// The cached index is always at or behind the true one, because only the
    /// consumer advances `tail` and it only moves forward. A stale value
    /// therefore makes the queue look *fuller* than it is, never emptier, so
    /// this may take the slow path unnecessarily but can never report space that
    /// does not exist. That one-sidedness is what makes the cache sound.
    #[inline]
    pub(super) fn has_room(&self, head: usize, cached_tail: &Cell<usize>) -> bool {
        if head.wrapping_sub(cached_tail.get()) < self.buffer.len() {
            return true;
        }
        // The cache says full; consult the consumer and try once more. This is
        // the only load that touches the consumer's cache line.
        let tail = self.tail.0.load(Ordering::Acquire);
        cached_tail.set(tail);
        head.wrapping_sub(tail) < self.buffer.len()
    }

    /// A value is available, consulting `cached_head` before the producer's real
    /// index. Mirrors [`Self::has_room`]: a stale cache understates what is
    /// queued, so it can cost an extra load but never invent an element.
    #[inline]
    pub(super) fn has_value(&self, tail: usize, cached_head: &Cell<usize>) -> bool {
        if tail != cached_head.get() {
            return true;
        }
        let head = self.head.0.load(Ordering::Acquire);
        cached_head.set(head);
        tail != head
    }

    pub(super) fn try_send_cached(&self, value: T, cached_tail: &Cell<usize>) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(ChannelError::Closed);
        }

        let head = self.head.0.load(Ordering::Relaxed);
        if !self.has_room(head, cached_tail) {
            return Err(ChannelError::Full);
        }

        // SAFETY: `head` is at or beyond the consumer's index, so this slot is
        // not one the consumer may read until the release store below publishes
        // it, and only this producer writes slots.
        unsafe {
            let slot = &mut *self.buffer[head & self.mask].get();
            slot.write(value);
        }
        self.head.0.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    pub(super) fn try_recv_cached(&self, cached_head: &Cell<usize>) -> Result<T> {
        let tail = self.tail.0.load(Ordering::Relaxed);

        if !self.has_value(tail, cached_head) {
            if self.closed.load(Ordering::Acquire) {
                // The sender publishes the element before it publishes closure,
                // so re-read `head` after observing `closed`: the load above may
                // have preceded both releases and seen an empty queue.
                let published = self.head.0.load(Ordering::Acquire);
                cached_head.set(published);
                if tail == published {
                    return Err(ChannelError::Closed);
                }
            } else {
                return Err(ChannelError::Empty);
            }
        }

        // SAFETY: `tail` is behind the published producer index, so this slot was
        // written and released by the producer. It has not been read before —
        // `tail` advances once per value, and only this consumer advances it.
        let value = unsafe {
            let slot = &*self.buffer[tail & self.mask].get();
            slot.assume_init_read()
        };
        self.tail.0.store(tail.wrapping_add(1), Ordering::Release);
        Ok(value)
    }

    /// Blocking send.
    ///
    /// Written as its own loop rather than through [`Self::blocking`] because
    /// `value` must survive a failed attempt: `try_send_cached` takes it by
    /// value, so a closure would move it on the first iteration. Here it stays
    /// owned by this frame and is moved exactly once, when a slot is claimed.
    pub(super) fn send_cached(&self, value: T, cached_tail: &Cell<usize>) -> Result<()> {
        let mut spin = 0;
        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(ChannelError::Closed);
            }

            let head = self.head.0.load(Ordering::Relaxed);
            if self.has_room(head, cached_tail) {
                // SAFETY: as `try_send_cached` — the slot is past the consumer's
                // index and only this producer writes slots.
                unsafe {
                    let slot = &mut *self.buffer[head & self.mask].get();
                    slot.write(value);
                }
                self.head.0.store(head.wrapping_add(1), Ordering::Release);
                return Ok(());
            }

            back_off(&mut spin);
        }
    }
}

impl<T: Send> Channel<T> for SpscChannel<T> {
    fn send(&self, value: T) -> Result<()> {
        // Implement blocking send with exponential backoff spin-wait
        let mut spin_count = 0;
        loop {
            // Check if channel is closed first
            if self.closed.load(Ordering::Acquire) {
                return Err(ChannelError::Closed);
            }

            let head = self.head.0.load(Ordering::Relaxed);
            let tail = self.tail.0.load(Ordering::Acquire);

            // Check if there's space
            if head.wrapping_sub(tail) < self.buffer.len() {
                // SAFETY: sole-producer role of this blocking sender plus
                // the space check keep the masked slot outside the consumer
                // window and uninitialized before this write.
                unsafe {
                    let slot = &mut *self.buffer[head & self.mask].get();
                    slot.write(value);
                }
                self.head.0.store(head.wrapping_add(1), Ordering::Release);
                return Ok(());
            }

            // Channel is full, spin-wait with exponential backoff
            if spin_count < SPSC_BLOCK_SPINS {
                // Active spinning for low latency (up to ~64 iterations)
                for _ in 0..(1 << spin_count) {
                    std::hint::spin_loop();
                }
                spin_count += 1;
            } else {
                // After initial spinning, yield to OS scheduler
                std::thread::yield_now();
            }
        }
    }

    fn try_send(&self, value: T) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(ChannelError::Closed);
        }

        let head = self.head.0.load(Ordering::Relaxed);
        let tail = self.tail.0.load(Ordering::Acquire);

        // Check if full
        if head.wrapping_sub(tail) >= self.buffer.len() {
            return Err(ChannelError::Full);
        }

        // SAFETY: sole-producer role plus the fullness check guarantee an
        // uninitialized, unconsumed masked slot for this write.
        unsafe {
            let slot = &mut *self.buffer[head & self.mask].get();
            slot.write(value);
        }

        self.head.0.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    fn recv(&self) -> Result<T> {
        // Implement blocking recv with exponential backoff spin-wait
        let mut spin_count = 0;
        loop {
            match self.try_recv() {
                Ok(value) => return Ok(value),
                Err(ChannelError::Empty) => {
                    // Channel is empty, spin-wait with exponential backoff
                    if spin_count < SPSC_BLOCK_SPINS {
                        // Active spinning for low latency (up to ~64 iterations)
                        for _ in 0..(1 << spin_count) {
                            std::hint::spin_loop();
                        }
                        spin_count += 1;
                    } else {
                        // After initial spinning, yield to OS scheduler
                        std::thread::yield_now();
                    }
                }
                Err(e) => return Err(e), // Closed or other error
            }
        }
    }

    fn try_recv(&self) -> Result<T> {
        let tail = self.tail.0.load(Ordering::Relaxed);
        let head = self.head.0.load(Ordering::Acquire);

        if tail == head {
            if self.closed.load(Ordering::Acquire) {
                // The sender publishes the element before it publishes closure.
                // Re-read `head` after acquiring `closed`: the first `head` load
                // may have preceded both releases and observed the empty state.
                let published_head = self.head.0.load(Ordering::Acquire);
                if tail == published_head {
                    return Err(ChannelError::Closed);
                }
            } else {
                return Err(ChannelError::Empty);
            }
        }

        let value = unsafe {
            let slot = &*self.buffer[tail & self.mask].get();
            // SAFETY: head > tail check ensures initialized data
            slot.assume_init_read()
        };

        self.tail.0.store(tail.wrapping_add(1), Ordering::Release);
        Ok(value)
    }

    fn is_empty(&self) -> bool {
        let tail = self.tail.0.load(Ordering::Relaxed);
        let head = self.head.0.load(Ordering::Acquire);
        tail == head
    }

    fn is_full(&self) -> bool {
        let head = self.head.0.load(Ordering::Relaxed);
        let tail = self.tail.0.load(Ordering::Acquire);
        head.wrapping_sub(tail) >= self.buffer.len()
    }

    fn capacity(&self) -> Option<usize> {
        Some(self.buffer.len())
    }
}
