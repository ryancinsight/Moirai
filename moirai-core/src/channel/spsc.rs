//! Lock-free Single Producer Single Consumer channel.
//!
//! Optimized for low latency with zero-copy semantics.

use super::error::{CacheAligned, Channel, ChannelError, Result};
use std::cell::{Cell, UnsafeCell};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

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
    /// Channel state
    closed: AtomicBool,
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

    /// Create a channel pair (sender, receiver) for ergonomic usage
    pub fn channel(capacity: usize) -> (SpscSender<T>, SpscReceiver<T>) {
        let channel = Arc::new(Self::new(capacity));
        (
            SpscSender {
                channel: channel.clone(),
                cached_tail: Cell::new(0),
            },
            SpscReceiver {
                channel,
                cached_head: Cell::new(0),
            },
        )
    }
}

impl<T> Drop for SpscChannel<T> {
    fn drop(&mut self) {
        let tail = self.tail.0.load(Ordering::Relaxed);
        let head = self.head.0.load(Ordering::Relaxed);
        let len = head.wrapping_sub(tail);
        for i in 0..len {
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
fn blocking<F, R>(mut attempt: F) -> Result<R>
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
    fn has_room(&self, head: usize, cached_tail: &Cell<usize>) -> bool {
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
    fn has_value(&self, tail: usize, cached_head: &Cell<usize>) -> bool {
        if tail != cached_head.get() {
            return true;
        }
        let head = self.head.0.load(Ordering::Acquire);
        cached_head.set(head);
        tail != head
    }

    fn try_send_cached(&self, value: T, cached_tail: &Cell<usize>) -> Result<()> {
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

    fn try_recv_cached(&self, cached_head: &Cell<usize>) -> Result<T> {
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
    fn send_cached(&self, value: T, cached_tail: &Cell<usize>) -> Result<()> {
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
                // There's space, try to send
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

/// Sender half of SPSC channel.
///
/// The single-producer half of the pair: not `Clone`, so only one exists, and
/// not `Sync`, so it cannot be shared while it exists.
pub struct SpscSender<T> {
    pub(super) channel: Arc<SpscChannel<T>>,
    /// Last known consumer index, and the reason the sender is `!Sync`.
    ///
    /// Reading the real `tail` on every send touches the consumer's cache line
    /// each time, which dominates the cost of a queue that is otherwise two
    /// loads and a store. Consulting this first means a producer that is not
    /// hitting a full queue never reads the consumer's line at all.
    ///
    /// It doubles as the marker keeping the sender unshareable: `send` takes
    /// `&self`, so a `Sync` sender could be driven by two threads at once, both
    /// claiming the same slot. `Cell` is `Send` but not `Sync`, which permits
    /// moving the sender to another thread while forbidding sharing it.
    /// Replacing it with something `Sync` reopens that race, and
    /// `halves_are_not_sync` fails if it happens.
    cached_tail: Cell<usize>,
}

impl<T: Send> SpscSender<T> {
    /// Send a value through the channel, blocking until there is room.
    pub fn send(&self, value: T) -> Result<()> {
        self.channel.send_cached(value, &self.cached_tail)
    }

    /// Try to send a value without blocking.
    pub fn try_send(&self, value: T) -> Result<()> {
        self.channel.try_send_cached(value, &self.cached_tail)
    }
}

/// Receiver half of SPSC channel.
///
/// The single-consumer half of the pair, `!Sync` for the same reason as
/// [`SpscSender`] — two shared receivers would `assume_init_read` one slot
/// twice, moving out of it and then dropping it twice.
pub struct SpscReceiver<T> {
    pub(super) channel: Arc<SpscChannel<T>>,
    /// Last known producer index, and the reason the receiver is `!Sync`.
    /// Mirrors `SpscSender::cached_tail` in both roles.
    cached_head: Cell<usize>,
}

impl<T: Send> SpscReceiver<T> {
    /// Receive a value from the channel, blocking until one arrives.
    pub fn recv(&self) -> Result<T> {
        blocking(|| self.channel.try_recv_cached(&self.cached_head))
    }

    /// Try to receive a value without blocking.
    pub fn try_recv(&self) -> Result<T> {
        self.channel.try_recv_cached(&self.cached_head)
    }
}

impl<T: Send> crate::channel::roles::Producer<T> for SpscSender<T> {
    #[inline]
    fn send(&self, value: T) -> Result<()> {
        SpscSender::send(self, value)
    }

    #[inline]
    fn try_send(&self, value: T) -> Result<()> {
        SpscSender::try_send(self, value)
    }

    #[inline]
    fn is_full(&self) -> bool {
        Channel::is_full(&*self.channel)
    }

    #[inline]
    fn capacity(&self) -> Option<usize> {
        Channel::capacity(&*self.channel)
    }
}

impl<T: Send> crate::channel::roles::Consumer<T> for SpscReceiver<T> {
    #[inline]
    fn recv(&self) -> Result<T> {
        SpscReceiver::recv(self)
    }

    #[inline]
    fn try_recv(&self) -> Result<T> {
        SpscReceiver::try_recv(self)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        Channel::is_empty(&*self.channel)
    }
}

impl<T> Drop for SpscSender<T> {
    fn drop(&mut self) {
        self.channel.closed.store(true, Ordering::Release);
    }
}

impl<T> Drop for SpscReceiver<T> {
    fn drop(&mut self) {
        self.channel.closed.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod auto_traits {
    use super::{SpscReceiver, SpscSender};
    use static_assertions::{assert_impl_all, assert_not_impl_any};

    // Each half may be moved to the thread that owns its role.
    assert_impl_all!(SpscSender<u64>: Send);
    assert_impl_all!(SpscReceiver<u64>: Send);

    /// Neither half may be *shared*, which is what keeps the channel SPSC.
    ///
    /// Both `send` and `recv` take `&self`, so a `Sync` half could be driven by
    /// two threads at once: two producers would write the same slot, and two
    /// consumers would read one slot twice. The `PhantomData<Cell<()>>` on each
    /// half is the only thing preventing that, and deleting it looks like
    /// removing an unused field — this assertion is what catches it.
    #[allow(dead_code)]
    fn halves_are_not_sync() {
        assert_not_impl_any!(SpscSender<u64>: Sync);
        assert_not_impl_any!(SpscReceiver<u64>: Sync);
    }
}
