//! Lock-free Single Producer Single Consumer channel.
//!
//! Optimized for low latency with zero-copy semantics.

use super::error::{CacheAligned, Channel, ChannelError, Result};
use std::cell::UnsafeCell;
use std::marker::PhantomData;
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
                _marker: PhantomData,
            },
            SpscReceiver {
                channel,
                _marker: PhantomData,
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
    /// Makes the sender `!Sync` while leaving it `Send`.
    ///
    /// `send` takes `&self`, so a `Sync` sender could be shared and driven by
    /// two threads at once — both would claim the same slot. `Cell<()>` is
    /// `Send` but not `Sync`, which permits moving the sender to another thread
    /// and forbids sharing it. Removing this marker reopens the race;
    /// `halves_are_not_sync` fails if it goes.
    _marker: PhantomData<std::cell::Cell<()>>,
}

impl<T: Send> SpscSender<T> {
    /// Send a value through the channel, blocking if necessary
    pub fn send(&self, value: T) -> Result<()> {
        self.channel.send(value)
    }

    /// Try to send a value without blocking
    pub fn try_send(&self, value: T) -> Result<()> {
        self.channel.try_send(value)
    }
}

/// Receiver half of SPSC channel.
///
/// The single-consumer half of the pair, `!Sync` for the same reason as
/// [`SpscSender`] — two shared receivers would `assume_init_read` one slot
/// twice, moving out of it and then dropping it twice.
pub struct SpscReceiver<T> {
    pub(super) channel: Arc<SpscChannel<T>>,
    /// Makes the receiver `!Sync` while leaving it `Send`. See [`SpscSender`].
    _marker: PhantomData<std::cell::Cell<()>>,
}

impl<T: Send> SpscReceiver<T> {
    /// Receive a value from the channel, blocking if necessary
    pub fn recv(&self) -> Result<T> {
        self.channel.recv()
    }

    /// Try to receive a value without blocking
    pub fn try_recv(&self) -> Result<T> {
        self.channel.try_recv()
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
