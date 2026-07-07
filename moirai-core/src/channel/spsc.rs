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

/// Lock-free Single Producer Single Consumer channel
/// Optimized for low latency with zero-copy semantics
pub struct SpscChannel<T> {
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
                return Err(ChannelError::Closed);
            }
            return Err(ChannelError::Empty);
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

/// Sender half of SPSC channel
pub struct SpscSender<T> {
    pub(super) channel: Arc<SpscChannel<T>>,
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

/// Receiver half of SPSC channel
pub struct SpscReceiver<T> {
    pub(super) channel: Arc<SpscChannel<T>>,
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
