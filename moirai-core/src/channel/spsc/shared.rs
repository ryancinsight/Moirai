//! The `Arc`-backed half pair: `spsc(capacity)`.
//!
//! Each half owns a reference-counted handle to the ring, so both are
//! `'static` and can be moved into freshly spawned threads. The cost is one
//! allocation for the ring plus an atomic refcount decrement per half at
//! drop; [`borrowed`](super::borrowed) trades that for a scope.

use super::ring::{blocking, SpscChannel};
use crate::channel::error::{Channel, Result};
use crate::channel::roles::{Consumer, Producer};
use std::cell::Cell;
use std::sync::atomic::Ordering;
use std::sync::Arc;

impl<T> SpscChannel<T> {
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

impl<T: Send> Producer<T> for SpscSender<T> {
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

impl<T: Send> Consumer<T> for SpscReceiver<T> {
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
    /// consumers would read one slot twice. The `Cell` cache on each half is
    /// what prevents it, so changing that field to a `Sync` type would reopen
    /// the race — this assertion is what catches it.
    #[allow(dead_code)]
    fn halves_are_not_sync() {
        assert_not_impl_any!(SpscSender<u64>: Sync);
        assert_not_impl_any!(SpscReceiver<u64>: Sync);
    }
}
