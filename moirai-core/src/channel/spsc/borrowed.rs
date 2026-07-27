//! The borrowed half pair: [`SpscRing::split`].
//!
//! [`super::shared`]'s halves each own an `Arc`, which makes them `'static` and
//! costs one heap allocation for the ring plus an atomic refcount decrement per
//! half. When both halves live inside a known scope — a `thread::scope`, a
//! frame loop, a pipeline stage — that reference counting buys nothing, because
//! the scope already proves the ring outlives them.
//!
//! These halves borrow the ring instead. The pair is allocation-free beyond the
//! ring's own buffer, and dropping a half is a single store rather than a
//! refcount decrement and a conditional deallocation.
//!
//! # Why `split` takes `&mut self`
//!
//! The whole SPSC discipline is "one producer, one consumer". Handing out a
//! second producer would break it, so the exclusive borrow is what forbids a
//! second [`split`](SpscRing::split) while either half is alive: the returned
//! halves hold the borrow, so the ring is frozen until both are dropped. No
//! runtime flag is involved — the same guarantee `Arc`-backed halves get from
//! being non-`Clone`, obtained here from the borrow checker.

use super::ring::{blocking, SpscChannel};
use crate::channel::error::{Channel, Result};
use crate::channel::roles::{Consumer, Producer};
use std::cell::Cell;
use std::sync::atomic::Ordering;

/// A bounded SPSC ring buffer owned in place, split into borrowing halves.
///
/// ```
/// use moirai_core::channel::{Consumer, Producer, SpscRing};
///
/// let mut ring = SpscRing::<u64>::new(64);
/// let (tx, rx) = ring.split();
///
/// std::thread::scope(|scope| {
///     scope.spawn(move || {
///         for value in 0..1000 {
///             if tx.send(value).is_err() {
///                 break;
///             }
///         }
///     });
///
///     let mut sum = 0;
///     for _ in 0..1000 {
///         match rx.recv() {
///             Ok(value) => sum += value,
///             Err(_) => break,
///         }
///     }
///     assert_eq!(sum, (0..1000).sum::<u64>());
/// });
/// ```
pub struct SpscRing<T> {
    channel: SpscChannel<T>,
}

impl<T: Send> SpscRing<T> {
    /// Create a ring with at least `capacity` slots, rounded up to a power of
    /// two so the index-to-slot mapping is a mask rather than a division.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            channel: SpscChannel::new(capacity),
        }
    }

    /// Borrow the ring as a producer and a consumer.
    ///
    /// Both halves borrow `self`, so the ring cannot be split again, moved, or
    /// dropped until they are.
    ///
    /// A ring may be split again once its previous halves are gone. Values left
    /// queued by an earlier round stay queued — the counters are not reset —
    /// which is what makes a ring reusable across phases without reallocating.
    ///
    /// Two details make re-splitting sound, and both are why the caches are
    /// seeded from the live counters rather than from zero:
    ///
    /// - `closed` is cleared. A half sets it on drop so its peer stops
    ///   blocking; leaving it set would make the next round's first operation
    ///   fail immediately.
    /// - The consumer's cache must satisfy `tail <= cached_head <= head`. A
    ///   zeroed `cached_head` against a non-zero `tail` breaks the left side,
    ///   and `has_value` would then report a value present in an empty ring and
    ///   read a slot that was never written. Seeding from `head` restores it;
    ///   seeding the producer's cache from `tail` is exact for the same reason.
    ///
    /// Taking `&mut self` is what makes this safe to do without atomics: no
    /// half exists, so nothing else can be touching either counter.
    pub fn split(&mut self) -> (SpscProducer<'_, T>, SpscConsumer<'_, T>) {
        let (head, tail) = self.channel.indices();
        self.channel.closed.store(false, Ordering::Release);

        (
            SpscProducer {
                channel: &self.channel,
                cached_tail: Cell::new(tail),
            },
            SpscConsumer {
                channel: &self.channel,
                cached_head: Cell::new(head),
            },
        )
    }

    /// Number of values currently queued.
    #[must_use]
    pub fn len(&self) -> usize {
        let (head, tail) = self.channel.indices();
        head.wrapping_sub(tail)
    }

    /// Whether the ring holds no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total slots, after the power-of-two rounding applied at construction.
    #[must_use]
    pub fn capacity(&self) -> usize {
        Channel::capacity(&self.channel).unwrap_or(0)
    }
}

/// The sending half of a borrowed [`SpscRing`].
///
/// `Send` so it can be moved into a scoped thread, and not `Sync` for the same
/// reason as the `Arc`-backed sender: `send` takes `&self`, so two threads
/// sharing one would claim the same slot.
pub struct SpscProducer<'ring, T> {
    channel: &'ring SpscChannel<T>,
    /// Last known consumer index; see [`SpscSender`](super::SpscSender) for why
    /// this both saves a cache-line read and keeps the half `!Sync`.
    cached_tail: Cell<usize>,
}

impl<T: Send> SpscProducer<'_, T> {
    /// Send a value, blocking until there is room.
    ///
    /// # Errors
    ///
    /// [`ChannelError::Closed`](crate::channel::ChannelError::Closed) once the
    /// consumer half has been dropped.
    pub fn send(&self, value: T) -> Result<()> {
        self.channel.send_cached(value, &self.cached_tail)
    }

    /// Send a value, or report
    /// [`ChannelError::Full`](crate::channel::ChannelError::Full) rather than
    /// waiting.
    ///
    /// # Errors
    ///
    /// `Full` when no slot is free, `Closed` once the consumer has been
    /// dropped.
    pub fn try_send(&self, value: T) -> Result<()> {
        self.channel.try_send_cached(value, &self.cached_tail)
    }
}

/// The receiving half of a borrowed [`SpscRing`].
pub struct SpscConsumer<'ring, T> {
    channel: &'ring SpscChannel<T>,
    /// Last known producer index. Mirrors [`SpscProducer::cached_tail`].
    cached_head: Cell<usize>,
}

impl<T: Send> SpscConsumer<'_, T> {
    /// Receive a value, blocking until one arrives.
    ///
    /// # Errors
    ///
    /// `Closed` once the producer has been dropped and the ring is drained.
    pub fn recv(&self) -> Result<T> {
        blocking(|| self.channel.try_recv_cached(&self.cached_head))
    }

    /// Receive a value, or report
    /// [`ChannelError::Empty`](crate::channel::ChannelError::Empty) rather than
    /// waiting.
    ///
    /// # Errors
    ///
    /// `Empty` when nothing is queued, `Closed` once the producer has been
    /// dropped and the ring is drained.
    pub fn try_recv(&self) -> Result<T> {
        self.channel.try_recv_cached(&self.cached_head)
    }
}

impl<T: Send> Producer<T> for SpscProducer<'_, T> {
    #[inline]
    fn send(&self, value: T) -> Result<()> {
        SpscProducer::send(self, value)
    }

    #[inline]
    fn try_send(&self, value: T) -> Result<()> {
        SpscProducer::try_send(self, value)
    }

    #[inline]
    fn is_full(&self) -> bool {
        Channel::is_full(self.channel)
    }

    #[inline]
    fn capacity(&self) -> Option<usize> {
        Channel::capacity(self.channel)
    }
}

impl<T: Send> Consumer<T> for SpscConsumer<'_, T> {
    #[inline]
    fn recv(&self) -> Result<T> {
        SpscConsumer::recv(self)
    }

    #[inline]
    fn try_recv(&self) -> Result<T> {
        SpscConsumer::try_recv(self)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        Channel::is_empty(self.channel)
    }
}

impl<T> Drop for SpscProducer<'_, T> {
    fn drop(&mut self) {
        self.channel.closed.store(true, Ordering::Release);
    }
}

impl<T> Drop for SpscConsumer<'_, T> {
    fn drop(&mut self) {
        self.channel.closed.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod auto_traits {
    use super::{SpscConsumer, SpscProducer, SpscRing};
    use static_assertions::{assert_impl_all, assert_not_impl_any};

    // The ring is handed between threads whole before it is split.
    assert_impl_all!(SpscRing<u64>: Send, Sync);

    // Each half moves to the scoped thread that owns its role.
    assert_impl_all!(SpscProducer<'static, u64>: Send);
    assert_impl_all!(SpscConsumer<'static, u64>: Send);

    /// Neither half may be *shared*, which is what keeps the ring SPSC.
    ///
    /// The `Cell` cache is what withholds `Sync`; changing it to a `Sync` type
    /// would let two threads drive one half and claim the same slot.
    #[allow(dead_code)]
    fn halves_are_not_sync() {
        assert_not_impl_any!(SpscProducer<'static, u64>: Sync);
        assert_not_impl_any!(SpscConsumer<'static, u64>: Sync);
    }
}

#[cfg(test)]
mod tests {
    use super::SpscRing;
    use crate::channel::roles::{Consumer, Producer};
    use crate::channel::ChannelError;

    #[test]
    fn split_halves_transfer_values_in_order() {
        let mut ring = SpscRing::<u64>::new(4);
        let (tx, rx) = ring.split();

        assert!(tx.try_send(1).is_ok());
        assert!(tx.try_send(2).is_ok());
        assert_eq!(rx.try_recv().expect("a value was sent"), 1);
        assert_eq!(rx.try_recv().expect("a value was sent"), 2);
        assert!(matches!(rx.try_recv(), Err(ChannelError::Empty)));
    }

    #[test]
    fn capacity_is_exact_with_no_sacrificed_slot() {
        let mut ring = SpscRing::<u64>::new(4);
        assert_eq!(ring.capacity(), 4);
        let (tx, _rx) = ring.split();

        for value in 0..4 {
            assert!(tx.try_send(value).is_ok(), "slot {value} must be usable");
        }
        assert!(matches!(tx.try_send(4), Err(ChannelError::Full)));
    }

    /// The re-split invariant: a ring reused after its halves drop must not
    /// invent a value.
    ///
    /// Seeding the consumer's cache from zero rather than from `head` breaks
    /// `tail <= cached_head`, and `has_value` then reports a value present in
    /// an empty ring and reads a slot that was never written. This is the test
    /// that fails if `split` stops reading the live counters.
    #[test]
    fn resplitting_a_drained_ring_reports_empty() {
        let mut ring = SpscRing::<u64>::new(4);
        {
            let (tx, rx) = ring.split();
            for value in 0..3 {
                tx.try_send(value).expect("capacity is 4");
            }
            for expected in 0..3 {
                assert_eq!(rx.try_recv().expect("three were sent"), expected);
            }
        }

        // Counters are now at 3, not 0, and the ring is empty.
        assert!(ring.is_empty());
        let (_tx, rx) = ring.split();
        assert!(
            matches!(rx.try_recv(), Err(ChannelError::Empty)),
            "a drained ring must report empty, not read an unwritten slot"
        );
    }

    #[test]
    fn resplitting_preserves_queued_values() {
        let mut ring = SpscRing::<u64>::new(8);
        {
            let (tx, _rx) = ring.split();
            tx.try_send(7).expect("capacity is 8");
            tx.try_send(8).expect("capacity is 8");
        }
        assert_eq!(ring.len(), 2);

        let (_tx, rx) = ring.split();
        assert_eq!(rx.try_recv().expect("queued across the split"), 7);
        assert_eq!(rx.try_recv().expect("queued across the split"), 8);
    }

    /// Dropping a half must release a peer blocked in the other, or a scoped
    /// thread would never join.
    #[test]
    fn dropping_the_producer_closes_the_consumer() {
        let mut ring = SpscRing::<u64>::new(4);
        let (tx, rx) = ring.split();
        tx.try_send(1).expect("capacity is 4");
        drop(tx);

        assert_eq!(rx.recv().expect("the queued value survives closure"), 1);
        assert!(matches!(rx.recv(), Err(ChannelError::Closed)));
    }

    /// The point of the roles: borrowed halves are usable by the same generic
    /// code as the `Arc`-backed ones, with no `'static` bound in sight.
    #[test]
    fn borrowed_halves_satisfy_the_roles() {
        fn drain_into<P: Producer<u64>>(producer: &P, values: &[u64]) -> usize {
            values
                .iter()
                .filter(|v| producer.try_send(**v).is_ok())
                .count()
        }
        fn take_all<C: Consumer<u64>>(consumer: &C, limit: usize) -> Vec<u64> {
            (0..limit)
                .filter_map(|_| consumer.try_recv().ok())
                .collect()
        }

        let mut ring = SpscRing::<u64>::new(8);
        let (tx, rx) = ring.split();

        assert_eq!(drain_into(&tx, &[1, 2, 3]), 3);
        assert_eq!(Producer::capacity(&tx), Some(8));
        assert!(!Consumer::is_empty(&rx));
        assert_eq!(take_all(&rx, 8), vec![1, 2, 3]);
    }

    /// Values left in a ring that is dropped without being drained must still
    /// be dropped exactly once.
    #[test]
    fn dropping_a_loaded_ring_drops_each_value_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct Counted(Arc<AtomicUsize>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        {
            let mut ring = SpscRing::<Counted>::new(4);
            let (tx, _rx) = ring.split();
            for _ in 0..3 {
                tx.try_send(Counted(Arc::clone(&drops)))
                    .expect("capacity is 4");
            }
        }

        assert_eq!(
            drops.load(Ordering::Relaxed),
            3,
            "every queued value is dropped exactly once with the ring"
        );
    }

    #[test]
    fn scoped_threads_move_the_halves_across_the_boundary() {
        const COUNT: u64 = 10_000;

        let mut ring = SpscRing::<u64>::new(64);
        let (tx, rx) = ring.split();

        let sum = std::thread::scope(|scope| {
            scope.spawn(move || {
                for value in 0..COUNT {
                    if tx.send(value).is_err() {
                        break;
                    }
                }
            });

            let mut sum = 0_u64;
            for _ in 0..COUNT {
                match rx.recv() {
                    Ok(value) => sum = sum.wrapping_add(value),
                    Err(_) => break,
                }
            }
            sum
        });

        assert_eq!(sum, (0..COUNT).sum::<u64>());
    }
}
