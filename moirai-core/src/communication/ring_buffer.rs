use moirai_utils::cache::CacheAligned;
use std::cell::{Cell, UnsafeCell};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Zero-copy ring buffer for high-throughput streaming
///
/// # Safety
///
/// This structure uses `MaybeUninit` for zero-copy performance:
/// - Values are written with `write()` before incrementing `producer_seq`
/// - The `assume_init_read()` in `try_consume()` is safe because we check
///   that `producer_seq` > current, ensuring data was written
///
/// # Why this ring also backs the SPSC channel
///
/// [`SpscRing`](crate::channel::SpscRing) and `channel::spsc`'s halves are backed
/// by this type. There is one Lamport protocol here — a masked slot array, a
/// `Relaxed` load of the owner's own cursor, an `Acquire` load of the peer's, a
/// `Release` store back, and the element written in between — reached through two
/// access disciplines:
///
/// - This type is public, exposes the uncached entry points, and stays `!Sync`
///   (see the `Send` impl below). Its `&self` methods mutate through
///   `UnsafeCell`, so a shared `&RingBuffer` would let two safe threads race one
///   end of the ring.
/// - `channel::spsc::SpscChannel` is crate-private and *is* `Sync`, because its
///   `Arc` must be `Send` to back `'static` halves. Its safety argument is the
///   non-`Clone` halves plus crate-private reach (ADR-024), which a public type
///   cannot invoke. It adds the cached-index layer, a `closed` flag, and the
///   spin-then-yield policy.
///
/// The *bound* is not shared: granting `Sync` here would be unsound for
/// downstream users, and a public type cannot invoke the channel's argument. The
/// *code* is shared instead. The cached primitives live on this type as
/// `pub(crate)` methods, so the discipline stays on the wrapper that can enforce
/// it while the storage, the cursors, and the publication algebra exist exactly
/// once (ADR-016 item 3).
pub struct RingBuffer<T> {
    /// Buffer storage
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Capacity mask for fast modulo
    mask: usize,
    /// Producer sequence number
    producer_seq: CacheAligned<AtomicUsize>,
    /// Consumer sequence number
    consumer_seq: CacheAligned<AtomicUsize>,
}

// SAFETY: the ring owns its `T` values inside `UnsafeCell<MaybeUninit<T>>`, so it
// may move between threads exactly when `T: Send`. It is deliberately NOT `Sync`:
// concurrent shared access is only sound under the single-producer/single-consumer
// discipline (producer touches `producer_seq` + tail slots, consumer touches
// `consumer_seq` + head slots, never the same slot), which is enforced by the
// non-`Clone` `HybridSender`/`HybridReceiver` halves rather than by the type
// system here. Granting `Sync` would permit two producers (or two consumers) to
// race the same end, so it is intentionally withheld.
unsafe impl<T: Send> Send for RingBuffer<T> {}

/// Outcome of a successful [`RingBuffer::try_produce`].
///
/// A consumer parks only after observing an empty ring, so only the
/// empty→non-empty transition can race its registration: a produce into an
/// already-occupied ring finds the next consumer consuming instead of parking.
/// A producer-side wake gate can therefore take its fence and its wake on
/// [`Self::BecameNonEmpty`] alone — the same shape as the bounded MPMC
/// channel's notifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProduceOutcome {
    /// The ring held nothing: this produce took it from empty to non-empty.
    BecameNonEmpty,
    /// The ring already held items, so no consumer can have parked against this
    /// produce.
    AlreadyOccupied,
}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let buffer = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            buffer,
            mask: capacity - 1,
            producer_seq: CacheAligned::new(AtomicUsize::new(0)),
            consumer_seq: CacheAligned::new(AtomicUsize::new(0)),
        }
    }

    /// Try to produce a value, reporting whether it took the ring from empty to
    /// non-empty.
    ///
    /// See [`ProduceOutcome`]: a consumer parks only after observing an empty
    /// ring, so only that transition can race its registration, and a producer
    /// wake gate can gate its fence and its wake on it.
    pub fn try_produce(&self, value: T) -> Result<ProduceOutcome, T> {
        let current = self.producer_seq.0.load(Ordering::Relaxed);
        let consumer = self.consumer_seq.0.load(Ordering::Acquire);

        // Check if full
        if current.wrapping_sub(consumer) >= self.buffer.len() {
            return Err(value);
        }

        // `producer_seq == consumer_seq` is exactly "the ring holds nothing":
        // only the consumer advances `consumer_seq`, and this thread is the sole
        // producer, so neither cursor can move between the loads above and this
        // decision.
        let outcome = if current == consumer {
            ProduceOutcome::BecameNonEmpty
        } else {
            ProduceOutcome::AlreadyOccupied
        };

        // SAFETY: SPSC capacity check keeps this slot outside the consumer
        // window; the write lock-free protocol makes this thread the sole
        // producer, and the slot is uninitialized until this write.
        unsafe {
            let slot = &mut *self.buffer[current & self.mask].get();
            slot.write(value);
        }

        self.producer_seq
            .0
            .store(current.wrapping_add(1), Ordering::Release);
        Ok(outcome)
    }

    /// Try to consume a value
    pub fn try_consume(&self) -> Option<T> {
        let current = self.consumer_seq.0.load(Ordering::Relaxed);
        let producer = self.producer_seq.0.load(Ordering::Acquire);

        if current == producer {
            return None;
        }

        let value = unsafe {
            let slot = &*self.buffer[current & self.mask].get();
            // SAFETY: producer > current check ensures this slot has data
            slot.assume_init_read()
        };

        self.consumer_seq
            .0
            .store(current.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    /// Get the capacity of the ring buffer
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the ring buffer is empty
    pub fn is_empty(&self) -> bool {
        let consumer = self.consumer_seq.0.load(Ordering::Acquire);
        let producer = self.producer_seq.0.load(Ordering::Acquire);
        consumer == producer
    }

    /// Check if the ring buffer is full
    pub fn is_full(&self) -> bool {
        let consumer = self.consumer_seq.0.load(Ordering::Acquire);
        let producer = self.producer_seq.0.load(Ordering::Acquire);
        producer.wrapping_sub(consumer) >= self.buffer.len()
    }

    /// Get the number of items currently in the buffer
    pub fn len(&self) -> usize {
        let consumer = self.consumer_seq.0.load(Ordering::Acquire);
        let producer = self.producer_seq.0.load(Ordering::Acquire);
        producer.wrapping_sub(consumer)
    }
}

/// The protocol's atoms, shared with the SPSC channel.
///
/// These are the primitives `channel::spsc::SpscChannel` composes: that wrapper
/// owns the *policy* (the cached-index layer, the `closed` flag, the
/// spin-then-yield schedule) and drives these for the *mechanism*. They are
/// `pub(crate)` rather than public because they take `&self` and mutate through
/// the ring's cells, so reaching them from outside the crate would let any number
/// of threads drive one end — the discipline `channel::spsc`'s non-`Clone` halves
/// enforce and a public type cannot.
impl<T> RingBuffer<T> {
    /// The producer and consumer cursors, in that order, read without
    /// synchronization.
    ///
    /// `Relaxed` is correct only because every caller holds the ring
    /// exclusively: [`SpscRing`](crate::channel::SpscRing)'s methods take `&self`
    /// or `&mut self`, and its halves borrow it, so no half can exist — and
    /// therefore no other thread can be advancing either cursor — while this
    /// runs. Reading these from a live half needs the acquire loads the send and
    /// receive paths use.
    pub(crate) fn indices(&self) -> (usize, usize) {
        (
            self.producer_seq.0.load(Ordering::Relaxed),
            self.consumer_seq.0.load(Ordering::Relaxed),
        )
    }

    /// The producer's own cursor, `Relaxed`: only the producing half advances it,
    /// and it only moves forward.
    pub(crate) fn producer_relaxed(&self) -> usize {
        self.producer_seq.0.load(Ordering::Relaxed)
    }

    /// The producer's cursor, `Acquire`: the `Release` store that publishes
    /// closure must be visible before the emptiness re-check that reads it.
    pub(crate) fn producer_acquire(&self) -> usize {
        self.producer_seq.0.load(Ordering::Acquire)
    }

    /// The consumer's own cursor, `Relaxed`, for the same reason as
    /// [`Self::producer_relaxed`].
    pub(crate) fn consumer_relaxed(&self) -> usize {
        self.consumer_seq.0.load(Ordering::Relaxed)
    }

    /// The consumer's cursor, `Acquire`: the `Release` store that publishes an
    /// element must be visible before that element is read.
    pub(crate) fn consumer_acquire(&self) -> usize {
        self.consumer_seq.0.load(Ordering::Acquire)
    }

    /// Room for one more value, consulting `cached_consumer` before the
    /// consumer's real cursor.
    ///
    /// The cached cursor is always at or behind the true one, because only the
    /// consumer advances it and it only moves forward. A stale value therefore
    /// makes the ring look *fuller* than it is, never emptier, so this may take
    /// the slow path unnecessarily but can never report room that does not exist.
    /// That one-sidedness is what makes the cache sound.
    pub(crate) fn has_room(&self, producer: usize, cached_consumer: &Cell<usize>) -> bool {
        if producer.wrapping_sub(cached_consumer.get()) < self.buffer.len() {
            return true;
        }
        // The cache says full; consult the consumer and try once more. This is
        // the only load that touches the consumer's cache line.
        let consumer = self.consumer_seq.0.load(Ordering::Acquire);
        cached_consumer.set(consumer);
        producer.wrapping_sub(consumer) < self.buffer.len()
    }

    /// A value is available, consulting `cached_producer` before the producer's
    /// real cursor. Mirrors [`Self::has_room`]: a stale cache understates what is
    /// queued, so it can cost an extra load but never invent an element.
    pub(crate) fn has_value(&self, consumer: usize, cached_producer: &Cell<usize>) -> bool {
        if consumer != cached_producer.get() {
            return true;
        }
        let producer = self.producer_seq.0.load(Ordering::Acquire);
        cached_producer.set(producer);
        consumer != producer
    }

    /// Write `value` into the slot at `producer` and publish the cursor.
    ///
    /// # Safety
    ///
    /// The caller must have established, through [`Self::has_room`], that
    /// `producer` is at or beyond the consumer's cursor, and must be the sole
    /// producer: this writes a slot the consumer may reach as soon as the release
    /// store below lands.
    pub(crate) unsafe fn produce_at(&self, producer: usize, value: T) {
        // SAFETY: `producer` is past the consumer's cursor, so this slot is not
        // one the consumer may read until the release store publishes it, and
        // only the producing half writes slots.
        unsafe {
            let slot = &mut *self.buffer[producer & self.mask].get();
            slot.write(value);
        }
        self.producer_seq
            .0
            .store(producer.wrapping_add(1), Ordering::Release);
    }

    /// Take the value in the slot at `consumer` and publish the cursor.
    ///
    /// # Safety
    ///
    /// The caller must have established, through [`Self::has_value`], that
    /// `consumer` is behind the published producer cursor, and must be the sole
    /// consumer: the slot is read once here and never again.
    pub(crate) unsafe fn consume_at(&self, consumer: usize) -> T {
        // SAFETY: `consumer` is behind the published producer cursor, so this
        // slot was written and released by the producer. It has not been read
        // before — the cursor advances once per value, and only the consuming
        // half advances it.
        let value = unsafe {
            let slot = &*self.buffer[consumer & self.mask].get();
            slot.assume_init_read()
        };
        self.consumer_seq
            .0
            .store(consumer.wrapping_add(1), Ordering::Release);
        value
    }
}

impl<T> Drop for RingBuffer<T> {
    fn drop(&mut self) {
        let consumer = *self.consumer_seq.0.get_mut();
        let producer = *self.producer_seq.0.get_mut();
        let len = producer.wrapping_sub(consumer);
        for i in 0..len {
            let idx = (consumer.wrapping_add(i)) & self.mask;
            // SAFETY: exclusive `&mut self` in drop; every live index in
            // `consumer..producer` was written by produce and not yet read,
            // so dropping it here discharges each value exactly once.
            unsafe {
                let slot = &mut *self.buffer[idx].get();
                slot.assume_init_drop();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The classification the producer wake gate hangs on: a produce reports
    /// whether it took the ring from empty to non-empty.
    ///
    /// Misclassifying an empty→non-empty produce as occupied would strand a
    /// parked consumer (its fence and wake would be skipped); the reverse only
    /// costs an unnecessary fence. So the empty case, the occupied case, and the
    /// return to empty are all pinned.
    #[test]
    fn reports_the_empty_to_non_empty_transition() {
        let rb = RingBuffer::<u8>::new(4);

        assert_eq!(rb.try_produce(1), Ok(ProduceOutcome::BecameNonEmpty));
        assert_eq!(rb.try_produce(2), Ok(ProduceOutcome::AlreadyOccupied));

        assert_eq!(rb.try_consume(), Some(1));
        assert_eq!(rb.try_produce(3), Ok(ProduceOutcome::AlreadyOccupied));

        assert_eq!(rb.try_consume(), Some(2));
        assert_eq!(rb.try_consume(), Some(3));
        // Drained: the next produce is the empty→non-empty transition again.
        assert_eq!(rb.try_produce(4), Ok(ProduceOutcome::BecameNonEmpty));
    }

    #[test]
    fn test_wrapping_drop_correctness() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);
        struct TrackDrop;
        impl Drop for TrackDrop {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        {
            let mut rb = RingBuffer::<TrackDrop>::new(4);
            let mask = rb.mask;
            unsafe {
                let slot1 = &mut *rb.buffer[(usize::MAX - 1) & mask].get();
                slot1.write(TrackDrop);
                let slot2 = &mut *rb.buffer[usize::MAX & mask].get();
                slot2.write(TrackDrop);
                let slot3 = &mut *rb.buffer[0].get();
                slot3.write(TrackDrop);
            }

            *rb.consumer_seq.0.get_mut() = usize::MAX - 1;
            *rb.producer_seq.0.get_mut() = 1;
        }

        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3);
    }
}
