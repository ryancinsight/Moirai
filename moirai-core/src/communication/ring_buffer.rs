use moirai_utils::cache::CacheAligned;
use std::cell::UnsafeCell;
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

    /// Try to produce a value
    pub fn try_produce(&self, value: T) -> Result<(), T> {
        let current = self.producer_seq.0.load(Ordering::Relaxed);
        let consumer = self.consumer_seq.0.load(Ordering::Acquire);

        // Check if full
        if current.wrapping_sub(consumer) >= self.buffer.len() {
            return Err(value);
        }

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
        Ok(())
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
