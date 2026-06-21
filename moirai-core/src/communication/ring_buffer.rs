use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use moirai_utils::cache::CachePadded;

/// Zero-copy ring buffer for high-throughput streaming
///
/// # Safety
///
/// This structure uses `MaybeUninit` for zero-copy performance:
/// - Values are written with `write()` before incrementing producer_seq
/// - The `assume_init_read()` in `try_consume()` is safe because we check
///   that producer_seq > current, ensuring data was written
pub struct RingBuffer<T> {
    /// Buffer storage
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Capacity mask for fast modulo
    mask: usize,
    /// Producer sequence number
    producer_seq: CachePadded<AtomicUsize>,
    /// Consumer sequence number
    consumer_seq: CachePadded<AtomicUsize>,
}

unsafe impl<T: Send> Send for RingBuffer<T> {}
unsafe impl<T: Send> Sync for RingBuffer<T> {}

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
            producer_seq: CachePadded::new(AtomicUsize::new(0)),
            consumer_seq: CachePadded::new(AtomicUsize::new(0)),
        }
    }

    /// Try to produce a value
    pub fn try_produce(&self, value: T) -> Result<(), T> {
        let current = self.producer_seq.value.load(Ordering::Relaxed);
        let consumer = self.consumer_seq.value.load(Ordering::Acquire);

        // Check if full
        if current.wrapping_sub(consumer) >= self.buffer.len() {
            return Err(value);
        }

        unsafe {
            let slot = &mut *self.buffer[current & self.mask].get();
            slot.write(value);
        }

        self.producer_seq
            .value
            .store(current.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Try to consume a value
    pub fn try_consume(&self) -> Option<T> {
        let current = self.consumer_seq.value.load(Ordering::Relaxed);
        let producer = self.producer_seq.value.load(Ordering::Acquire);

        if current == producer {
            return None;
        }

        let value = unsafe {
            let slot = &*self.buffer[current & self.mask].get();
            // SAFETY: producer > current check ensures this slot has data
            slot.assume_init_read()
        };

        self.consumer_seq
            .value
            .store(current.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    /// Get the capacity of the ring buffer
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the ring buffer is empty
    pub fn is_empty(&self) -> bool {
        let consumer = self.consumer_seq.value.load(Ordering::Acquire);
        let producer = self.producer_seq.value.load(Ordering::Acquire);
        consumer == producer
    }

    /// Check if the ring buffer is full
    pub fn is_full(&self) -> bool {
        let consumer = self.consumer_seq.value.load(Ordering::Acquire);
        let producer = self.producer_seq.value.load(Ordering::Acquire);
        producer.wrapping_sub(consumer) >= self.buffer.len()
    }

    /// Get the number of items currently in the buffer
    pub fn len(&self) -> usize {
        let consumer = self.consumer_seq.value.load(Ordering::Acquire);
        let producer = self.producer_seq.value.load(Ordering::Acquire);
        producer.wrapping_sub(consumer)
    }
}

impl<T> Drop for RingBuffer<T> {
    fn drop(&mut self) {
        let consumer = *self.consumer_seq.value.get_mut();
        let producer = *self.producer_seq.value.get_mut();
        let len = producer.wrapping_sub(consumer);
        for i in 0..len {
            let idx = (consumer.wrapping_add(i)) & self.mask;
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

            *rb.consumer_seq.value.get_mut() = usize::MAX - 1;
            *rb.producer_seq.value.get_mut() = 1;
        }

        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3);
    }
}
