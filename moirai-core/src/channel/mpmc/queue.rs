use crate::channel::error::CacheAligned;
use std::cell::UnsafeCell;
use std::cmp::Ordering as CmpOrdering;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) struct BoundedMpmcSlot<T> {
    sequence: AtomicUsize,
    value: UnsafeCell<MaybeUninit<T>>,
}

pub(super) struct BoundedMpmcQueue<T> {
    buffer: Box<[BoundedMpmcSlot<T>]>,
    mask: usize,
    capacity: usize,
    logical_capacity: usize,
    enqueue_pos: CacheAligned<AtomicUsize>,
    dequeue_pos: CacheAligned<AtomicUsize>,
}

impl<T> BoundedMpmcQueue<T> {
    pub(super) fn new(requested_capacity: usize) -> Self {
        let logical_capacity = requested_capacity.max(1);
        let capacity = logical_capacity.next_power_of_two().max(2);
        let buffer = (0..capacity)
            .map(|index| BoundedMpmcSlot {
                sequence: AtomicUsize::new(index),
                value: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            buffer,
            mask: capacity - 1,
            capacity,
            logical_capacity,
            enqueue_pos: CacheAligned::new(AtomicUsize::new(0)),
            dequeue_pos: CacheAligned::new(AtomicUsize::new(0)),
        }
    }

    pub(super) fn try_push(&self, value: T) -> std::result::Result<(), T> {
        let mut position = self.enqueue_pos.0.load(Ordering::Relaxed);

        loop {
            let slot = &self.buffer[position & self.mask];
            let sequence = slot.sequence.load(Ordering::Acquire);
            #[allow(clippy::cast_possible_wrap)]
            let difference = sequence.wrapping_sub(position) as isize;

            match difference.cmp(&0) {
                CmpOrdering::Equal => {
                    if position.wrapping_sub(self.dequeue_pos.0.load(Ordering::Acquire))
                        >= self.logical_capacity
                    {
                        return Err(value);
                    }

                    match self.enqueue_pos.0.compare_exchange_weak(
                        position,
                        position.wrapping_add(1),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            // SAFETY: winning the enqueue-position CAS grants
                            // exclusive right to fill this sequence slot; its
                            // value cell is uninit (fresh or drained) until
                            // this write publishes it.
                            unsafe {
                                (*slot.value.get()).write(value);
                            }
                            slot.sequence
                                .store(position.wrapping_add(1), Ordering::Release);
                            return Ok(());
                        }
                        Err(observed) => position = observed,
                    }
                }
                CmpOrdering::Less => return Err(value),
                CmpOrdering::Greater => {
                    position = self.enqueue_pos.0.load(Ordering::Relaxed);
                }
            }
        }
    }

    pub(super) fn try_pop(&self) -> Option<T> {
        let mut position = self.dequeue_pos.0.load(Ordering::Relaxed);

        loop {
            let slot = &self.buffer[position & self.mask];
            let sequence = slot.sequence.load(Ordering::Acquire);
            #[allow(clippy::cast_possible_wrap)]
            let difference = sequence.wrapping_sub(position.wrapping_add(1)) as isize;

            match difference.cmp(&0) {
                CmpOrdering::Equal => {
                    match self.dequeue_pos.0.compare_exchange_weak(
                        position,
                        position.wrapping_add(1),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            let value = unsafe { (*slot.value.get()).assume_init_read() };
                            slot.sequence
                                .store(position.wrapping_add(self.capacity), Ordering::Release);
                            return Some(value);
                        }
                        Err(observed) => position = observed,
                    }
                }
                CmpOrdering::Less => return None,
                CmpOrdering::Greater => {
                    position = self.dequeue_pos.0.load(Ordering::Relaxed);
                }
            }
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.enqueue_pos.0.load(Ordering::Acquire) == self.dequeue_pos.0.load(Ordering::Acquire)
    }

    pub(super) fn is_full(&self) -> bool {
        self.enqueue_pos
            .0
            .load(Ordering::Acquire)
            .wrapping_sub(self.dequeue_pos.0.load(Ordering::Acquire))
            >= self.logical_capacity
    }

    pub(super) fn logical_capacity(&self) -> usize {
        self.logical_capacity
    }
}

impl<T> Drop for BoundedMpmcQueue<T> {
    fn drop(&mut self) {
        let dequeue_pos = *self.dequeue_pos.0.get_mut();
        let enqueue_pos = *self.enqueue_pos.0.get_mut();
        let len = enqueue_pos.wrapping_sub(dequeue_pos);

        for i in 0..len {
            let pos = dequeue_pos.wrapping_add(i);
            let slot = &mut self.buffer[pos & self.mask];
            let sequence = *slot.sequence.get_mut();
            if sequence == pos.wrapping_add(1) {
                // SAFETY: exclusive `&mut self` in drop; the published
                // sequence marks the cell initialized, and drain order visits
                // each published slot once.
                unsafe {
                    (*slot.value.get()).assume_init_drop();
                }
            }
        }
    }
}

// SAFETY: values move between threads through sequence-gated slots, so
// `T: Send` is required and sufficient; no references escape.
unsafe impl<T: Send> Send for BoundedMpmcQueue<T> {}
// SAFETY: all shared access is arbitrated by the position CAS protocol on
// atomics; stored values are touched only by the thread that owns their
// sequence claim, so `T: Send` suffices.
unsafe impl<T: Send> Sync for BoundedMpmcQueue<T> {}
