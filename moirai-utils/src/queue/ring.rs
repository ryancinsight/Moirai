//! The bounded MPMC ring itself: storage, the two cursors, and the capacity
//! contract.

use core::cell::UnsafeCell;
use core::cmp::Ordering as CmpOrdering;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::cache::CacheAligned;

#[cfg(feature = "std")]
use std::boxed::Box;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

/// Default capacity for [`LockFreeQueue`]. Large enough to avoid backpressure
/// under normal scheduling load while bounding memory under adversarial
/// producer rates per the bounded-resource policy.
const DEFAULT_QUEUE_CAPACITY: usize = 65536;

/// Spins a producer makes while a dequeue reopens a slot, before it yields its
/// time slice. Reopening is one store after the item is moved out, so a spin or
/// two covers it; only a consumer preempted in between needs the yield.
const SPINS_BEFORE_YIELD: u32 = 64;

/// Times a producer found a claimed slot not yet reopened, so a test can
/// reopen the slot once a producer is observed waiting on it.
#[cfg(test)]
pub(super) static REOPEN_WAITS: AtomicUsize = AtomicUsize::new(0);

/// A single slot in the bounded MPMC queue.
pub(super) struct Slot<T> {
    /// Monotonic sequence number that distinguishes empty, full, and stale
    /// states without an ABA hazard.
    sequence: AtomicUsize,
    /// The slot's data. It is uninitialized exactly while the sequence number
    /// reports the slot empty, so occupancy needs no separate flag and no
    /// `Option` discriminant: the slot costs one machine word per payload.
    data: UnsafeCell<MaybeUninit<T>>,
}

/// A bounded, genuinely lock-free multi-producer multi-consumer queue.
///
/// This is an array-based MPMC queue using per-slot sequence numbers (the
/// Vyukov algorithm). Producers and consumers operate through independent
/// atomic head/tail cursors and never acquire a mutex or spinlock. The
/// sequence-number protocol eliminates the ABA problem without tagged
/// pointers or epoch-based reclamation: slots are reused in place, so no
/// node allocation or deallocation occurs during enqueue/dequeue.
///
/// # Capacity
///
/// The queue is bounded. [`LockFreeQueue::new`] creates a queue with
/// `DEFAULT_QUEUE_CAPACITY` usable slots. [`LockFreeQueue::with_capacity`]
/// accepts any request of one slot or more: the ring is sized to the next power
/// of two at least two, while [`capacity`](LockFreeQueue::capacity) keeps
/// reporting the request, so a non-power-of-two capacity bounds the queue
/// exactly and only the ring's unused tail is wasted. When the queue is full,
/// [`enqueue`] retries with exponential backoff (preserving the
/// unblocked-sender contract of the previous API), while [`try_enqueue`]
/// returns `Err(item)` for callers that prefer explicit backpressure. Full
/// means `capacity` items queued: a slot a dequeue has emptied but not yet
/// reopened is waited for, never reported as fullness.
///
/// # Memory safety
///
/// Each slot's `MaybeUninit<T>` is written by the producer — the only writer,
/// between `sequence == pos` and `sequence == pos + 1` — and moved out by the
/// consumer, which is the only reader, between `sequence == pos + 1` and
/// `sequence == pos + ring_len`. The sequence-number protocol therefore
/// guarantees that only one thread ever touches a slot's payload.
///
/// [`enqueue`]: LockFreeQueue::enqueue
/// [`try_enqueue`]: LockFreeQueue::try_enqueue
// No struct-level `repr(align)`: `head`/`tail` are `CacheAligned`, so the
// struct's alignment already equals `DESTRUCTIVE_INTERFERENCE_SIZE` and tracks
// the per-target table in `cache.rs` instead of pinning a second literal here.
pub struct LockFreeQueue<T> {
    buffer: Box<[Slot<T>]>,
    mask: usize,
    /// Slots in the ring: a power of two, at least two, so that a slot's empty
    /// and full generations never alias. Private because the sequence protocol
    /// addresses slots with it; callers reason in `capacity`.
    ring_len: usize,
    /// Usable slots: what `try_enqueue` accepts before reporting full.
    capacity: usize,
    head: CacheAligned<AtomicUsize>,
    tail: CacheAligned<AtomicUsize>,
}

// Safety: The sequence-number protocol ensures that each slot's data is
// accessed by at most one thread at a time: a producer writes between
// sequence == pos and sequence == pos+1; a consumer takes between
// sequence == pos+1 and sequence == pos+ring_len. The head and tail atomics
// are independently advanced via CAS, so no global lock is needed. T: Send
// is sufficient because ownership of the value transfers between threads
// through the slot, never shared concurrently.
unsafe impl<T: Send> Send for LockFreeQueue<T> {}
unsafe impl<T: Send> Sync for LockFreeQueue<T> {}

impl<T> LockFreeQueue<T> {
    /// Create a new queue with the default capacity.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_QUEUE_CAPACITY)
    }

    /// Create a new queue holding up to `capacity` items.
    ///
    /// The ring is the next power of two at least two, which the sequence
    /// protocol needs to tell a slot's empty generation from its full one; a
    /// one-slot request therefore gets a two-slot ring and still bounds the
    /// queue at one item.
    #[track_caller]
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let ring_len = capacity.next_power_of_two().max(2);

        #[cfg(feature = "std")]
        let buffer: Box<[Slot<T>]> = (0..ring_len)
            .map(|i| Slot {
                sequence: AtomicUsize::new(i),
                data: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect::<std::vec::Vec<_>>()
            .into_boxed_slice();

        #[cfg(not(feature = "std"))]
        let buffer: Box<[Slot<T>]> = (0..ring_len)
            .map(|i| Slot {
                sequence: AtomicUsize::new(i),
                data: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect::<alloc::vec::Vec<_>>()
            .into_boxed_slice();

        Self {
            buffer,
            mask: ring_len - 1,
            ring_len,
            capacity,
            head: CacheAligned::new(AtomicUsize::new(0)),
            tail: CacheAligned::new(AtomicUsize::new(0)),
        }
    }

    /// Try to enqueue an item without waiting for space.
    ///
    /// Returns `Ok(())` if the item was enqueued, or `Err(item)` if the queue
    /// holds [`capacity`](LockFreeQueue::capacity) items. It takes no lock.
    /// When the queue has room but the item's slot still belongs to a dequeue
    /// that has moved its item out and not yet reopened the slot, it waits for
    /// that reopening instead of reporting a full queue; the wait is one store
    /// unless the dequeuing thread was preempted.
    #[inline]
    pub fn try_enqueue(&self, item: T) -> Result<(), T> {
        let mut pos = self.tail.load(Ordering::Relaxed);
        let mut waits = 0;
        loop {
            let slot = &self.buffer[pos & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            #[allow(clippy::cast_possible_wrap)]
            let diff = seq.wrapping_sub(pos) as isize;

            match diff.cmp(&0) {
                CmpOrdering::Equal => {
                    // The ring can be larger than the requested capacity, so
                    // fullness is the request, not the ring size.
                    if self.holds_capacity(pos) {
                        return Err(item);
                    }

                    match self.tail.compare_exchange_weak(
                        pos,
                        pos.wrapping_add(1),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            // SAFETY: winning the tail CAS grants exclusive
                            // right to fill this slot's sequence generation; its
                            // payload cell is uninitialized (fresh or drained)
                            // until this write publishes it.
                            unsafe {
                                (*slot.data.get()).write(item);
                            }
                            slot.sequence.store(pos.wrapping_add(1), Ordering::Release);
                            return Ok(());
                        }
                        Err(actual) => pos = actual,
                    }
                }
                // The slot still holds the generation written one lap ago.
                // Below capacity, a dequeue has claimed that item and not yet
                // reopened the slot: wait for it rather than report a fullness
                // that does not exist, as crossbeam's `ArrayQueue::push` does.
                CmpOrdering::Less => {
                    if self.holds_capacity(pos) {
                        return Err(item);
                    }
                    wait_for_reopen(&mut waits);
                    pos = self.tail.load(Ordering::Relaxed);
                }
                // Another producer advanced tail before us: reload and retry.
                CmpOrdering::Greater => pos = self.tail.load(Ordering::Relaxed),
            }
        }
    }

    /// Whether the queue holds `capacity` items once the tail reaches `pos`.
    ///
    /// `pos` may be stale: once other producers fill that position and
    /// consumers drain it, the head passes `pos` and the wrapped difference is
    /// huge. No real occupancy exceeds the ring, so a difference past
    /// `ring_len` is a stale position the caller retries, not a full queue.
    #[inline]
    pub(super) fn holds_capacity(&self, pos: usize) -> bool {
        let queued = pos.wrapping_sub(self.head.load(Ordering::Acquire));
        (self.capacity..=self.ring_len).contains(&queued)
    }

    /// Enqueue an item, retrying with exponential backoff if the queue is full.
    ///
    /// This preserves the unblocked-sender contract of the previous API: the
    /// call always eventually succeeds (assuming consumers make progress).
    /// The backoff path uses `core::hint::spin_loop` and, on std targets,
    /// `std::thread::yield_now` after heavy contention, but never acquires a
    /// global lock, so multiple producers can enqueue concurrently.
    #[inline]
    pub fn enqueue(&self, item: T) {
        let mut backoff: usize = 1;
        let mut item = Some(item);
        loop {
            match self.try_enqueue(item.take().expect("invariant: item present")) {
                Ok(()) => return,
                Err(returned) => {
                    item = Some(returned);
                    for _ in 0..backoff {
                        core::hint::spin_loop();
                    }
                    if backoff < 64 {
                        backoff = backoff.saturating_mul(2);
                    } else {
                        #[cfg(feature = "std")]
                        {
                            std::thread::yield_now();
                        }
                        backoff = 1;
                    }
                }
            }
        }
    }

    /// Try to dequeue an item from the front of the queue.
    /// Returns `None` if the queue is empty.
    ///
    /// This is the lock-free fast path: no spinlock, no mutex.
    #[inline]
    pub fn try_dequeue(&self) -> Option<T> {
        let (pos, item) = self.claim_front()?;
        // SAFETY: `pos` was claimed just above and is reopened once.
        unsafe { self.reopen(pos) };
        Some(item)
    }

    /// Claims the front item: advances the head past it and moves it out,
    /// leaving its slot closed to producers until [`Self::reopen`].
    #[inline]
    pub(super) fn claim_front(&self) -> Option<(usize, T)> {
        let mut pos = self.head.load(Ordering::Relaxed);
        loop {
            let slot = &self.buffer[pos & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            #[allow(clippy::cast_possible_wrap)]
            let diff = seq.wrapping_sub(pos.wrapping_add(1)) as isize;

            match diff.cmp(&0) {
                CmpOrdering::Equal => {
                    // Slot has data: try to claim it by advancing head.
                    match self.head.compare_exchange_weak(
                        pos,
                        pos.wrapping_add(1),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            // SAFETY: winning the head CAS means no other
                            // consumer can claim this slot, and the sequence
                            // == pos+1 invariant means the producer finished
                            // writing; no producer writes it again until
                            // `reopen` opens the next generation.
                            let item = unsafe { (*slot.data.get()).assume_init_read() };
                            return Some((pos, item));
                        }
                        Err(actual) => pos = actual,
                    }
                }
                // Queue is empty: sequence has not advanced past pos+1.
                CmpOrdering::Less => return None,
                // Another consumer advanced head before us: reload and retry.
                CmpOrdering::Greater => pos = self.head.load(Ordering::Relaxed),
            }
        }
    }

    /// Reopens the slot of the item claimed at `pos` for the next lap's
    /// producer.
    ///
    /// # Safety
    /// `pos` must come from [`Self::claim_front`] on this queue and be
    /// reopened exactly once: reopening any other slot lets a producer write
    /// a payload a consumer may still be reading.
    #[inline]
    pub(super) unsafe fn reopen(&self, pos: usize) {
        self.buffer[pos & self.mask]
            .sequence
            .store(pos.wrapping_add(self.ring_len), Ordering::Release);
    }

    /// Check if the queue is empty.
    ///
    /// This is a best-effort check: the queue may have items added or removed
    /// between this call and the next operation. It is safe to call
    /// concurrently with enqueue/dequeue.
    pub fn is_empty(&self) -> bool {
        self.tail.load(Ordering::Acquire) == self.head.load(Ordering::Acquire)
    }

    /// Check if the queue holds `capacity` items.
    ///
    /// Best-effort in the same sense as [`is_empty`](LockFreeQueue::is_empty).
    pub fn is_full(&self) -> bool {
        self.tail
            .load(Ordering::Acquire)
            .wrapping_sub(self.head.load(Ordering::Acquire))
            >= self.capacity
    }

    /// Number of items the queue accepts before `try_enqueue` reports full.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of items currently queued.
    ///
    /// Best-effort in the same sense as [`is_empty`](LockFreeQueue::is_empty):
    /// it is the cursor difference, so a push that has reserved its position but
    /// not yet published its slot is already counted, and the value can change
    /// under the caller immediately after the read.
    pub fn len(&self) -> usize {
        self.tail
            .load(Ordering::Acquire)
            .wrapping_sub(self.head.load(Ordering::Acquire))
    }
}

/// Pauses a producer waiting for a dequeue to reopen a slot: spin briefly, then
/// yield where the platform can, so a preempted consumer gets to run.
#[inline]
fn wait_for_reopen(waits: &mut u32) {
    #[cfg(test)]
    REOPEN_WAITS.fetch_add(1, Ordering::Relaxed);
    if *waits < SPINS_BEFORE_YIELD {
        *waits += 1;
        core::hint::spin_loop();
    } else {
        #[cfg(feature = "std")]
        std::thread::yield_now();
        #[cfg(not(feature = "std"))]
        core::hint::spin_loop();
    }
}

impl<T> Default for LockFreeQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for LockFreeQueue<T> {
    fn drop(&mut self) {
        // Exclusive access: walk the published range and drop what nobody took,
        // reading the sequence numbers without atomics.
        let head = *self.head.0.get_mut();
        let tail = *self.tail.0.get_mut();

        for pos in head..tail {
            let slot = &mut self.buffer[pos & self.mask];
            if *slot.sequence.get_mut() == pos.wrapping_add(1) {
                // SAFETY: the published sequence marks the payload
                // initialized, and each position is visited once.
                unsafe {
                    (*slot.data.get()).assume_init_drop();
                }
            }
        }
    }
}
