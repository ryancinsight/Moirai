use super::stack::CacheAligned;
use crate::platform::*;

#[cfg(target_pointer_width = "64")]
mod pack {
    pub const INDEX_MASK: usize = 0xFFFF_FFFF;
    pub const GEN_SHIFT: u32 = 32;
}

#[cfg(not(target_pointer_width = "64"))]
mod pack {
    pub const INDEX_MASK: usize = 0xFFFF;
    pub const GEN_SHIFT: u32 = 16;
}

use pack::*;

#[inline]
fn pack(index: usize, generation: usize) -> usize {
    (index & INDEX_MASK) | (generation << GEN_SHIFT)
}

#[inline]
fn unpack(val: usize) -> (usize, usize) {
    (val & INDEX_MASK, val >> GEN_SHIFT)
}

/// Slab allocator for efficient task storage (inspired by Tokio)
///
/// This provides O(1) allocation and deallocation with minimal fragmentation.
pub struct SlabAllocator<T> {
    /// Storage for all items
    entries: Box<[SlabEntry<T>]>,
    /// Next free slot (generation-packed to prevent ABA)
    next_free: AtomicUsize,
    /// Number of allocated items
    len: CacheAligned<AtomicUsize>,
}

struct SlabEntry<T> {
    /// The stored value (if occupied)
    value: UnsafeCell<MaybeUninit<T>>,
    /// Next free index (if vacant)
    next: AtomicUsize,
    /// Whether this slot is occupied
    occupied: AtomicBool,
}

// Safety: SlabEntry is Send and Sync because access is controlled by SlabAllocator
unsafe impl<T: Send> Send for SlabEntry<T> {}
unsafe impl<T: Sync> Sync for SlabEntry<T> {}

impl<T> SlabAllocator<T> {
    /// Create a new slab allocator with the given capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity <= INDEX_MASK,
            "Capacity exceeds maximum allowed for SlabAllocator"
        );
        let mut entries = Vec::with_capacity(capacity);

        // Initialize free list
        for i in 0..capacity {
            entries.push(SlabEntry {
                value: UnsafeCell::new(MaybeUninit::uninit()),
                next: AtomicUsize::new(i + 1),
                occupied: AtomicBool::new(false),
            });
        }

        Self {
            entries: entries.into_boxed_slice(),
            next_free: AtomicUsize::new(pack(0, 0)),
            len: CacheAligned::new(AtomicUsize::new(0)),
        }
    }

    /// Allocate a slot and store the value
    ///
    /// Returns the index of the allocated slot, or None if full
    pub fn insert(&self, value: T) -> Option<usize> {
        loop {
            let packed_free = self.next_free.load(Ordering::Acquire);
            let (free_idx, gen) = unpack(packed_free);

            if free_idx >= self.entries.len() {
                return None; // Slab is full
            }

            let entry = &self.entries[free_idx];
            let next_idx = entry.next.load(Ordering::Relaxed);

            let new_packed = pack(next_idx, gen.wrapping_add(1));

            // Try to claim this slot
            if self
                .next_free
                .compare_exchange_weak(
                    packed_free,
                    new_packed,
                    Ordering::Release,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                // Successfully claimed the slot
                // SAFETY: winning the free-list CAS grants exclusive
                // ownership of `free_idx` until it re-enters the list; the
                // value cell starts uninit and was moved out on any prior
                // removal, so writing here cannot alias or double-drop.
                unsafe {
                    (*entry.value.get()).write(value);
                }

                // Mark as occupied after writing the value
                debug_assert!(
                    !entry.occupied.load(Ordering::Relaxed),
                    "Slot should be vacant before marking occupied"
                );
                entry.occupied.store(true, Ordering::Release);
                self.len.0.fetch_add(1, Ordering::Relaxed);

                return Some(free_idx);
            }
        }
    }

    /// Remove and return the value at the given index
    pub fn remove(&self, idx: usize) -> Option<T> {
        if idx >= self.entries.len() {
            return None;
        }

        let entry = &self.entries[idx];

        if !entry.occupied.swap(false, Ordering::Acquire) {
            return None; // Slot was already vacant
        }

        // Extract the value
        // SAFETY: The occupied flag ensures this slot contains initialized data.
        // The swap(false) above gives us exclusive access to this slot.
        // The data was initialized in insert() with write(value).
        let value = unsafe { (*entry.value.get()).assume_init_read() };

        // Add to free list
        loop {
            let packed_free = self.next_free.load(Ordering::Relaxed);
            let (free_idx, gen) = unpack(packed_free);
            entry.next.store(free_idx, Ordering::Relaxed);

            let new_packed = pack(idx, gen.wrapping_add(1));

            if self
                .next_free
                .compare_exchange_weak(
                    packed_free,
                    new_packed,
                    Ordering::Release,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }

        self.len.0.fetch_sub(1, Ordering::Relaxed);
        Some(value)
    }

    /// Get a reference to the value at the given index.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no concurrent `remove` call targets `idx`
    /// while the returned reference is live. Concurrent read access to different
    /// indices is safe, but concurrent read/write to the same index requires external synchronization.
    pub unsafe fn get(&self, idx: usize) -> Option<&T> {
        if idx >= self.entries.len() {
            return None;
        }

        let entry = &self.entries[idx];

        if entry.occupied.load(Ordering::Acquire) {
            Some(&*(*entry.value.get()).as_ptr())
        } else {
            None
        }
    }

    /// Get the number of allocated items
    pub fn len(&self) -> usize {
        self.len.0.load(Ordering::Relaxed)
    }

    /// Check if the slab allocator is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Drop for SlabAllocator<T> {
    fn drop(&mut self) {
        // Iterate through all entries and drop occupied ones.
        for entry in &mut self.entries {
            if *entry.occupied.get_mut() {
                // SAFETY: `occupied` is set only after `value` is initialized, and
                // `&mut self` makes this the sole drop of the occupied entry.
                unsafe {
                    entry.value.get_mut().assume_init_drop();
                }
            }
        }
    }
}
