use crate::platform::*;
use super::stack::CachePadded;

/// Slab allocator for efficient task storage (inspired by Tokio)
///
/// This provides O(1) allocation and deallocation with minimal fragmentation.
pub struct SlabAllocator<T> {
    /// Storage for all items
    entries: Box<[SlabEntry<T>]>,
    /// Next free slot
    next_free: AtomicUsize,
    /// Number of allocated items
    len: CachePadded<AtomicUsize>,
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
            next_free: AtomicUsize::new(0),
            len: CachePadded {
                value: AtomicUsize::new(0),
            },
        }
    }

    /// Allocate a slot and store the value
    ///
    /// Returns the index of the allocated slot, or None if full
    pub fn insert(&self, value: T) -> Option<usize> {
        loop {
            let free_idx = self.next_free.load(Ordering::Acquire);

            if free_idx >= self.entries.len() {
                return None; // Slab is full
            }

            let entry = &self.entries[free_idx];
            let next = entry.next.load(Ordering::Relaxed);

            // Try to claim this slot
            if self
                .next_free
                .compare_exchange_weak(free_idx, next, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                // Successfully claimed the slot
                unsafe {
                    (*entry.value.get()).write(value);
                }

                // Mark as occupied after writing the value
                debug_assert!(
                    !entry.occupied.load(Ordering::Relaxed),
                    "Slot should be vacant before marking occupied"
                );
                entry.occupied.store(true, Ordering::Release);
                self.len.value.fetch_add(1, Ordering::Relaxed);

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
            let current_free = self.next_free.load(Ordering::Relaxed);
            entry.next.store(current_free, Ordering::Relaxed);

            if self
                .next_free
                .compare_exchange_weak(current_free, idx, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        self.len.value.fetch_sub(1, Ordering::Relaxed);
        Some(value)
    }

    /// Get a reference to the value at the given index
    pub fn get(&self, idx: usize) -> Option<&T> {
        if idx >= self.entries.len() {
            return None;
        }

        let entry = &self.entries[idx];

        if entry.occupied.load(Ordering::Acquire) {
            Some(unsafe { &*(*entry.value.get()).as_ptr() })
        } else {
            None
        }
    }

    /// Get the number of allocated items
    pub fn len(&self) -> usize {
        self.len.value.load(Ordering::Relaxed)
    }
}
