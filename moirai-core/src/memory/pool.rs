use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

const SENTINEL: u32 = u32::MAX;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PackedState {
    index: u32,
    generation: u32,
}

impl PackedState {
    #[inline]
    fn from_u64(val: u64) -> Self {
        Self {
            index: val as u32,
            generation: (val >> 32) as u32,
        }
    }

    #[inline]
    fn to_u64(self) -> u64 {
        u64::from(self.index) | (u64::from(self.generation) << 32)
    }
}

/// Memory pool for reducing allocation overhead.
///
/// Implements an O(1) lock-free index-based stack for contention-free performance,
/// completely avoiding the sequential linear scanning bottleneck of the original design.
pub struct MemoryPool<T> {
    /// Slots holding the actual values
    slots: Box<[UnsafeCell<Option<T>>]>,
    /// Head of the occupied list of indices (data is Some)
    occupied_head: AtomicU64,
    /// Head of the free list of indices (data is None)
    free_head: AtomicU64,
    /// Link pointers for lists
    next: Box<[AtomicU32]>,
    /// Current pool size (occupied count)
    size: AtomicUsize,
    /// Maximum pool size
    max_size: usize,
}

unsafe impl<T: Send> Send for MemoryPool<T> {}
unsafe impl<T: Send> Sync for MemoryPool<T> {}

impl<T> MemoryPool<T> {
    /// Create a new memory pool with specified maximum size
    pub fn new(max_size: usize) -> Self {
        let mut slots = Vec::with_capacity(max_size);
        let mut next = Vec::with_capacity(max_size);
        for i in 0..max_size {
            slots.push(UnsafeCell::new(None));
            next.push(AtomicU32::new(i as u32 + 1));
        }
        if max_size > 0 {
            next[max_size - 1].store(SENTINEL, Ordering::Relaxed);
        }

        Self {
            slots: slots.into_boxed_slice(),
            occupied_head: AtomicU64::new(
                PackedState {
                    index: SENTINEL,
                    generation: 0,
                }
                .to_u64(),
            ),
            free_head: AtomicU64::new(
                PackedState {
                    index: if max_size > 0 { 0 } else { SENTINEL },
                    generation: 0,
                }
                .to_u64(),
            ),
            next: next.into_boxed_slice(),
            size: AtomicUsize::new(0),
            max_size,
        }
    }

    #[inline]
    fn push_list(&self, head: &AtomicU64, index: u32) {
        loop {
            let current_val = head.load(Ordering::Acquire);
            let state = PackedState::from_u64(current_val);

            self.next[index as usize].store(state.index, Ordering::Release);

            let new_state = PackedState {
                index,
                generation: state.generation.wrapping_add(1),
            };

            if head
                .compare_exchange_weak(
                    current_val,
                    new_state.to_u64(),
                    Ordering::Release,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }
    }

    #[inline]
    fn pop_list(&self, head: &AtomicU64) -> Option<u32> {
        loop {
            let current_val = head.load(Ordering::Acquire);
            let state = PackedState::from_u64(current_val);
            if state.index == SENTINEL {
                return None;
            }

            let next_idx = self.next[state.index as usize].load(Ordering::Acquire);

            let new_state = PackedState {
                index: next_idx,
                generation: state.generation.wrapping_add(1),
            };

            if head
                .compare_exchange_weak(
                    current_val,
                    new_state.to_u64(),
                    Ordering::Release,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return Some(state.index);
            }
        }
    }

    /// Allocate an object from the pool or create new if pool is empty
    pub fn allocate(&self) -> T
    where
        T: Default,
    {
        if let Some(index) = self.pop_list(&self.occupied_head) {
            // Safety: We popped the index from occupied_head, so we have exclusive ownership
            // of the slot until we push it to free_head.
            let item = unsafe { (*self.slots[index as usize].get()).take() };
            self.push_list(&self.free_head, index);
            self.size.fetch_sub(1, Ordering::Release);
            item.unwrap_or_else(T::default)
        } else {
            T::default()
        }
    }

    /// Return an object to the pool for reuse
    pub fn deallocate(&self, item: T) {
        if self.size.load(Ordering::Acquire) >= self.max_size {
            // Pool is full, just drop the item
            return;
        }

        if let Some(index) = self.pop_list(&self.free_head) {
            // Safety: We popped the index from free_head, so we have exclusive ownership
            // of the slot until we push it to occupied_head.
            unsafe {
                *self.slots[index as usize].get() = Some(item);
            }
            self.push_list(&self.occupied_head, index);
            self.size.fetch_add(1, Ordering::Release);
        }
    }

    /// Get current pool size
    pub fn size(&self) -> usize {
        self.size.load(Ordering::Acquire)
    }
}

/// Global memory pool manager for reduced allocation overhead
pub struct GlobalMemoryManager {
    /// Pools for common vector capacity classes
    pools: [MemoryPool<Vec<u8>>; 8],
}

impl GlobalMemoryManager {
    /// Get the global memory manager instance
    pub fn instance() -> &'static Self {
        use std::sync::OnceLock;
        static INSTANCE: OnceLock<GlobalMemoryManager> = OnceLock::new();

        INSTANCE.get_or_init(|| GlobalMemoryManager {
            pools: [
                MemoryPool::new(1024), // 8 byte pool
                MemoryPool::new(1024), // 16 byte pool
                MemoryPool::new(1024), // 32 byte pool
                MemoryPool::new(1024), // 64 byte pool
                MemoryPool::new(512),  // 128 byte pool
                MemoryPool::new(512),  // 256 byte pool
                MemoryPool::new(256),  // 512 byte pool
                MemoryPool::new(256),  // 1024 byte pool
            ],
        })
    }

    /// Allocate from appropriate pool based on size
    pub fn allocate(&self, size: usize) -> Option<Vec<u8>> {
        let pool_index = match size {
            1..=8 => 0,
            9..=16 => 1,
            17..=32 => 2,
            33..=64 => 3,
            65..=128 => 4,
            129..=256 => 5,
            257..=512 => 6,
            513..=1024 => 7,
            _ => return None, // Too large for pooling
        };

        let mut vec = self.pools[pool_index].allocate();
        let target_capacity = match pool_index {
            0 => 8,
            1 => 16,
            2 => 32,
            3 => 64,
            4 => 128,
            5 => 256,
            6 => 512,
            7 => 1024,
            _ => unreachable!(),
        };

        if vec.capacity() < target_capacity {
            vec = Vec::with_capacity(target_capacity);
        }
        vec.resize(size, 0u8);
        Some(vec)
    }

    /// Return a vector to the appropriate pool based on capacity
    pub fn deallocate(&self, mut vec: Vec<u8>) {
        let capacity = vec.capacity();
        let pool_index = match capacity {
            8..=15 => 0,
            16..=31 => 1,
            32..=63 => 2,
            64..=127 => 3,
            128..=255 => 4,
            256..=511 => 5,
            512..=1023 => 6,
            1024..=2048 => 7,
            _ => return,
        };
        vec.clear();
        self.pools[pool_index].deallocate(vec);
    }
}
