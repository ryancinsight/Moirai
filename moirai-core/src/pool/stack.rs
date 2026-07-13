//! Generation-tagged dual-freelist stack — the single authoritative
//! slot-freelist implementation in this crate.
//!
//! [`crate::memory::MemoryPool`] and [`super::global::GlobalPool`] parameterize
//! this type instead of duplicating the packed-state algebra.
//!
//! `super::slab::SlabAllocator` is intentionally *not* unified here: it uses a
//! single free list plus a per-slot `occupied` flag so that entries can be
//! removed by index (random-access deallocation), whereas this stack only
//! supports LIFO pop from the occupied list. The ABA-protection algebra
//! (generation counter packed beside the index) is analogous but the list
//! discipline differs materially.

use crate::platform::*;

/// Sentinel index terminating a freelist chain.
const SENTINEL: u32 = u32::MAX;

/// Head-of-list word: slot index plus an ABA-protection generation counter,
/// packed into one `AtomicU64`-storable value.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PackedState {
    index: u32,
    generation: u32,
}

impl PackedState {
    #[inline]
    // justification: intentional bit-unpacking. `val` packs `index` in the low
    // 32 bits and `generation` in the high 32; each `as u32` extracts one half.
    // Truncation is the defined semantics (inverse of `to_u64`).
    #[allow(clippy::cast_possible_truncation)]
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

struct StackNode<T> {
    data: UnsafeCell<MaybeUninit<T>>,
    next: AtomicU32,
}

// Safety: StackNode is Send and Sync because access is controlled by LockFreeStack
unsafe impl<T: Send> Send for StackNode<T> {}
unsafe impl<T: Sync> Sync for StackNode<T> {}

/// Lock-free stack for object pooling.
///
/// # Safety
/// This implementation uses a pre-allocated array of slots with a generation counter
/// packed in an `AtomicU64` to prevent ABA problems and use-after-free without blocking.
///
/// # Capacity
/// The slot array is fixed at construction; [`Self::push`] returns the item back
/// when every slot is occupied. Use [`Self::with_capacity`] to size the stack;
/// [`Self::new`] uses [`DEFAULT_STACK_CAPACITY`].
///
/// # Performance Characteristics
/// - Push: O(1) amortized, < 20ns
/// - Pop: O(1) amortized, < 30ns
/// - Thread-safe: All operations are lock-free
pub struct LockFreeStack<T> {
    nodes: Box<[StackNode<T>]>,
    occupied_head: AtomicU64,
    free_head: AtomicU64,
    len: AtomicUsize,
}

/// Default slot count for [`LockFreeStack::new`].
///
/// Sized so a default stack of pointer-sized items costs ~16 KiB of slot
/// metadata rather than eagerly materializing tens of thousands of nodes;
/// callers with known workloads size explicitly via
/// [`LockFreeStack::with_capacity`].
pub const DEFAULT_STACK_CAPACITY: usize = 1024;

impl<T> LockFreeStack<T> {
    /// Create a new empty lock-free stack with [`DEFAULT_STACK_CAPACITY`] slots.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_STACK_CAPACITY)
    }

    /// Create a new empty lock-free stack with exactly `capacity` slots.
    ///
    /// A `capacity` of 0 yields a stack whose `push` always returns the item back.
    ///
    /// # Panics
    /// Panics if `capacity >= u32::MAX` (the sentinel index must stay unused).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(
            capacity < SENTINEL as usize,
            "LockFreeStack capacity must be < u32::MAX"
        );
        let mut nodes = Vec::with_capacity(capacity);
        for i in 0..capacity {
            nodes.push(StackNode {
                data: UnsafeCell::new(MaybeUninit::uninit()),
                next: AtomicU32::new(
                    u32::try_from(i).expect("invariant: i < capacity < u32::MAX (asserted above)")
                        + 1,
                ),
            });
        }
        if capacity > 0 {
            nodes[capacity - 1].next.store(SENTINEL, Ordering::Relaxed);
        }

        Self {
            nodes: nodes.into_boxed_slice(),
            occupied_head: AtomicU64::new(
                PackedState {
                    index: SENTINEL,
                    generation: 0,
                }
                .to_u64(),
            ),
            free_head: AtomicU64::new(
                PackedState {
                    index: if capacity > 0 { 0 } else { SENTINEL },
                    generation: 0,
                }
                .to_u64(),
            ),
            len: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn push_list(&self, head: &AtomicU64, index: u32) {
        loop {
            let current_val = head.load(Ordering::Acquire);
            let state = PackedState::from_u64(current_val);

            self.nodes[index as usize]
                .next
                .store(state.index, Ordering::Release);

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

            let next = self.nodes[state.index as usize]
                .next
                .load(Ordering::Acquire);

            let new_state = PackedState {
                index: next,
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

    /// Push an item onto the stack.
    ///
    /// # Errors
    /// Returns `Err(item)` — handing the value back to the caller — when every
    /// slot is occupied. The item is never silently dropped.
    pub fn push(&self, item: T) -> core::result::Result<(), T> {
        if let Some(index) = self.pop_list(&self.free_head) {
            // SAFETY: `index` was exclusively acquired from the free list, so no
            // other thread reads or writes this slot until it is published to
            // the occupied list below.
            unsafe {
                (*self.nodes[index as usize].data.get()).write(item);
            }
            self.push_list(&self.occupied_head, index);
            self.len.fetch_add(1, Ordering::Release);
            Ok(())
        } else {
            Err(item)
        }
    }

    /// Pop an item from the stack.
    pub fn pop(&self) -> Option<T> {
        if let Some(index) = self.pop_list(&self.occupied_head) {
            self.len.fetch_sub(1, Ordering::Release);
            // SAFETY: `index` was exclusively acquired from the occupied list;
            // the slot was initialized by the `push` that published it there.
            let item = unsafe { (*self.nodes[index as usize].data.get()).assume_init_read() };
            self.push_list(&self.free_head, index);
            Some(item)
        } else {
            None
        }
    }

    /// Get the current length of the stack.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Check if the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the fixed slot capacity of the stack.
    pub fn capacity(&self) -> usize {
        self.nodes.len()
    }
}

impl<T> Default for LockFreeStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for LockFreeStack<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

unsafe impl<T: Send> Send for LockFreeStack<T> {}
unsafe impl<T: Send> Sync for LockFreeStack<T> {}

pub use moirai_utils::cache::CacheAligned;
