//! Chase-Lev work-stealing deque.
//!
//! Single owner (`push`/`pop` at the bottom), many thieves (`steal` at the top),
//! implemented per the weak-memory-correct formulation of Lê, Pop, Cohen &
//! Nardelli (PPoPP 2013). `storage.rs` backs the slots; `tests/loom_chase_lev.rs`
//! model-checks the protocol exhaustively under `--cfg loom`.
//!
//! # Memory ordering
//!
//! The protocol synchronizes two indices — `bottom` (owner-advanced) and `top`
//! (thief-advanced) — so that a slot is transferred to exactly one consumer. The
//! happens-before edges each atomic access establishes:
//!
//! - **`push`**: the slot write is published to thieves by the `Release` store to
//!   `bottom`; a thief's `Acquire` load of `bottom` that observes the new value
//!   therefore sees the initialized slot. `top` is read `Acquire` to observe
//!   completed steals before deciding whether to grow.
//! - **`pop`**: `bottom` is decremented (`Relaxed`, owner-private) to claim the
//!   slot, then a `SeqCst` fence orders that store before the `top` load so the
//!   owner and a racing thief cannot both take the last element — the fence pairs
//!   with the thief's `SeqCst` fence, and the last-element tie is resolved by a
//!   `SeqCst` CAS on `top`. On x86/x86_64 (TSO) the fence is skipped when
//!   `bottom - top >= MAX_BATCH_STEAL`, since no steal can then reach the popped
//!   slot (Morrison–Afek); a plain `MOV` load of `top` observes every completed
//!   `lock`-prefixed steal CAS, and an in-flight steal has not yet advanced `top`.
//! - **`steal`**: `top` is read `Acquire`, then a `SeqCst` fence orders it before
//!   the `Acquire` load of `bottom` (pairing with `pop`'s fence); the thief first
//!   claims the slot's generation state, then uses the successful `SeqCst` CAS
//!   to claim the index before reading it. The generation state prevents the
//!   owner from reusing a wrapped slot until the read completes, so a losing
//!   thief never creates a speculative second value. The array pointer is loaded
//!   `Acquire` to pair with `resize`'s `Release` store, so a thief never
//!   dereferences a stale buffer.
//!
//! A resize closes the steal gate, waits for active thieves to leave, copies the
//! live generation state, and then publishes the new buffer. Old buffers freed
//! by `resize` are retired to a guarded list and reclaimed only once no accessor
//! is in-flight (epoch reclamation via the `ReclaimPolicy`), closing the
//! use-after-free window a thief's `Acquire` array load would open.

use super::reclaim::{DeferredReclaim, DequeReclaimPolicy, DequeReclaimState, SharedEpochReclaim};
use moirai_core::CacheAligned;
use std::{
    cell::Cell,
    marker::PhantomData,
    mem::MaybeUninit,
    sync::{
        atomic::{AtomicBool, AtomicIsize, AtomicPtr, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

mod storage;
use storage::Array;

pub(crate) const MIN_DEQUE_CAPACITY: usize = 16;
const MAX_BATCH_STEAL: usize = 16;

// Compile-time guarantee of the false-sharing fix: wrapping `bottom`/`top` in
// `CacheAligned` forces the whole deque to ≥64-byte alignment, so two deques in
// a priority array (`[ChaseLevDeque<_>; N]`) can never share a cache line.
// Alignment is independent of `T`, so `u8` is a representative witness.
const _: () = assert!(core::mem::align_of::<ChaseLevInner<u8, DeferredReclaim>>() >= 64);

// ── StealResult ───────────────────────────────────────────────────────────────

/// Outcome of a steal attempt against another worker's deque.
///
/// [`Empty`](Self::Empty) and [`Retry`](Self::Retry) are deliberately
/// distinct: the first is a fact about the victim, the second is a fact
/// about this attempt. Collapsing them would make a thief either spin on
/// a genuinely empty deque or abandon a victim that still has work.
#[derive(Debug, Clone, PartialEq)]
pub enum StealResult<T> {
    /// An item was taken from the victim.
    Success(T),
    /// The victim held no work; look elsewhere.
    Empty,
    /// The steal lost a race against the owner or another thief. The
    /// victim may still hold work, so retrying the same deque is
    /// worthwhile.
    Retry,
}

/// Allocation-free ownership container returned by a batch steal.
pub struct StolenBatch<T> {
    items: [MaybeUninit<T>; MAX_BATCH_STEAL],
    next: usize,
    len: usize,
}

impl<T> Iterator for StolenBatch<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.len {
            return None;
        }
        let index = self.next;
        self.next += 1;
        // SAFETY: `[next, len)` is initialized and advancing `next` transfers
        // this slot exactly once.
        Some(unsafe { self.items[index].assume_init_read() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.next;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for StolenBatch<T> {}

impl<T> Drop for StolenBatch<T> {
    fn drop(&mut self) {
        for item in &mut self.items[self.next..self.len] {
            // SAFETY: `[next, len)` is the initialized, unconsumed tail.
            unsafe { item.assume_init_drop() };
        }
    }
}

// ── ChaseLevDeque ─────────────────────────────────────────────────────────────

/// The unique bottom-side endpoint of a Chase-Lev work-stealing deque.
///
/// This endpoint is `Send`, but neither `Sync` nor `Clone`; therefore safe code
/// cannot create two concurrent push/pop owners. Use [`Self::stealer`] to
/// create cloneable top-side endpoints.
///
/// ```compile_fail
/// use moirai_scheduler::ChaseLevDeque;
/// let owner = ChaseLevDeque::<usize>::new(16);
/// owner.steal();
/// ```
pub struct ChaseLevDeque<T, P = DeferredReclaim>
where
    P: DequeReclaimPolicy,
{
    pub(crate) inner: Arc<ChaseLevInner<T, P>>,
    not_sync: PhantomData<Cell<()>>,
}

/// A cloneable top-side endpoint of a Chase-Lev work-stealing deque.
///
/// ```compile_fail
/// use moirai_scheduler::ChaseLevDeque;
/// let owner = ChaseLevDeque::<usize>::new(16);
/// let mut stealer = owner.stealer();
/// stealer.push(1);
/// ```
pub struct ChaseLevStealer<T, P = DeferredReclaim>
where
    P: DequeReclaimPolicy,
{
    pub(crate) inner: Arc<ChaseLevInner<T, P>>,
}

pub(crate) struct ChaseLevInner<T, P>
where
    P: DequeReclaimPolicy,
{
    // `bottom` is written only by the owning worker (push/pop); `top` is
    // CAS'd by thieves (steal). Co-locating them on one cache line makes every
    // steal invalidate the owner's `bottom` line and vice versa, so each is
    // isolated to its own 64-byte line. This also forces the whole struct to
    // 64-byte alignment, eliminating false sharing between adjacent deques in
    // the per-worker priority array (`[ChaseLevDeque<_>; PRIORITY_LEVELS]`).
    pub(crate) bottom: CacheAligned<AtomicIsize>,
    pub(crate) top: CacheAligned<AtomicIsize>,
    array: AtomicPtr<Array<T>>,
    retired_arrays: Mutex<Vec<*mut Array<T>>>,
    steal_accesses: AtomicUsize,
    resizing: AtomicBool,
    pub(crate) reclaim: P::State,
    policy: PhantomData<P>,
}

struct StealAccessGuard<'a> {
    accesses: &'a AtomicUsize,
}

impl Drop for StealAccessGuard<'_> {
    fn drop(&mut self) {
        self.accesses.fetch_sub(1, Ordering::SeqCst);
    }
}

struct ResizeGate<'a> {
    resizing: &'a AtomicBool,
}

impl Drop for ResizeGate<'_> {
    fn drop(&mut self) {
        self.resizing.store(false, Ordering::SeqCst);
    }
}

impl<T, P> ChaseLevInner<T, P>
where
    P: DequeReclaimPolicy,
{
    fn enter_steal_access(&self) -> StealAccessGuard<'_> {
        loop {
            if self.resizing.load(Ordering::SeqCst) {
                std::thread::yield_now();
                continue;
            }

            self.steal_accesses.fetch_add(1, Ordering::SeqCst);
            if !self.resizing.load(Ordering::SeqCst) {
                return StealAccessGuard {
                    accesses: &self.steal_accesses,
                };
            }
            self.steal_accesses.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn new(initial_capacity: usize) -> Self {
        let capacity = initial_capacity.next_power_of_two().max(MIN_DEQUE_CAPACITY);
        let array = Box::new(Array::new(capacity, 0));

        Self {
            bottom: CacheAligned::new(AtomicIsize::new(0)),
            top: CacheAligned::new(AtomicIsize::new(0)),
            array: AtomicPtr::new(Box::into_raw(array)),
            retired_arrays: Mutex::new(Vec::new()),
            steal_accesses: AtomicUsize::new(0),
            resizing: AtomicBool::new(false),
            reclaim: P::State::default(),
            policy: PhantomData,
        }
    }

    fn push(&self, item: T) {
        let _guard = self.reclaim.enter();
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Acquire);

        let array_ptr = self.array.load(Ordering::Relaxed);
        // SAFETY: the array pointer is never null after construction, and this
        // owner holds a reclaim guard (`_guard`), so `resize` cannot free the
        // buffer while it is borrowed here. Owner-only access needs no acquire.
        let array = unsafe { &*array_ptr };

        if b.wrapping_sub(t) >= array.capacity() as isize - 1 {
            self.resize();
        }

        // Re-load: `resize` may have installed a larger buffer above.
        let array_ptr = self.array.load(Ordering::Relaxed);
        // SAFETY: as above — non-null, guard-protected, owner-only.
        let array = unsafe { &*array_ptr };

        // The generation claim waits for any thief that still owns the previous
        // occupant of this wrapped slot. It makes the following write disjoint
        // from every in-flight read without allocating per-item nodes.
        array.claim_for_write(b);

        // SAFETY: the generation claim makes this slot owner-exclusive and the
        // slot is uninitialized for generation `b` — `Array::write`'s
        // precondition.
        unsafe {
            array.write(b, item);
        }
        array.publish(b);

        self.bottom.store(b.wrapping_add(1), Ordering::Release);
    }

    fn pop(&self) -> Option<T> {
        let _guard = self.reclaim.enter();
        let b = self.bottom.load(Ordering::Relaxed).wrapping_sub(1);
        let array_ptr = self.array.load(Ordering::Relaxed);
        // SAFETY: non-null after construction and guard-protected against a
        // concurrent `resize` free; `pop` is owner-only.
        let array = unsafe { &*array_ptr };

        self.bottom.store(b, Ordering::Relaxed);

        // Morrison-Afek fence-free pop optimization on TSO (x86/x86_64)
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let t = self.top.load(Ordering::Relaxed);
            if b.wrapping_sub(t) >= MAX_BATCH_STEAL as isize {
                if array.claim(b) {
                    // SAFETY: the generation claim makes this initialized slot
                    // owner-exclusive.
                    let item = unsafe { array.read(b) };
                    array.publish(b);
                    return Some(item);
                }
                self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
                return None;
            }
        }

        std::sync::atomic::fence(Ordering::SeqCst);
        let t = self.top.load(Ordering::Relaxed);

        if b.wrapping_sub(t) > 0 {
            if array.claim(b) {
                // SAFETY: the generation claim makes this initialized slot
                // owner-exclusive.
                let item = unsafe { array.read(b) };
                array.publish(b);
                return Some(item);
            }
            self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
            return None;
        }

        if b.wrapping_sub(t) == 0 {
            if !array.claim(t) {
                self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
                return None;
            }
            if self
                .top
                .compare_exchange_weak(t, t.wrapping_add(1), Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
                // SAFETY: the generation claim and last-element CAS make this
                // initialized slot owner-exclusive.
                let item = unsafe { array.read(b) };
                array.release(b);
                return Some(item);
            }

            array.publish(t);
            self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
            return None;
        }

        self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
        None
    }

    fn steal(&self) -> StealResult<T> {
        let _access = self.enter_steal_access();
        let _guard = self.reclaim.enter();
        let t = self.top.load(Ordering::Acquire);
        std::sync::atomic::fence(Ordering::SeqCst);
        let b = self.bottom.load(Ordering::Acquire);

        if b.wrapping_sub(t) > 0 {
            let array_ptr = self.array.load(Ordering::Acquire);
            // SAFETY: the `Acquire` load pairs with `resize`'s `Release` store, so
            // this is a live buffer; the reclaim guard keeps it from being freed
            // while borrowed.
            let array = unsafe { &*array_ptr };

            if !array.claim(t) {
                return StealResult::Retry;
            }

            if self
                .top
                .compare_exchange_weak(t, t.wrapping_add(1), Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                // SAFETY: the generation claim and successful CAS claim this
                // index against every other thief and the owner; the reclaim
                // guard keeps the array allocation live while the value moves.
                let value = unsafe { array.read(t) };
                array.release(t);
                return StealResult::Success(value);
            }

            array.publish(t);
            return StealResult::Retry;
        }

        StealResult::Empty
    }

    fn steal_batch(&self) -> StealResult<StolenBatch<T>> {
        let mut items: [MaybeUninit<T>; MAX_BATCH_STEAL] =
            [const { MaybeUninit::uninit() }; MAX_BATCH_STEAL];
        let mut count = 0;
        let mut retry = false;

        // Claim each element through the single-item protocol before moving it
        // out of storage. A single atomic range claim can overlap owner pops
        // that advance `bottom` while leaving `top` unchanged, so batching the
        // reads must not bypass the last-item arbitration in `steal`.
        while count < MAX_BATCH_STEAL {
            match self.steal() {
                StealResult::Success(item) => {
                    items[count].write(item);
                    count += 1;
                }
                StealResult::Empty => break,
                StealResult::Retry => {
                    retry = true;
                    break;
                }
            }
        }

        if count == 0 {
            return if retry {
                StealResult::Retry
            } else {
                StealResult::Empty
            };
        }

        StealResult::Success(StolenBatch {
            items,
            next: 0,
            len: count,
        })
    }

    fn len(&self) -> usize {
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        b.wrapping_sub(t).max(0) as usize
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn resize(&self) {
        self.resizing.store(true, Ordering::SeqCst);
        let _resize_gate = ResizeGate {
            resizing: &self.resizing,
        };
        while self.steal_accesses.load(Ordering::SeqCst) != 0 {
            std::hint::spin_loop();
        }

        let old_array_ptr = self.array.load(Ordering::Relaxed);
        // SAFETY: non-null and owner-only (`resize` is reached only from `push`);
        // the reclaim guard and resize gate keep the buffer live and free of
        // in-flight thief accesses.
        let old_array = unsafe { &*old_array_ptr };
        let new_capacity = old_array.capacity() * 2;

        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        let new_array = Box::new(Array::new(new_capacity, b));

        let len = b.wrapping_sub(t);
        for i in 0..len {
            // SAFETY: `i < len = bottom - top`, so slot `t + i` is an initialized,
            // live element being relocated into the fresh (distinct) buffer; the
            // bitwise copy moves ownership without running a destructor.
            unsafe {
                old_array.copy_slot_to(&new_array, t.wrapping_add(i));
            }
        }

        let new_array_ptr = Box::into_raw(new_array);
        self.array.store(new_array_ptr, Ordering::Release);

        let mut retired_arrays = self
            .retired_arrays
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        retired_arrays.push(old_array_ptr);
    }
}

impl<T> ChaseLevInner<T, SharedEpochReclaim> {
    fn try_reclaim_shared(&self) -> bool {
        if !self.reclaim.can_reclaim_shared() {
            return false;
        }

        let mut retired = self
            .retired_arrays
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        if retired.is_empty() {
            return false;
        }

        if self.reclaim.active_accesses() == 0 {
            for array_ptr in retired.drain(..) {
                if !array_ptr.is_null() {
                    unsafe {
                        drop(Box::from_raw(array_ptr));
                    }
                }
            }
            true
        } else {
            false
        }
    }
}

impl<T, P> Drop for ChaseLevInner<T, P>
where
    P: DequeReclaimPolicy,
{
    fn drop(&mut self) {
        // `.0.get_mut()` reaches the inner atomic: `CacheAligned` has its own
        // inherent `get_mut` that would otherwise shadow `AtomicIsize::get_mut`.
        let top = *self.top.0.get_mut();
        let bottom = *self.bottom.0.get_mut();
        let array_ptr = *self.array.get_mut();

        if !array_ptr.is_null() {
            let array = unsafe { Box::from_raw(array_ptr) };
            let len = bottom.wrapping_sub(top);
            for i in 0..len {
                let index = top.wrapping_add(i);
                unsafe {
                    drop(array.read(index));
                }
            }
        }

        let retired_arrays = self
            .retired_arrays
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for array_ptr in retired_arrays.drain(..) {
            unsafe {
                drop(Box::from_raw(array_ptr));
            }
        }
    }
}

unsafe impl<T, P> Send for ChaseLevInner<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
    P::State: Send,
{
}

unsafe impl<T, P> Sync for ChaseLevInner<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
    P::State: Sync,
{
}

impl<T, P> ChaseLevDeque<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
{
    /// Creates an empty deque with at least `initial_capacity` slots.
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            inner: Arc::new(ChaseLevInner::new(initial_capacity)),
            not_sync: PhantomData,
        }
    }

    /// Creates a cloneable top-side stealing endpoint.
    pub fn stealer(&self) -> ChaseLevStealer<T, P> {
        ChaseLevStealer {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Pushes an item at the owner-only bottom side.
    pub fn push(&mut self, item: T) {
        self.inner.push(item);
    }

    /// Pops an item from the owner-only bottom side.
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }

    /// Returns the current advisory length.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the deque is observably empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn retired_array_count(&self) -> usize {
        self.inner
            .retired_arrays
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    #[cfg(test)]
    pub(crate) fn poison_retired_array_lock_for_test(&self) {
        let _guard = self
            .inner
            .retired_arrays
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        panic!("poison retired-array mutex for recovery regression");
    }

    #[cfg(test)]
    pub(crate) fn set_indices_for_test(&self, index: isize) {
        self.inner.top.store(index, Ordering::Relaxed);
        self.inner.bottom.store(index, Ordering::Relaxed);
        let array_ptr = self.inner.array.load(Ordering::Relaxed);
        // SAFETY: tests call this only while the deque is empty and uniquely
        // owned, so resetting the generation markers is owner-exclusive.
        unsafe { &*array_ptr }.reset_states(index);
    }
}

impl<T> ChaseLevDeque<T, SharedEpochReclaim>
where
    T: Send,
{
    /// Reclaims retired arrays if no endpoint operation is active.
    pub fn try_reclaim_shared(&self, _policy: SharedEpochReclaim) -> bool {
        self.inner.try_reclaim_shared()
    }
}

impl<T, P> Clone for ChaseLevStealer<T, P>
where
    P: DequeReclaimPolicy,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T, P> ChaseLevStealer<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
{
    /// Steals one item from the top side.
    pub fn steal(&self) -> StealResult<T> {
        self.inner.steal()
    }

    /// Steals an allocation-free, panic-safe batch from the top side.
    pub fn steal_batch(&self) -> StealResult<StolenBatch<T>> {
        self.inner.steal_batch()
    }

    /// Returns the current advisory length.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the deque is observably empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
