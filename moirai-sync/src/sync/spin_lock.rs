use moirai_utils::CacheAligned;
use std::cell::UnsafeCell;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};

/// Maximum backoff iterations for `SpinLock` (TBB-inspired); the shared
/// schedule doubles `1 << round` hints while the round is below
/// [`SPINLOCK_MAX_BACKOFF_ROUND`] and holds there.
const SPINLOCK_MAX_BACKOFF: usize = 64;

/// [`SPINLOCK_MAX_BACKOFF`] as a round index: the round stops here, so the
/// `1 << round` hint schedule is capped at the maximum backoff.
const SPINLOCK_MAX_BACKOFF_ROUND: usize = SPINLOCK_MAX_BACKOFF.trailing_zeros() as usize;

/// Maximum spin attempts before yielding to scheduler
const SPINLOCK_MAX_SPINS_BEFORE_YIELD: usize = 1000;

/// A spin lock for very short critical sections with TBB-inspired exponential backoff.
///
/// This implementation uses exponential backoff and adaptive yielding for better
/// performance under contention. The lock is cache-line aligned to prevent false sharing.
///
/// Use only when you know the critical section is extremely short (< 1μs).
///
/// `locked` is wrapped in [`CacheAligned`], which both aligns the lock to
/// `moirai_utils::DESTRUCTIVE_INTERFERENCE_SIZE` (so a neighbouring object
/// cannot share its sector) and separates the contended flag from `data` (so a
/// spinning acquirer does not invalidate the payload). The separation is
/// per-target and owned by `moirai-utils`, not a literal here.
pub struct SpinLock<T> {
    locked: CacheAligned<AtomicBool>,
    data: UnsafeCell<T>,
}

impl<T> fmt::Debug for SpinLock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let locked = self.locked.load(Ordering::Relaxed);
        f.debug_struct("SpinLock")
            .field("locked", &locked)
            .finish_non_exhaustive()
    }
}

// SAFETY: guarded values move with the lock; no address-sensitive state
// beyond `T`.
unsafe impl<T: Send> Send for SpinLock<T> {}
// SAFETY: the atomic `locked` bit serializes access, so data references via
// guards are exclusive while held; `T: Send` covers cross-thread transfer.
unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    /// Create a new spin lock.
    pub const fn new(data: T) -> Self {
        Self {
            locked: CacheAligned::new(AtomicBool::new(false)),
            data: UnsafeCell::new(data),
        }
    }

    /// Lock the spin lock with TBB-inspired exponential backoff.
    ///
    /// This implementation uses:
    /// - Read-before-CAS to reduce memory contention
    /// - Exponential backoff starting from 1 iteration up to 64
    /// - Adaptive yielding after prolonged spinning
    pub fn lock(&self) -> SpinLockGuard<'_, T> {
        let mut round = 0usize;
        let mut total_spins = 0usize;

        loop {
            // Read-before-CAS: only attempt atomic write if lock is observed unlocked
            if !self.locked.load(Ordering::Relaxed)
                && self
                    .locked
                    .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
            {
                return SpinLockGuard {
                    lock: self,
                    _phantom: std::marker::PhantomData,
                };
            }

            // Exponential backoff with CPU pause instructions, through the shared
            // schedule: `1 << round` hints, capped at `SPINLOCK_MAX_BACKOFF`.
            let hints =
                moirai_utils::backoff::spin_round::<true>(round.min(SPINLOCK_MAX_BACKOFF_ROUND));

            // The schedule doubled its backoff *after* spinning and charged the
            // doubled value to the yield budget, so charge the next round's hint
            // count while still doubling and the held cap once the round is
            // capped.
            total_spins += (hints << 1).min(SPINLOCK_MAX_BACKOFF);
            round = (round + 1).min(SPINLOCK_MAX_BACKOFF_ROUND);

            // After many attempts, yield to scheduler to be cooperative
            if total_spins >= SPINLOCK_MAX_SPINS_BEFORE_YIELD {
                std::thread::yield_now();
                total_spins = 0;
                round = 0; // Reset the schedule after yielding
            }
        }
    }

    /// Try to lock without spinning.
    pub fn try_lock(&self) -> Option<SpinLockGuard<'_, T>> {
        if !self.locked.load(Ordering::Relaxed)
            && self
                .locked
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        {
            Some(SpinLockGuard {
                lock: self,
                _phantom: std::marker::PhantomData,
            })
        } else {
            None
        }
    }
}

/// Guard for SpinLock that automatically unlocks on drop.
pub struct SpinLockGuard<'a, T> {
    lock: &'a SpinLock<T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> Drop for SpinLockGuard<'a, T> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}

impl<'a, T> Deref for SpinLockGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: holding the guard proves the lock bit is ours, so no other
        // reference to `data` exists; shared reborrow cannot race a writer.
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T> DerefMut for SpinLockGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: unique guard plus the lock bit exclude all other access to
        // `data` for the guard's lifetime.
        unsafe { &mut *self.lock.data.get() }
    }
}
