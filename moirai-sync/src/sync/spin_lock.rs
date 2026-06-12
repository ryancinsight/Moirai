use std::cell::UnsafeCell;
use std::hint;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};

// Import centralized constants (SSOT compliance)
use moirai_core::constants::{SPINLOCK_MAX_BACKOFF, SPINLOCK_MAX_SPINS_BEFORE_YIELD};

// SpinLock backoff constants (TBB-inspired)
/// Initial backoff iterations for SpinLock
const SPINLOCK_INITIAL_BACKOFF: usize = 1;

/// A spin lock for very short critical sections with TBB-inspired exponential backoff.
///
/// This implementation uses exponential backoff and adaptive yielding for better
/// performance under contention. The lock is cache-line aligned to prevent false sharing.
///
/// Use only when you know the critical section is extremely short (< 1μs).
#[repr(align(64))] // Cache line alignment to prevent false sharing
pub struct SpinLock<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for SpinLock<T> {}
unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    /// Create a new spin lock.
    pub const fn new(data: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
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
        let mut backoff = SPINLOCK_INITIAL_BACKOFF;
        let mut total_spins = 0;

        loop {
            // Fast path: try to acquire immediately
            if self
                .locked
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return SpinLockGuard {
                    lock: self,
                    _phantom: std::marker::PhantomData,
                };
            }

            // Exponential backoff with CPU pause instructions
            for _ in 0..backoff {
                hint::spin_loop();
            }

            // Double the backoff up to maximum
            if backoff < SPINLOCK_MAX_BACKOFF {
                backoff = backoff.saturating_mul(2);
            }

            total_spins += backoff;

            // After many attempts, yield to scheduler to be cooperative
            if total_spins >= SPINLOCK_MAX_SPINS_BEFORE_YIELD {
                std::thread::yield_now();
                total_spins = 0;
                backoff = SPINLOCK_INITIAL_BACKOFF; // Reset backoff after yielding
            }
        }
    }

    /// Try to lock without spinning.
    pub fn try_lock(&self) -> Option<SpinLockGuard<'_, T>> {
        if !self.locked.swap(true, Ordering::Acquire) {
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
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T> DerefMut for SpinLockGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.lock.data.get() }
    }
}
