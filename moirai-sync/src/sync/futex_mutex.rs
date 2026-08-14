//! Mutex backed by a Linux futex, with a condvar fallback elsewhere.
//!
//! Both builds present the same API and the same guarantees; only the blocking
//! mechanism differs, so the two protocols are described separately below.
//!
//! # Linux: three-state futex
//!
//! `state` is the classic Drepper encoding — 0 unlocked, 1 locked, 2 locked with
//! waiters. The one rule that is easy to get wrong is stated at `lock_slow`:
//! every acquisition through the slow path takes the lock with `swap(2)` rather
//! than `CAS 0 -> 1`, because unlock only wakes when it observes 2. Acquiring at
//! 1 after a wakeup would erase that marker while other threads are still
//! parked, and the next unlock would skip the wake and strand them.
//!
//! # Elsewhere: `locked` flag plus condvar
//!
//! A waiter registers in `waiters`, then blocks on `condvar` while holding
//! `fallback`; the unlocker clears `locked` and notifies only if `waiters` is
//! non-zero. That pair of accesses is a store-buffer pattern — the waiter stores
//! `waiters` then loads `locked`, the unlocker stores `locked` then loads
//! `waiters` — so both sides carry a `SeqCst` fence between their store and
//! their load. Without them each load may miss the other's store, and the
//! unlocker skips a notify for a waiter that is about to sleep forever. Holding
//! `fallback` across the `locked` check and the `condvar.wait` is what keeps a
//! notify from landing in between.
//!
//! # `Send` and `Sync`
//!
//! `FutexMutex<T>` is `Send + Sync` for `T: Send`, matching `std::sync::Mutex`:
//! the lock hands `&mut T` to one thread at a time, so `T` must be able to move
//! between threads, but never needs to be shared by two at once.
//!
//! `FutexMutexGuard` is deliberately `Send`, unlike `std::sync::MutexGuard`.
//! Neither protocol requires the releasing thread to be the acquiring one: the
//! futex state is a plain atomic, and `fallback` is only ever held inside
//! `lock_slow`/`unlock`, never across the guard's lifetime.
//!
//! The guard's `PhantomData<T>` is load-bearing rather than decorative — see the
//! note on the field.

#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use std::cell::UnsafeCell;
use std::fmt;
use std::hint;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::Ordering;

#[cfg(not(target_os = "linux"))]
use std::sync::atomic::AtomicBool;

#[cfg(target_os = "linux")]
use std::sync::atomic::AtomicI32;

/// Maximum generic spin attempts before falling back to blocking
const MAX_SPIN_ATTEMPTS: usize = 64;

#[cfg(target_os = "linux")]
mod futex {
    // Linux futex operations
    const FUTEX_WAIT: i32 = 0;
    const FUTEX_WAKE: i32 = 1;

    /// Wait on a futex if the value matches expected
    pub fn futex_wait(addr: *const i32, expected: i32) -> i32 {
        // SAFETY: `addr` points at the caller's live `AtomicI32`. The kernel
        // reads that word, compares it with `expected`, and blocks only while
        // they match — the comparison is atomic with the sleep, so an unlock
        // landing in between cannot be missed. The null timeout means no
        // deadline, and the trailing arguments are unused by FUTEX_WAIT.
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                addr,
                FUTEX_WAIT,
                expected,
                std::ptr::null::<libc::timespec>(),
                std::ptr::null::<i32>(),
                0,
            ) as i32
        }
    }

    /// Wake up waiters on a futex
    pub fn futex_wake(addr: *const i32, num_waiters: i32) -> i32 {
        // SAFETY: `addr` points at the caller's live `AtomicI32`. FUTEX_WAKE
        // only reads the address as a queue key; it neither loads nor stores
        // the word, and the trailing arguments are unused by this operation.
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                addr,
                FUTEX_WAKE,
                num_waiters,
                std::ptr::null::<libc::timespec>(),
                std::ptr::null::<i32>(),
                0,
            ) as i32
        }
    }
}

/// A futex-backed mutex on Linux with adaptive spinning; falls back to atomic spin on non-Linux.
pub struct FutexMutex<T> {
    #[cfg(target_os = "linux")]
    state: AtomicI32, // 0 = unlocked, 1 = locked, 2 = locked with waiters
    #[cfg(not(target_os = "linux"))]
    locked: AtomicBool,
    #[cfg(not(target_os = "linux"))]
    waiters: std::sync::atomic::AtomicUsize,
    #[cfg(not(target_os = "linux"))]
    fallback: std::sync::Mutex<()>,
    #[cfg(not(target_os = "linux"))]
    condvar: std::sync::Condvar,
    data: UnsafeCell<T>,
}

impl<T> fmt::Debug for FutexMutex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[cfg(target_os = "linux")]
        let state = self.state.load(Ordering::Relaxed);
        #[cfg(not(target_os = "linux"))]
        let locked = self.locked.load(Ordering::Relaxed);

        let mut d = f.debug_struct("FutexMutex");
        #[cfg(target_os = "linux")]
        d.field("state", &state);
        #[cfg(not(target_os = "linux"))]
        d.field("locked", &locked);

        d.finish_non_exhaustive()
    }
}

// SAFETY: the only non-`Sync` field is the `UnsafeCell<T>`, and it is reachable
// solely through a guard, which the lock protocol hands to one thread at a time.
// `T: Send` is therefore the exact bound — ownership of the data moves between
// threads, but is never shared by two at once — and it is the same bound
// `std::sync::Mutex` carries for the same reason.
unsafe impl<T: Send> Send for FutexMutex<T> {}
unsafe impl<T: Send> Sync for FutexMutex<T> {}

impl<T> FutexMutex<T> {
    /// Create a new fast mutex.
    pub const fn new(data: T) -> Self {
        Self {
            #[cfg(target_os = "linux")]
            state: AtomicI32::new(0),
            #[cfg(not(target_os = "linux"))]
            locked: AtomicBool::new(false),
            #[cfg(not(target_os = "linux"))]
            waiters: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(not(target_os = "linux"))]
            fallback: std::sync::Mutex::new(()),
            #[cfg(not(target_os = "linux"))]
            condvar: std::sync::Condvar::new(),
            data: UnsafeCell::new(data),
        }
    }

    /// Lock the mutex with adaptive spinning.
    pub fn lock(&self) -> FutexMutexGuard<'_, T> {
        // Try to acquire the lock with spinning first
        for _ in 0..MAX_SPIN_ATTEMPTS {
            if self.try_lock_immediate() {
                return FutexMutexGuard {
                    mutex: self,
                    _phantom: std::marker::PhantomData,
                };
            }
            hint::spin_loop();
        }

        // Fall back to blocking
        self.lock_slow();
        FutexMutexGuard {
            mutex: self,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Try to lock the mutex without spinning or blocking.
    /// Returns `Some(FutexMutexGuard)` if successful, `None` otherwise.
    pub fn try_lock(&self) -> Option<FutexMutexGuard<'_, T>> {
        if self.try_lock_immediate() {
            Some(FutexMutexGuard {
                mutex: self,
                _phantom: std::marker::PhantomData,
            })
        } else {
            None
        }
    }

    #[inline]
    fn try_lock_immediate(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.state
                .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        }
        #[cfg(not(target_os = "linux"))]
        {
            if self.locked.load(Ordering::Relaxed) {
                false
            } else {
                !self.locked.swap(true, Ordering::Acquire)
            }
        }
    }

    #[cold]
    fn lock_slow(&self) {
        #[cfg(target_os = "linux")]
        {
            // Three-state futex mutex (Drepper / Rust std `futex` mutex):
            // 0 = unlocked, 1 = locked (no waiters), 2 = locked with waiters.
            //
            // A single uncontended attempt may leave the state at 1; every other
            // acquisition through this slow path acquires by `swap(2)`, which
            // conservatively preserves the "waiters present" marker. Acquiring
            // via `CAS 0 -> 1` after a wakeup would erase that marker while other
            // waiters are still parked, so the next `unlock` (which only wakes
            // when `swap(0)` observes 2) would skip the wake and strand them — a
            // lost-wakeup deadlock.
            let mut state = self.state.load(Ordering::Relaxed);
            if state == 0 {
                match self
                    .state
                    .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
                {
                    Ok(_) => return,
                    Err(s) => state = s,
                }
            }

            loop {
                // Mark as contended and check whether it was actually free.
                if state != 2 {
                    state = self.state.swap(2, Ordering::Acquire);
                    if state == 0 {
                        return;
                    }
                }

                // Sleep only while the state is still 2; `futex_wait` rechecks
                // the value atomically, so a concurrent unlock to 0 cannot be
                // missed here.
                futex::futex_wait(self.state.as_ptr(), 2);
                state = self.state.load(Ordering::Relaxed);
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            self.waiters.fetch_add(1, Ordering::Relaxed);
            // SeqCst fence pairs with the one in `unlock`: it separates this
            // waiter's `waiters` store from its `locked` load (the swap below) in
            // the global SeqCst order. Without it the two sides form an unguarded
            // store-buffer pattern (waiter stores `waiters`/loads `locked`,
            // unlocker stores `locked`/loads `waiters`) in which both loads may
            // observe stale values: the unlocker sees `waiters == 0` and skips the
            // notify while this waiter sees `locked == true` and sleeps forever.
            std::sync::atomic::fence(Ordering::SeqCst);
            let mut guard = self.fallback.lock().unwrap();
            loop {
                if !self.locked.swap(true, Ordering::Acquire) {
                    self.waiters.fetch_sub(1, Ordering::Relaxed);
                    return;
                }
                guard = self.condvar.wait(guard).unwrap();
            }
        }
    }

    fn unlock(&self) {
        #[cfg(target_os = "linux")]
        {
            if self.state.swap(0, Ordering::Release) == 2 {
                futex::futex_wake(self.state.as_ptr(), 1);
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            self.locked.store(false, Ordering::Release);
            // SeqCst fence pairs with the one in `lock_slow`: it separates this
            // unlock's `locked` store from the `waiters` load below so the pair of
            // accesses participates in the global SeqCst order. Without it, a
            // StoreLoad reorder could let this load observe `waiters == 0` while a
            // concurrently-registering waiter has not yet been made visible,
            // skipping the wakeup and stranding that waiter on the condvar.
            std::sync::atomic::fence(Ordering::SeqCst);
            if self.waiters.load(Ordering::Acquire) > 0 {
                let _guard = self.fallback.lock().unwrap();
                self.condvar.notify_one();
            }
        }
    }
}

/// Guard for FutexMutex that automatically unlocks on drop.
pub struct FutexMutexGuard<'a, T> {
    mutex: &'a FutexMutex<T>,
    /// Ties the guard's auto traits to `T`, which is what makes `Sync` require
    /// `T: Sync`.
    ///
    /// Without this field the guard's only member would be
    /// `&FutexMutex<T>`, and that is `Sync` for any `T: Send` — so the guard
    /// would be `Sync` too, and `&guard` would hand out `&T` to several threads
    /// for a `T` that cannot be shared. Removing this is a soundness change, not
    /// a cleanup; `guard_sync_requires_sync_data` fails if it goes.
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> Drop for FutexMutexGuard<'a, T> {
    fn drop(&mut self) {
        self.mutex.unlock();
    }
}

impl<'a, T> Deref for FutexMutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: holding the guard means this thread holds the lock, so no
        // other thread can hold a reference to the data at the same time.
        unsafe { &*self.mutex.data.get() }
    }
}

impl<'a, T> DerefMut for FutexMutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: as `deref`, and `&mut self` rules out any other borrow taken
        // through this guard, so the returned reference is unique.
        unsafe { &mut *self.mutex.data.get() }
    }
}

#[cfg(test)]
mod auto_traits {
    use super::{FutexMutex, FutexMutexGuard};
    use static_assertions::{assert_impl_all, assert_not_impl_any};
    use std::cell::Cell;

    assert_impl_all!(FutexMutex<u32>: Send, Sync);

    // Unlike `std::sync::MutexGuard`, this guard may cross threads: neither
    // protocol requires the releasing thread to be the acquiring one.
    assert_impl_all!(FutexMutexGuard<'static, u32>: Send, Sync);

    /// `Sync` must follow `T`, not the mutex reference.
    ///
    /// `Cell<u32>` is `Send` but not `Sync`, so `&FutexMutex<Cell<u32>>` is
    /// `Sync` on its own. Only the guard's `PhantomData<T>` stops the guard from
    /// inheriting that and handing `&Cell<u32>` to several threads at once, so
    /// this is the assertion that fails if the field is ever dropped.
    #[allow(dead_code)]
    fn guard_sync_requires_sync_data() {
        assert_not_impl_any!(FutexMutexGuard<'static, Cell<u32>>: Sync);
    }
}
