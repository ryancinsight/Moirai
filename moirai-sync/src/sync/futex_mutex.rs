use std::cell::UnsafeCell;
use std::hint;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_os = "linux")]
use std::sync::atomic::AtomicI32;

// Import centralized constants (SSOT compliance)
use moirai_core::constants::MAX_SPIN_ATTEMPTS;

#[cfg(target_os = "linux")]
mod futex {
    // Linux futex operations
    const FUTEX_WAIT: i32 = 0;
    const FUTEX_WAKE: i32 = 1;

    /// Wait on a futex if the value matches expected
    pub fn futex_wait(addr: *const i32, expected: i32) -> i32 {
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
    data: UnsafeCell<T>,
}

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
            !self.locked.swap(true, Ordering::Acquire)
        }
    }

    #[cold]
    fn lock_slow(&self) {
        #[cfg(target_os = "linux")]
        {
            loop {
                let state = self.state.load(Ordering::Relaxed);

                if state == 0
                    && self
                        .state
                        .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
                        .is_ok()
                {
                    return;
                }

                if state == 1 {
                    self.state
                        .compare_exchange_weak(1, 2, Ordering::Relaxed, Ordering::Relaxed)
                        .ok();
                }

                futex::futex_wait(self.state.as_ptr(), 2);
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            while self.locked.load(Ordering::Relaxed) {
                std::thread::yield_now();
            }
            while self.locked.swap(true, Ordering::Acquire) {
                while self.locked.load(Ordering::Relaxed) {
                    std::thread::yield_now();
                }
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
        }
    }
}

/// Guard for FutexMutex that automatically unlocks on drop.
pub struct FutexMutexGuard<'a, T> {
    mutex: &'a FutexMutex<T>,
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
        unsafe { &*self.mutex.data.get() }
    }
}

impl<'a, T> DerefMut for FutexMutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.mutex.data.get() }
    }
}
