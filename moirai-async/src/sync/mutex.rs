use std::cell::UnsafeCell;
use std::future::Future;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::sync::wait_queue::{WaitQueue, WaiterPoll};

/// Async mutual-exclusion lock over `T`.
pub struct Mutex<T> {
    data: UnsafeCell<T>,
    state: std::sync::Mutex<MutexState>,
}

unsafe impl<T: Send + Sync> Sync for Mutex<T> {}
unsafe impl<T: Send> Send for Mutex<T> {}

struct MutexState {
    locked: bool,
    waiters: WaitQueue<()>,
}

impl<T> Mutex<T> {
    /// Create an unlocked mutex owning `data`.
    pub fn new(data: T) -> Self {
        Self {
            data: UnsafeCell::new(data),
            state: std::sync::Mutex::new(MutexState {
                locked: false,
                waiters: WaitQueue::new(),
            }),
        }
    }

    /// Acquire the lock, waiting for the current holder to release.
    pub fn lock(&self) -> MutexLockFuture<'_, T> {
        MutexLockFuture {
            mutex: self,
            id: None,
        }
    }

    /// Acquire without waiting; `None` when already held.
    pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        let mut state = self.state.lock().unwrap();
        if !state.locked {
            state.locked = true;
            Some(MutexGuard { mutex: self })
        } else {
            None
        }
    }

    fn release(&self) {
        let mut state = self.state.lock().unwrap();
        match state.waiters.grant_oldest(()) {
            Some(waker) => waker.wake(),
            None => state.locked = false,
        }
    }
}

impl<T: Default> Default for Mutex<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> From<T> for Mutex<T> {
    fn from(data: T) -> Self {
        Self::new(data)
    }
}

/// Future returned by [`Mutex::lock`].
pub struct MutexLockFuture<'a, T> {
    mutex: &'a Mutex<T>,
    id: Option<u64>,
}

impl<'a, T> Future for MutexLockFuture<'a, T> {
    type Output = MutexGuard<'a, T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.mutex.state.lock().unwrap();

        if let Some(id) = self.id {
            match state.waiters.poll_waiter(id, cx.waker()) {
                WaiterPoll::Granted(()) => {
                    self.id = None;
                    return Poll::Ready(MutexGuard { mutex: self.mutex });
                }
                WaiterPoll::Pending => return Poll::Pending,
                WaiterPoll::NotRegistered => {}
            }
        }

        if !state.locked {
            state.locked = true;
            if let Some(id) = self.id.take() {
                let _removed = state.waiters.deregister(id);
            }
            return Poll::Ready(MutexGuard { mutex: self.mutex });
        }

        if self.id.is_none() {
            self.id = Some(state.waiters.register(cx.waker().clone()));
        }

        Poll::Pending
    }
}

impl<'a, T> Drop for MutexLockFuture<'a, T> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            if let Ok(mut state) = self.mutex.state.lock() {
                if state.waiters.deregister(id).is_some() {
                    drop(state);
                    self.mutex.release();
                }
            }
        }
    }
}

/// Exclusive access guard; releases the lock on drop.
pub struct MutexGuard<'a, T> {
    pub(crate) mutex: &'a Mutex<T>,
}

impl<'a, T> Deref for MutexGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.mutex.data.get() }
    }
}

impl<'a, T> DerefMut for MutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.mutex.data.get() }
    }
}

impl<'a, T> Drop for MutexGuard<'a, T> {
    fn drop(&mut self) {
        self.mutex.release();
    }
}

#[cfg(test)]
mod tests {
    use super::Mutex;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, Waker};

    fn poll_future<F: Future + Unpin>(future: &mut F) -> Poll<F::Output> {
        let mut context = Context::from_waker(Waker::noop());
        Pin::new(future).poll(&mut context)
    }

    #[test]
    fn test_mutex_lock_unlock() {
        let lock = Mutex::new(42_u32);
        let mut guard = lock.try_lock().expect("lock must succeed");
        assert_eq!(*guard, 42);
        *guard = 7;
        drop(guard);
        let guard = lock.try_lock().expect("lock must succeed after drop");
        assert_eq!(*guard, 7);
    }

    #[test]
    fn test_mutex_async_lock_release_grants_waiter() {
        let lock = Mutex::new(10_u32);
        let guard = lock.try_lock().expect("lock must succeed");
        let mut waiter = lock.lock();
        assert!(matches!(poll_future(&mut waiter), Poll::Pending));
        drop(guard);
        match poll_future(&mut waiter) {
            Poll::Ready(mut guard) => *guard += 5,
            Poll::Pending => panic!("waiter must be granted after release"),
        }
        let guard = lock.try_lock().expect("lock must succeed after waiter");
        assert_eq!(*guard, 15);
    }

    #[test]
    fn test_mutex_cancellation_safety() {
        let lock = Mutex::new(0_u32);
        let guard = lock.try_lock().expect("lock must succeed");
        let mut waiter = lock.lock();
        assert!(matches!(poll_future(&mut waiter), Poll::Pending));
        drop(waiter);
        drop(guard);
        let guard = lock
            .try_lock()
            .expect("lock must be available after cancel+release");
        assert_eq!(*guard, 0);
    }

    #[test]
    fn test_mutex_cancellation_restores_permit() {
        let lock = Mutex::new(0_u32);
        let guard = lock.try_lock().expect("lock must succeed");
        let mut waiter = lock.lock();
        assert!(matches!(poll_future(&mut waiter), Poll::Pending));
        drop(guard);
        drop(waiter);
        let guard = lock.try_lock().expect("lock must be available");
        assert_eq!(*guard, 0);
    }

    #[test]
    fn test_mutex_exclusive_access() {
        let lock = Mutex::new(Vec::<i32>::new());
        let guard = lock.try_lock().expect("lock must succeed");
        let mut waiter = lock.lock();
        assert!(matches!(poll_future(&mut waiter), Poll::Pending));
        drop(guard);
        match poll_future(&mut waiter) {
            Poll::Ready(mut guard) => guard.push(1),
            Poll::Pending => panic!("waiter must be granted"),
        }
        assert!(lock.try_lock().is_some());
    }
}
