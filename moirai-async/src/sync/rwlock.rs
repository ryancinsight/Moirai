//! Async-aware RwLock for concurrent read/exclusive write access
//!
//! Provides an async-compatible RwLock that allows multiple concurrent readers
//! or a single writer, following SLAP principle design.

use std::cell::UnsafeCell;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Async-aware RwLock
pub struct RwLock<T> {
    data: UnsafeCell<T>,
    state: Arc<Mutex<RwLockState>>,
}

unsafe impl<T: Send + Sync> Sync for RwLock<T> {}
unsafe impl<T: Send> Send for RwLock<T> {}

struct RwWaiter {
    waker: Waker,
    /// Set when the lock has been handed to this waiter.
    granted: bool,
}

struct RwLockState {
    readers: usize,
    writer: bool,
    /// Reader and writer waiters keyed by a monotonic id. Keyed (rather than a
    /// linear `VecDeque`) so per-waiter `poll`/drop lookups and removals are
    /// O(log n) instead of O(n), shortening the time the single state lock is
    /// held under contention. Monotonic ids keep in-order iteration FIFO, so a
    /// released lock goes to the oldest waiter first.
    read_waiters: BTreeMap<u64, RwWaiter>,
    write_waiters: BTreeMap<u64, RwWaiter>,
    next_id: u64,
}

impl RwLockState {
    /// Grant the lock to the oldest ungranted writer, marking `writer`, and
    /// return its waker to wake. Returns `None` if no writer is waiting.
    fn grant_oldest_writer(&mut self) -> Option<Waker> {
        let waker = {
            let waiter = self.write_waiters.values_mut().find(|w| !w.granted)?;
            waiter.granted = true;
            waiter.waker.clone()
        };
        self.writer = true;
        Some(waker)
    }
}

impl<T> RwLock<T> {
    /// Create a new async RwLock
    pub fn new(data: T) -> Self {
        Self {
            data: UnsafeCell::new(data),
            state: Arc::new(Mutex::new(RwLockState {
                readers: 0,
                writer: false,
                read_waiters: BTreeMap::new(),
                write_waiters: BTreeMap::new(),
                next_id: 0,
            })),
        }
    }

    /// Acquire a read lock asynchronously
    pub fn read(&self) -> RwLockReadFuture<'_, T> {
        RwLockReadFuture {
            lock: self,
            id: None,
        }
    }

    /// Acquire a write lock asynchronously
    pub fn write(&self) -> RwLockWriteFuture<'_, T> {
        RwLockWriteFuture {
            lock: self,
            id: None,
        }
    }

    /// Try to acquire a read lock immediately
    pub fn try_read(&self) -> Option<RwLockReadGuard<'_, T>> {
        let mut state = self.state.lock().unwrap();
        if !state.writer && state.write_waiters.is_empty() {
            state.readers += 1;
            Some(RwLockReadGuard { lock: self })
        } else {
            None
        }
    }

    /// Try to acquire a write lock immediately
    pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, T>> {
        let mut state = self.state.lock().unwrap();
        if state.readers == 0 && !state.writer {
            state.writer = true;
            Some(RwLockWriteGuard { lock: self })
        } else {
            None
        }
    }

    fn release_read(&self) {
        let mut state = self.state.lock().unwrap();
        state.readers -= 1;
        if state.readers == 0 {
            let waker = state.grant_oldest_writer();
            drop(state);
            if let Some(w) = waker {
                w.wake();
            }
        }
    }

    fn release_write(&self) {
        let mut state = self.state.lock().unwrap();
        state.writer = false;

        // Prefer waking every pending reader (reader batch); only if there are
        // none, hand the lock to the oldest waiting writer.
        let mut reader_wakers = Vec::new();
        for waiter in state.read_waiters.values_mut() {
            if !waiter.granted {
                waiter.granted = true;
                reader_wakers.push(waiter.waker.clone());
            }
        }

        if !reader_wakers.is_empty() {
            state.readers += reader_wakers.len();
            drop(state);
            for waker in reader_wakers {
                waker.wake();
            }
        } else {
            let waker = state.grant_oldest_writer();
            drop(state);
            if let Some(w) = waker {
                w.wake();
            }
        }
    }
}

/// Future for async read lock acquisition
pub struct RwLockReadFuture<'a, T> {
    lock: &'a RwLock<T>,
    id: Option<u64>,
}

impl<'a, T> Future for RwLockReadFuture<'a, T> {
    type Output = RwLockReadGuard<'a, T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.lock.state.lock().unwrap();

        // 1. Check if we were already registered and have been granted the lock
        if let Some(id) = self.id {
            match state.read_waiters.get(&id).map(|w| w.granted) {
                Some(true) => {
                    state.read_waiters.remove(&id);
                    self.id = None;
                    return Poll::Ready(RwLockReadGuard { lock: self.lock });
                }
                Some(false) => {
                    state.read_waiters.get_mut(&id).unwrap().waker = cx.waker().clone();
                    return Poll::Pending;
                }
                None => {} // registration lost; fall through
            }
        }

        // 2. Try to acquire the read lock
        if !state.writer && state.write_waiters.is_empty() {
            state.readers += 1;
            if let Some(id) = self.id.take() {
                state.read_waiters.remove(&id);
            }
            return Poll::Ready(RwLockReadGuard { lock: self.lock });
        }

        // 3. Register as a reader waiter
        if self.id.is_none() {
            let id = state.next_id;
            state.next_id += 1;
            state.read_waiters.insert(
                id,
                RwWaiter {
                    waker: cx.waker().clone(),
                    granted: false,
                },
            );
            self.id = Some(id);
        }

        Poll::Pending
    }
}

impl<'a, T> Drop for RwLockReadFuture<'a, T> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            if let Ok(mut state) = self.lock.state.lock() {
                if state.read_waiters.remove(&id).is_some_and(|w| w.granted) {
                    drop(state);
                    self.lock.release_read();
                }
            }
        }
    }
}

/// Future for async write lock acquisition
pub struct RwLockWriteFuture<'a, T> {
    lock: &'a RwLock<T>,
    id: Option<u64>,
}

impl<'a, T> Future for RwLockWriteFuture<'a, T> {
    type Output = RwLockWriteGuard<'a, T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.lock.state.lock().unwrap();

        // 1. Check if we were already registered and have been granted the lock
        if let Some(id) = self.id {
            match state.write_waiters.get(&id).map(|w| w.granted) {
                Some(true) => {
                    state.write_waiters.remove(&id);
                    self.id = None;
                    return Poll::Ready(RwLockWriteGuard { lock: self.lock });
                }
                Some(false) => {
                    state.write_waiters.get_mut(&id).unwrap().waker = cx.waker().clone();
                    return Poll::Pending;
                }
                None => {} // registration lost; fall through
            }
        }

        // 2. Try to acquire the write lock
        if state.readers == 0 && !state.writer {
            state.writer = true;
            if let Some(id) = self.id.take() {
                state.write_waiters.remove(&id);
            }
            return Poll::Ready(RwLockWriteGuard { lock: self.lock });
        }

        // 3. Register as a writer waiter
        if self.id.is_none() {
            let id = state.next_id;
            state.next_id += 1;
            state.write_waiters.insert(
                id,
                RwWaiter {
                    waker: cx.waker().clone(),
                    granted: false,
                },
            );
            self.id = Some(id);
        }

        Poll::Pending
    }
}

impl<'a, T> Drop for RwLockWriteFuture<'a, T> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            if let Ok(mut state) = self.lock.state.lock() {
                if state.write_waiters.remove(&id).is_some_and(|w| w.granted) {
                    drop(state);
                    self.lock.release_write();
                }
            }
        }
    }
}

/// Guard for RwLock read access
pub struct RwLockReadGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<'a, T> std::ops::Deref for RwLockReadGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T> Drop for RwLockReadGuard<'a, T> {
    fn drop(&mut self) {
        self.lock.release_read();
    }
}

/// Guard for RwLock write access
pub struct RwLockWriteGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<'a, T> std::ops::Deref for RwLockWriteGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T> std::ops::DerefMut for RwLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<'a, T> Drop for RwLockWriteGuard<'a, T> {
    fn drop(&mut self) {
        self.lock.release_write();
    }
}

#[cfg(test)]
mod tests {
    use super::RwLock;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct NoopWake;

    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }

    fn poll_future<F>(future: &mut F) -> Poll<F::Output>
    where
        F: Future + Unpin,
    {
        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        Pin::new(future).poll(&mut context)
    }

    #[test]
    fn last_reader_release_grants_first_waiting_writer() {
        let lock = RwLock::new(5_u32);
        let reader = lock.try_read().expect("read lock must be acquired");
        let mut writer = lock.write();

        assert!(matches!(poll_future(&mut writer), Poll::Pending));

        drop(reader);

        match poll_future(&mut writer) {
            Poll::Ready(mut guard) => {
                *guard += 7;
            }
            Poll::Pending => panic!("writer waiter must be granted after final reader release"),
        }

        let reader = lock
            .try_read()
            .expect("read lock must be acquired after writer release");
        assert_eq!(*reader, 12);
    }

    #[test]
    fn writer_release_grants_all_registered_readers() {
        let lock = RwLock::new(11_u32);
        let writer = lock.try_write().expect("write lock must be acquired");
        let mut first_reader = lock.read();
        let mut second_reader = lock.read();

        assert!(matches!(poll_future(&mut first_reader), Poll::Pending));
        assert!(matches!(poll_future(&mut second_reader), Poll::Pending));

        drop(writer);

        let first_guard = match poll_future(&mut first_reader) {
            Poll::Ready(guard) => guard,
            Poll::Pending => panic!("first reader waiter must be granted after writer release"),
        };
        let second_guard = match poll_future(&mut second_reader) {
            Poll::Ready(guard) => guard,
            Poll::Pending => panic!("second reader waiter must be granted after writer release"),
        };

        assert_eq!(*first_guard, 11);
        assert_eq!(*second_guard, 11);
        assert!(
            lock.try_write().is_none(),
            "active granted readers must exclude writers"
        );

        drop(first_guard);
        drop(second_guard);

        let mut writer = lock
            .try_write()
            .expect("write lock must be acquired after readers release");
        *writer = 19;
        drop(writer);

        let reader = lock
            .try_read()
            .expect("read lock must be acquired after writer release");
        assert_eq!(*reader, 19);
    }
}
