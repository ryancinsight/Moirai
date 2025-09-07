//! Async-aware RwLock for concurrent read/exclusive write access
//!
//! Provides an async-compatible RwLock that allows multiple concurrent readers
//! or a single writer, following SLAP principle design.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Async-aware RwLock
pub struct RwLock<T> {
    inner: std::sync::RwLock<T>,
    read_waiters: Arc<Mutex<VecDeque<Waker>>>,
    write_waiters: Arc<Mutex<VecDeque<Waker>>>,
}

impl<T> RwLock<T> {
    /// Create a new async RwLock
    pub fn new(data: T) -> Self {
        Self {
            inner: std::sync::RwLock::new(data),
            read_waiters: Arc::new(Mutex::new(VecDeque::new())),
            write_waiters: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Acquire a read lock asynchronously
    pub fn read(&self) -> RwLockReadFuture<'_, T> {
        RwLockReadFuture {
            lock: self,
            registered: false,
        }
    }

    /// Acquire a write lock asynchronously
    pub fn write(&self) -> RwLockWriteFuture<'_, T> {
        RwLockWriteFuture {
            lock: self,
            registered: false,
        }
    }

    /// Try to acquire a read lock immediately
    pub fn try_read(&self) -> Option<std::sync::RwLockReadGuard<'_, T>> {
        self.inner.try_read().ok()
    }

    /// Try to acquire a write lock immediately
    pub fn try_write(&self) -> Option<std::sync::RwLockWriteGuard<'_, T>> {
        self.inner.try_write().ok()
    }
}

/// Future for async read lock acquisition
pub struct RwLockReadFuture<'a, T> {
    lock: &'a RwLock<T>,
    registered: bool,
}

impl<'a, T> Future for RwLockReadFuture<'a, T> {
    type Output = std::sync::RwLockReadGuard<'a, T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Ok(guard) = self.lock.inner.try_read() {
            Poll::Ready(guard)
        } else {
            if !self.registered {
                let mut waiters = self.lock.read_waiters.lock().unwrap();
                waiters.push_back(cx.waker().clone());
                self.registered = true;
            }
            Poll::Pending
        }
    }
}

/// Future for async write lock acquisition
pub struct RwLockWriteFuture<'a, T> {
    lock: &'a RwLock<T>,
    registered: bool,
}

impl<'a, T> Future for RwLockWriteFuture<'a, T> {
    type Output = std::sync::RwLockWriteGuard<'a, T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Ok(guard) = self.lock.inner.try_write() {
            Poll::Ready(guard)
        } else {
            if !self.registered {
                let mut waiters = self.lock.write_waiters.lock().unwrap();
                waiters.push_back(cx.waker().clone());
                self.registered = true;
            }
            Poll::Pending
        }
    }
}