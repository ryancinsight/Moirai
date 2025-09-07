//! Notification primitive for efficient async task coordination
//!
//! Provides notification mechanisms for waking up waiting async tasks,
//! following SLAP principle with focused responsibility.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Notification primitive for efficient task coordination
pub struct Notify {
    waiters: Arc<Mutex<VecDeque<Waker>>>,
}

impl Notify {
    /// Create a new notification primitive
    pub fn new() -> Self {
        Self {
            waiters: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Wait for a notification
    pub fn notified(&self) -> NotifyFuture<'_> {
        NotifyFuture {
            notify: self,
            registered: false,
        }
    }

    /// Notify one waiting task
    pub fn notify_one(&self) {
        let mut waiters = self.waiters.lock().unwrap();
        if let Some(waker) = waiters.pop_front() {
            drop(waiters);
            waker.wake();
        }
    }

    /// Notify all waiting tasks
    pub fn notify_waiters(&self) {
        let mut waiters = self.waiters.lock().unwrap();
        let wakers: Vec<_> = waiters.drain(..).collect();
        drop(waiters);
        
        for waker in wakers {
            waker.wake();
        }
    }
}

impl Default for Notify {
    fn default() -> Self {
        Self::new()
    }
}

/// Future for waiting on notifications
pub struct NotifyFuture<'a> {
    notify: &'a Notify,
    registered: bool,
}

impl<'a> Future for NotifyFuture<'a> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.registered {
            let mut waiters = self.notify.waiters.lock().unwrap();
            waiters.push_back(cx.waker().clone());
            self.registered = true;
            Poll::Pending
        } else {
            // This future has already been registered and woken
            Poll::Ready(())
        }
    }
}