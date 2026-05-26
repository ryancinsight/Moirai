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
    state: Arc<Mutex<NotifyState>>,
}

struct NotifyState {
    notified: bool,
    waiters: VecDeque<(u64, Waker, bool)>,
    next_id: u64,
}

impl Notify {
    /// Create a new notification primitive
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(NotifyState {
                notified: false,
                waiters: VecDeque::new(),
                next_id: 0,
            })),
        }
    }

    /// Wait for a notification
    pub fn notified(&self) -> NotifyFuture<'_> {
        NotifyFuture {
            notify: self,
            id: None,
        }
    }

    /// Notify one waiting task
    pub fn notify_one(&self) {
        let mut state = self.state.lock().unwrap();
        if let Some(waiter) = state.waiters.iter_mut().find(|(_, _, woken)| !*woken) {
            waiter.2 = true;
            waiter.1.wake_by_ref();
        } else {
            state.notified = true;
        }
    }

    /// Notify all waiting tasks
    pub fn notify_waiters(&self) {
        let mut state = self.state.lock().unwrap();
        state.notified = false;
        let mut wakers = Vec::new();
        for waiter in &mut state.waiters {
            if !waiter.2 {
                waiter.2 = true;
                wakers.push(waiter.1.clone());
            }
        }
        drop(state);

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
    id: Option<u64>,
}

impl<'a> Future for NotifyFuture<'a> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.notify.state.lock().unwrap();

        // 1. Check if a permit is available
        if state.notified {
            state.notified = false;
            if let Some(id) = self.id.take() {
                state.waiters.retain(|(w_id, _, _)| *w_id != id);
            }
            return Poll::Ready(());
        }

        // 2. Check if already registered
        if let Some(id) = self.id {
            if let Some(pos) = state.waiters.iter().position(|(w_id, _, _)| *w_id == id) {
                if state.waiters[pos].2 {
                    // Woken! Remove ourselves and return Ready
                    state.waiters.remove(pos);
                    self.id = None;
                    Poll::Ready(())
                } else {
                    // Update waker
                    state.waiters[pos].1 = cx.waker().clone();
                    Poll::Pending
                }
            } else {
                // Re-register if lost
                let new_id = state.next_id;
                state.next_id += 1;
                state.waiters.push_back((new_id, cx.waker().clone(), false));
                self.id = Some(new_id);
                Poll::Pending
            }
        } else {
            // First time polling, register a new waiter
            let id = state.next_id;
            state.next_id += 1;
            state.waiters.push_back((id, cx.waker().clone(), false));
            self.id = Some(id);
            Poll::Pending
        }
    }
}

impl<'a> Drop for NotifyFuture<'a> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            if let Ok(mut state) = self.notify.state.lock() {
                state.waiters.retain(|(w_id, _, _)| *w_id != id);
            }
        }
    }
}
