//! Notification primitive for efficient async task coordination
//!
//! Provides notification mechanisms for waking up waiting async tasks,
//! following SLAP principle with focused responsibility. Waiter-queue
//! mechanics live in `WaitQueue`; this module keeps only the notify
//! admission state (the stored single permit) and the grant-restoration
//! policy for cancelled futures.

#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll};

use crate::sync::wait_queue::{WaitQueue, WaiterPoll};

/// Grant payload distinguishing how a waiter was notified: a `notify_one`
/// grant is a transferable single permit (restored to the next waiter or the
/// stored-permit slot when the granted future is cancelled), while a
/// `notify_waiters` grant is a broadcast wakeup that is not restored.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NotifyGrant {
    One,
    All,
}

/// Notification primitive for efficient task coordination
pub struct Notify {
    state: Mutex<NotifyState>,
}

struct NotifyState {
    /// Single stored permit from a `notify_one` issued with no waiters.
    notified: bool,
    waiters: WaitQueue<NotifyGrant>,
}

impl Notify {
    /// Create a new notification primitive
    pub fn new() -> Self {
        Self {
            state: Mutex::new(NotifyState {
                notified: false,
                waiters: WaitQueue::new(),
            }),
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
        match state.waiters.grant_oldest(NotifyGrant::One) {
            Some(waker) => waker.wake(),
            None => state.notified = true,
        }
    }

    /// Notify all waiting tasks.
    ///
    /// Wakes every currently-registered waiter. This is independent of the
    /// single-permit `notify_one` mechanism: a permit stored by a prior
    /// `notify_one` (issued with no waiters present) is left intact, so a
    /// subsequent `notified()` still observes it.
    pub fn notify_waiters(&self) {
        let mut state = self.state.lock().unwrap();
        let wakers = state.waiters.grant_all(NotifyGrant::All);
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

        // 1. Check if a stored permit is available
        if state.notified {
            state.notified = false;
            if let Some(id) = self.id.take() {
                // Parity with the pre-consolidation state machine: the entry
                // is removed regardless of grant state, so a `One` grant on
                // our own entry is consumed together with the stored permit.
                let _removed_grant = state.waiters.deregister(id);
            }
            return Poll::Ready(());
        }

        // 2. Check if already registered
        if let Some(id) = self.id {
            match state.waiters.poll_waiter(id, cx.waker()) {
                WaiterPoll::Granted(_) => {
                    self.id = None;
                    return Poll::Ready(());
                }
                WaiterPoll::Pending => return Poll::Pending,
                // Our registration was lost; re-register below.
                WaiterPoll::NotRegistered => {}
            }
        }

        // 3. Register a (new) waiter.
        self.id = Some(state.waiters.register(cx.waker().clone()));
        Poll::Pending
    }
}

impl<'a> Drop for NotifyFuture<'a> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            if let Ok(mut state) = self.notify.state.lock() {
                // If we were holding a single-task permit but never observed
                // it, hand it to the next pending waiter (or store it) so it
                // is not lost. Broadcast (`All`) grants are not restored.
                if state.waiters.deregister(id) == Some(NotifyGrant::One) {
                    match state.waiters.grant_oldest(NotifyGrant::One) {
                        Some(waker) => waker.wake(),
                        None => state.notified = true,
                    }
                }
            }
        }
    }
}
