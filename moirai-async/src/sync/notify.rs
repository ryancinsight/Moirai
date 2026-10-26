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
use std::sync::atomic::{AtomicU64, Ordering};
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
    /// Creation snapshots do not allocate or acquire the waiter-state lock.
    broadcasts: AtomicU64,
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
            broadcasts: AtomicU64::new(0),
        }
    }

    /// Wait for a notification.
    ///
    /// A broadcast issued after this future is created is observed even if the
    /// future has not been polled yet. A single-task notification is admitted
    /// when the future is polled, retaining FIFO registered-waiter semantics.
    pub fn notified(&self) -> NotifyFuture<'_> {
        NotifyFuture {
            notify: self,
            id: None,
            broadcasts: self.broadcasts.load(Ordering::Acquire),
        }
    }

    /// Notify one waiting task
    ///
    /// The waker is taken under the state lock and woken after it is released:
    /// `Waker::wake` may poll the task inline on this thread, and that poll
    /// re-locks this state — waking under the lock would self-deadlock. Same
    /// discipline as `notify_waiters` below and `hybrid::notify`.
    pub fn notify_one(&self) {
        let waker = {
            let mut state = self.state.lock().unwrap();
            let waker = state.waiters.grant_oldest(NotifyGrant::One);
            if waker.is_none() {
                state.notified = true;
            }
            waker
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    /// Notify all waiting tasks.
    ///
    /// Wakes every currently-registered waiter and completes futures created
    /// before this call that have not yet been polled. This is independent of the
    /// single-permit `notify_one` mechanism: a permit stored by a prior
    /// `notify_one` (issued with no waiters present) is left intact, so a
    /// subsequent `notified()` still observes it.
    pub fn notify_waiters(&self) {
        let mut state = self.state.lock().unwrap();
        // Linearize the broadcast under the waiter lock. A first poll either
        // sees this generation or registers before grant_all, closing the
        // creation-to-first-poll gap without storing a broadcast permit for
        // futures created later. Release/Acquire binds creation snapshots;
        // waiter admission and grant visibility use the existing mutex.
        self.broadcasts.fetch_add(1, Ordering::Release);
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
    broadcasts: u64,
}

impl<'a> Future for NotifyFuture<'a> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.notify.state.lock().unwrap();

        // Quinn-style protocol drivers create this future while holding their
        // own state lock and first poll it after releasing that lock. A
        // broadcast in between must not be lost or consume a separate stored
        // notify_one permit. Registered futures consume their existing grant.
        if self.id.is_none() && self.broadcasts != self.notify.broadcasts.load(Ordering::Acquire) {
            return Poll::Ready(());
        }

        // Consume an assigned grant before a separate stored permit. Taking
        // both for one completed future would lose the second notification.
        if let Some(id) = self.id {
            match state.waiters.poll_waiter(id, cx.waker()) {
                WaiterPoll::Granted(_) => {
                    self.id = None;
                    return Poll::Ready(());
                }
                WaiterPoll::Pending => return Poll::Pending,
                // Our registration was lost; re-register below.
                WaiterPoll::NotRegistered => self.id = None,
            }
        }

        if state.notified {
            state.notified = false;
            return Poll::Ready(());
        }

        // Register a (new) waiter under the same lock as notify_waiters.
        self.id = Some(state.waiters.register(cx.waker().clone()));
        Poll::Pending
    }
}

#[cfg(test)]
#[path = "notify_tests.rs"]
mod tests;

impl<'a> Drop for NotifyFuture<'a> {
    fn drop(&mut self) {
        // The restored permit's waker leaves the state lock before it is woken;
        // `Waker::wake` may poll the task inline on this thread, and that poll
        // re-locks this state. Same discipline as `notify_one`.
        let waker = if let Some(id) = self.id {
            let mut state = match self.notify.state.lock() {
                Ok(state) => state,
                Err(_) => return,
            };
            // If we were holding a single-task permit but never observed
            // it, hand it to the next pending waiter (or store it) so it
            // is not lost. Broadcast (`All`) grants are not restored.
            if state.waiters.deregister(id) == Some(NotifyGrant::One) {
                let waker = state.waiters.grant_oldest(NotifyGrant::One);
                if waker.is_none() {
                    state.notified = true;
                }
                waker
            } else {
                None
            }
        } else {
            None
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}
