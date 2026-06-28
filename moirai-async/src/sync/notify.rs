//! Notification primitive for efficient async task coordination
//!
//! Provides notification mechanisms for waking up waiting async tasks,
//! following SLAP principle with focused responsibility.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WaiterState {
    Pending,
    NotifiedOne,
    NotifiedAll,
}

struct Waiter {
    waker: Waker,
    state: WaiterState,
}

/// Notification primitive for efficient task coordination
pub struct Notify {
    state: Arc<Mutex<NotifyState>>,
}

struct NotifyState {
    notified: bool,
    /// Registered waiters keyed by a monotonic id. Keyed (rather than a linear
    /// `VecDeque`) so per-waiter `poll`/drop lookups and removals are O(log n)
    /// instead of O(n) — the lock is held for less time per operation, which
    /// matters when many tasks wait on one `Notify`. Because ids increase
    /// monotonically, in-order iteration is still FIFO, preserving
    /// `notify_one` fairness (oldest pending waiter first).
    waiters: BTreeMap<u64, Waiter>,
    next_id: u64,
}

impl NotifyState {
    /// Mark the oldest pending waiter as `NotifiedOne` and return its waker to
    /// wake (outside the lock if the caller prefers). Returns `None` if no
    /// waiter is pending.
    fn notify_oldest_pending(&mut self) -> Option<Waker> {
        let waiter = self
            .waiters
            .values_mut()
            .find(|w| w.state == WaiterState::Pending)?;
        waiter.state = WaiterState::NotifiedOne;
        Some(waiter.waker.clone())
    }
}

impl Notify {
    /// Create a new notification primitive
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(NotifyState {
                notified: false,
                waiters: BTreeMap::new(),
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
        match state.notify_oldest_pending() {
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
        let mut wakers = Vec::new();
        for waiter in state.waiters.values_mut() {
            if waiter.state == WaiterState::Pending {
                waiter.state = WaiterState::NotifiedAll;
                wakers.push(waiter.waker.clone());
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

impl<'a> NotifyFuture<'a> {
    /// Register a fresh pending waiter and record its id on the future.
    fn register(&mut self, state: &mut NotifyState, waker: Waker) {
        let id = state.next_id;
        state.next_id += 1;
        state.waiters.insert(
            id,
            Waiter {
                waker,
                state: WaiterState::Pending,
            },
        );
        self.id = Some(id);
    }
}

impl<'a> Future for NotifyFuture<'a> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.notify.state.lock().unwrap();

        // 1. Check if a permit is available
        if state.notified {
            state.notified = false;
            if let Some(id) = self.id.take() {
                state.waiters.remove(&id);
            }
            return Poll::Ready(());
        }

        // 2. Check if already registered
        if let Some(id) = self.id {
            match state.waiters.get(&id).map(|w| w.state) {
                Some(WaiterState::NotifiedOne | WaiterState::NotifiedAll) => {
                    // Woken! Remove ourselves and return Ready.
                    state.waiters.remove(&id);
                    self.id = None;
                    Poll::Ready(())
                }
                Some(WaiterState::Pending) => {
                    // Refresh the stored waker in case it changed.
                    state.waiters.get_mut(&id).unwrap().waker = cx.waker().clone();
                    Poll::Pending
                }
                None => {
                    // Our registration was lost; re-register.
                    let waker = cx.waker().clone();
                    self.register(&mut state, waker);
                    Poll::Pending
                }
            }
        } else {
            // First time polling, register a new waiter.
            let waker = cx.waker().clone();
            self.register(&mut state, waker);
            Poll::Pending
        }
    }
}

impl<'a> Drop for NotifyFuture<'a> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            if let Ok(mut state) = self.notify.state.lock() {
                let removed = state.waiters.remove(&id);

                // If we were holding a single-task permit but never observed it,
                // hand it to the next pending waiter (or store it) so it is not
                // lost.
                if removed.is_some_and(|w| w.state == WaiterState::NotifiedOne) {
                    match state.notify_oldest_pending() {
                        Some(waker) => waker.wake(),
                        None => state.notified = true,
                    }
                }
            }
        }
    }
}
