//! Ordering between a handle's stream operations and its `&self` observers.
//!
//! Stream operations run as pool jobs on any worker, and a write may still be
//! queued after its caller moved on (`poll_write` returns once the write is
//! queued). An observer such as `read_at` or `metadata` is a separate job
//! that could run first and miss those bytes. Each stream operation therefore
//! takes a [`Ticket`] when it is queued, and the ticket completes when the job
//! finishes, is skipped, or panics. An observer waits until every ticket
//! issued before it has completed.
//!
//! Lock order: the fence's mutex is a leaf. It is held only to update the
//! counters and the waker list, never across a syscall, the file's cursor
//! lock, or a wake. Wakers run after the mutex is released. The ticket
//! completes after the job has returned from the handle, so the cursor lock
//! is already released. No cycle with the cursor lock is possible.

use std::future::poll_fn;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Poll, Waker};

#[derive(Default)]
struct Counters {
    issued: u64,
    completed: u64,
    /// Observers waiting for `completed` to reach their target.
    waiters: Vec<(u64, Waker)>,
}

/// Counts the stream operations a handle has queued and finished.
#[derive(Default)]
pub(super) struct Fence {
    counters: Mutex<Counters>,
}

/// One queued stream operation; completes the fence entry on drop.
pub(super) struct Ticket(Arc<Fence>);

impl Fence {
    /// Record a stream operation about to be queued.
    pub(super) fn issue(self: &Arc<Self>) -> Ticket {
        self.counters
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .issued += 1;
        Ticket(Arc::clone(self))
    }

    /// Wait until every stream operation issued before this call finished.
    pub(super) async fn settled(&self) {
        let target = self
            .counters
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .issued;
        poll_fn(|cx| {
            let mut counters = self.counters.lock().unwrap_or_else(PoisonError::into_inner);
            if counters.completed >= target {
                return Poll::Ready(());
            }
            if !counters
                .waiters
                .iter()
                .any(|(waiting_for, waker)| *waiting_for == target && waker.will_wake(cx.waker()))
            {
                counters.waiters.push((target, cx.waker().clone()));
            }
            Poll::Pending
        })
        .await;
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        // Tickets complete in any order across workers; `completed` counts
        // them, and a target is met once as many have completed as were
        // issued before it. Jobs on one handle are sequential (one stream
        // operation in flight), so the count reaches a target only when every
        // earlier ticket has completed.
        let due: Vec<Waker> = {
            let mut counters = self
                .0
                .counters
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            counters.completed += 1;
            let completed = counters.completed;
            let (due, waiting): (Vec<_>, Vec<_>) = counters
                .waiters
                .drain(..)
                .partition(|(target, _)| *target <= completed);
            counters.waiters = waiting;
            due.into_iter().map(|(_, waker)| waker).collect()
        };
        for waker in due {
            // One observer's panicking waker must not strand the others.
            let _contained = catch_unwind(AssertUnwindSafe(|| waker.wake()));
        }
    }
}
