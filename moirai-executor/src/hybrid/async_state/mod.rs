//! Ownership, polling, and completion of one pinned async future.
//!
//! `AsyncFutureState` retains a future in a stable `Arc` allocation. Its atomic
//! phase serializes access to the future, lifecycle token, and result sender;
//! concurrent wakers never dereference those cells.
//!
//! # State machine
//!
//! ```text
//! CREATED -> QUEUED -> POLLING -> IDLE
//!               |         |       |
//!               |         +-> NOTIFIED -> POLLING or QUEUED
//!               |         |
//!               +---------+-> COMPLETED
//! ```
//!
//! The spawner alone transfers `CREATED` to `QUEUED`. Tracking can expose a
//! cancellation waker before first admission, but that waker cannot claim
//! `CREATED`. Cancellation publishes its registry flag before waking; the first
//! accepted poll observes it. Track rejection leaves the spawner the only owner
//! of `CREATED`; scheduler rejection retains its exclusive `QUEUED` epoch until
//! `complete_rejected` settles the task. Neither rejection restores `IDLE`.
//!
//! # Exclusive cell access
//!
//! A successful `QUEUED` to `POLLING` compare-exchange grants one poll owner
//! access to the cells. Successful enqueue transfers the queued epoch to that
//! job. Rejected enqueue leaves the admitting caller its completion ownership,
//! because no job can poll that epoch. Concurrent wakes during `POLLING` only
//! mark `NOTIFIED`; they gain no cell access. The mutable future borrow ends
//! before ownership transfers to an inline repoll or another queued job.
//!
//! Wakes of `IDLE` claim an enqueue obligation. A full queue triggers bounded
//! inline polling; nested saturation completes with `ResourceExhausted`.
//! Self-wakes consume a bounded inline-repoll budget before rescheduling.
//! Scheduler closure settles the owned epoch as cancelled. The phase remains
//! exclusive until the future has been destroyed, even if its destructor
//! panics; `future_present` prevents a second destruction.
//!
//! # Executor ownership and terminal publication
//!
//! The executor's live-waker table retains pending tasks, including futures
//! that save no waker. Each task holds only a weak table reference. Shutdown
//! closes registration and snapshots live tasks without removing their entries;
//! overlapping callers can settle the same live tasks. Completion removes each
//! entry. Cancellation needs no subsequent wake from an external event source.
//!
//! The poll or rejected-admission owner publishes lifecycle metadata and result
//! state before its registry notification callbacks. Result and registry callbacks
//! are both attempted; callback panics are counted and retained until guarded
//! output/payload cleanup finishes. Shutdown contains each wake's unwind before
//! continuing its snapshot and scheduler teardown. Final state destruction has
//! exclusive access after the last `Arc`; it skips an already-destroyed future.

mod admission;
mod cancellation;
mod completion;
mod future_state;
mod inline_poll;
mod lifecycle;
mod phase;
mod polling;
mod tasks;
#[cfg(test)]
mod tests;
mod waker;

pub(crate) use future_state::AsyncFutureState;
pub(crate) use tasks::AsyncTasks;
