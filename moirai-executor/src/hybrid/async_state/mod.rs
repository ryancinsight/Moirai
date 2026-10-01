//! Async future state machine.
//!
//! `AsyncFutureState` drives one `Future` to completion across the hybrid
//! scheduler's worker threads. It is shared as an `Arc` (it is its own `Waker`),
//! so a `Future` — which is `!Sync` to poll — is held in `UnsafeCell`s reachable
//! from every clone. A single `AtomicU8` `state` makes the concurrent access to
//! those cells sound.
//!
//! # State machine
//!
//! ```text
//! IDLE ──schedule──▶ QUEUED ──poll claims──▶ POLLING ──Pending, no wake──▶ IDLE
//!                       │                      │  │
//!     rejected wake ────┴──▶ COMPLETED        │  └── Ready / panic / cancel ─▶ COMPLETED
//!     shutdown on wake ─────────▶ COMPLETED   └── wake during poll ─▶ NOTIFIED
//!     spawn rejection ──────────▶ IDLE               (inline repoll or reschedule)
//! ```
//!
//! # Exclusivity invariant
//!
//! The `QUEUED → POLLING` compare-exchange in `AsyncFutureState::poll` has
//! exactly one winner; the loser returns without touching anything. That winner
//! is the **poll owner**. A second exclusive role exists only when that queue
//! admission is rejected: the caller that won `IDLE → QUEUED`, or transferred
//! `NOTIFIED → QUEUED`, remains the **rejected-queue completion owner** because
//! no scheduler job was admitted. While either role accesses `future`,
//! `lifecycle`, `result_sender`, or `future_present`, concurrent wakers only
//! load/CAS `state`; `QUEUED` and `POLLING` both prevent them from becoming an
//! accessor. Every `UnsafeCell` dereference is therefore single-threaded despite
//! the shared `Arc`. A concurrent `POLLING → NOTIFIED` transition may occur
//! while the poll owner's `&mut` future borrow is live because it transfers no
//! cell-access permission. That borrow is dropped before any transition that
//! does transfer permission to a successor poll or rejected-queue completion
//! owner, so neither can observe it.
//!
//! A wake arriving mid-poll CASes `POLLING → NOTIFIED` rather than enqueuing, so
//! it is never lost: the poll owner re-polls inline (bounded by
//! `ASYNC_INLINE_REPOLL_LIMIT`) or reschedules. If that bounded reschedule is
//! rejected by a full queue, the task completes with `ResourceExhausted`
//! instead of recursively re-polling itself on the waking thread. Cross-task
//! inline polls are independently bounded by `ASYNC_INLINE_POLL_DEPTH_LIMIT`;
//! a saturated nested wake completes with the same typed error instead of
//! growing the caller's stack. The future is dropped once, by the
//! `future_present` flag: either the poll owner or rejected-queue completion
//! owner drops it on completion, and `Drop` (reached only after the last `Arc`,
//! whose refcount release/acquire orders the owner's write before the
//! destructor's read) skips an already-dropped future.
//!
//! # Enqueue obligation (wakes survive admission rejection)
//!
//! The `IDLE → QUEUED` winner owns exactly one *enqueue obligation*. A successful
//! admission transfers it to the queued job, whose poll claims `POLLING`. A
//! rejected admission leaves it with the caller, which must either poll inline
//! or complete the task; dropping that obligation would strand `QUEUED` with no
//! job and make every later wake short-circuit as "already scheduled".
//! `schedule_wake` therefore polls inline after the first rejected admission
//! when the thread-local depth budget is available. A nested rejection past that
//! budget exits `QUEUED` through `complete_resource_exhausted` as typed task
//! exhaustion. Only scheduler
//! shutdown — after which no job of any kind can ever be admitted or run —
//! releases the obligation, by completing the task as cancelled
//! (`complete_cancelled`): the wake can never be honored, and a task left idle
//! would leave its waiters pending until the last waker clone drops.
//! The spawn-time `schedule` instead propagates admission failure to the
//! spawner (the spawn-backpressure contract) after reverting `QUEUED → IDLE`,
//! which is race-free there because wakers are minted only inside `poll`.

mod admission;
mod completion;
mod future_state;
mod inline_poll;
mod lifecycle;
mod phase;
mod polling;
#[cfg(test)]
mod tests;
mod waker;

pub(crate) use future_state::AsyncFutureState;
