//! Single-producer/single-consumer result slot backing one `AsyncHandle`.
//!
//! The async half of the completion path: the executor (the **producer**, which
//! calls `complete` exactly once) hands a task's output to the awaiting
//! `AsyncHandle` future (the **consumer**, which calls `try_take_ready` /
//! `register_waker` from `poll`).
//!
//! This is [`ResultCell`] with `Waker` as the waiter payload, so the state
//! machine, the cell-access invariants and the ordering argument live in
//! `moirai-utils` once and the blocking `TaskResultSlot` in `moirai-core` runs
//! the same code — see [`ResultCell`] for the protocol, including why the
//! consumer's re-check after registering is load-bearing rather than defensive.
//!
//! `Waker` registers with `REPLACE_ON_REPEAT`, which is what this side needs: a
//! re-poll may carry a different waker and the newest must win.

use moirai_utils::ResultCell;
use std::task::Waker;

/// Single-producer result slot for one async handle.
pub(super) struct AsyncResultSlot<T> {
    cell: ResultCell<T, Waker>,
}

impl<T> AsyncResultSlot<T> {
    pub(super) fn new() -> Self {
        Self {
            cell: ResultCell::new(),
        }
    }

    pub(super) fn complete(&self, result: T) {
        self.cell.complete(result);
    }

    pub(super) fn try_take_ready(&self) -> Option<T> {
        self.cell.try_take_ready()
    }

    pub(super) fn register_waker(&self, waker: &Waker) {
        self.cell.register(waker);
    }
}
