//! Single-producer/single-consumer result slot backing one `AsyncHandle`.
//!
//! `AsyncResultSlot` hands a task's output from the executor (the **producer**,
//! which calls `complete` exactly once) to the awaiting `AsyncHandle` future (the
//! **consumer**, which calls `try_take_ready` / `register_waker` from `poll`). The
//! slot lives in an `Arc` shared by exactly those two owners, so the `result` and
//! `waiter` cells need no lock:
//!
//! - the producer runs once — the wrapped future's tail, `let r = fut.await;
//!   slot.complete(r);`;
//! - the consumer is serialized with itself, because `poll` takes `Pin<&mut Self>`,
//!   so no two `try_take_ready` / `register_waker` calls overlap;
//! - `Drop` runs only after the last `Arc`, so it has exclusive access and races
//!   neither side.
//!
//! # State machine
//!
//! `state: AtomicU8` is the sole synchronization variable; the `result` and
//! `waiter` cells are touched only while a transition grants exclusive access to
//! them (C = consumer, P = producer):
//!
//! ```text
//!   PENDING ──C register──▶ WAITING ──C re-poll──▶ UPDATING_WAKER ──C──▶ WAITING
//!      │                       │
//!      │ P complete            │ P complete
//!      ▼                       ▼
//!   WRITING ────────────────▶ WRITING ──P──▶ READY ──C take──▶ TAKEN
//! ```
//!
//! `WRITING` is the producer's exclusive claim on `result`, and `READY` publishes
//! it; `UPDATING_WAKER` is the consumer's exclusive claim on `waiter` while it
//! swaps a stale waker, past which the producer spins.
//!
//! # Cell-access invariants (what the per-site `// Safety:` comments rely on)
//!
//! 1. **`result`: written once, read once.** Only the producer writes it, only
//!    under `WRITING` (entered by winning the producer CAS, so no consumer can see
//!    it yet). It is read exactly once — by the unique `READY -> TAKEN` consumer
//!    CAS, or by `Drop` at `READY` when the consumer never took it.
//! 2. **`waiter`: written by the consumer, read once by the producer.** The
//!    consumer writes it under `PENDING` (published by the `PENDING -> WAITING`
//!    release CAS) or under the `UPDATING_WAKER` lock. The producer reads it
//!    exactly once, on `WAITING -> WRITING`; otherwise `Drop` at `WAITING` drops
//!    it. A *failed* publish CAS proves no producer observed the write, so the
//!    consumer drops its own clone.
//! 3. **No lost wakeup — a contract shared with `AsyncHandle::poll`.** `complete`
//!    on the `PENDING -> WRITING` path (it beat registration) deliberately does
//!    not wake: no waker is registered yet. Liveness therefore requires the
//!    consumer to check → `register_waker` → **re-check**; should `complete` land
//!    in that window, the re-check observes `READY`. That re-check is load-bearing,
//!    not defensive — dropping it reintroduces a hang.
//!
//! # Ordering
//!
//! Each cell access is ordered by a release/acquire pair on `state`: the writer
//! releases on the publishing transition, the reader acquires on the transition
//! that reads the cell. Thus `PENDING -> WAITING` and the `UPDATING_WAKER ->
//! WAITING` store are `Release` (they publish `waiter`), while `WAITING -> WRITING`
//! and `READY -> TAKEN` are `Acquire` (they read a cell). The `PENDING -> WRITING`
//! success is `Relaxed`: that path reads neither cell before its own `store(READY,
//! Release)` publishes `result`, so it carries no incoming edge to establish.

use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicU8, Ordering},
    task::Waker,
};

const ASYNC_RESULT_PENDING: u8 = 0;
const ASYNC_RESULT_WAITING: u8 = 1;
const ASYNC_RESULT_UPDATING_WAKER: u8 = 2;
const ASYNC_RESULT_WRITING: u8 = 3;
const ASYNC_RESULT_READY: u8 = 4;
const ASYNC_RESULT_TAKEN: u8 = 5;

/// Single-producer result slot for one async handle.
pub(super) struct AsyncResultSlot<T> {
    result: UnsafeCell<MaybeUninit<T>>,
    state: AtomicU8,
    waiter: UnsafeCell<MaybeUninit<Waker>>,
}

// Safety: the slot has one producer and one consumer. Atomic states serialize
// result publication, result consumption, and inline waker updates.
unsafe impl<T: Send> Send for AsyncResultSlot<T> {}

// Safety: shared access is mediated by the state machine; result and waker cells
// are touched only after the corresponding atomic transition succeeds.
unsafe impl<T: Send> Sync for AsyncResultSlot<T> {}

impl<T> AsyncResultSlot<T> {
    pub(super) fn new() -> Self {
        Self {
            result: UnsafeCell::new(MaybeUninit::uninit()),
            state: AtomicU8::new(ASYNC_RESULT_PENDING),
            waiter: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    pub(super) fn complete(&self, result: T) {
        let Some(waiting) = self.begin_completion() else {
            return;
        };

        // Safety: WRITING is reachable only after `begin_completion` wins the
        // producer transition, so no consumer can read this cell yet.
        unsafe {
            (*self.result.get()).write(result);
        }

        self.state.store(ASYNC_RESULT_READY, Ordering::Release);

        if waiting {
            // Safety: WAITING is reachable only after `register_waker` writes
            // the waker and publishes it with a release transition.
            let waker = unsafe { (*self.waiter.get()).assume_init_read() };
            waker.wake();
        }
    }

    pub(super) fn try_take_ready(&self) -> Option<T> {
        if self
            .state
            .compare_exchange(
                ASYNC_RESULT_READY,
                ASYNC_RESULT_TAKEN,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            // Safety: READY is published only after the producer initializes
            // the result cell; READY -> TAKEN is a unique consumer transition.
            Some(unsafe { (*self.result.get()).assume_init_read() })
        } else {
            None
        }
    }

    pub(super) fn register_waker(&self, waker: &Waker) {
        loop {
            match self.state.load(Ordering::Acquire) {
                ASYNC_RESULT_PENDING => {
                    // Safety: there is one consumer. If the publish CAS fails,
                    // this local clone is dropped before retry.
                    unsafe {
                        (*self.waiter.get()).write(waker.clone());
                    }

                    if self
                        .state
                        .compare_exchange(
                            ASYNC_RESULT_PENDING,
                            ASYNC_RESULT_WAITING,
                            Ordering::Release,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return;
                    }

                    // Safety: the CAS failed, so no producer can observe this
                    // waiter cell as initialized by the WAITING state.
                    unsafe {
                        (*self.waiter.get()).assume_init_drop();
                    }
                }
                ASYNC_RESULT_WAITING => {
                    if self
                        .state
                        .compare_exchange(
                            ASYNC_RESULT_WAITING,
                            ASYNC_RESULT_UPDATING_WAKER,
                            Ordering::Acquire,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        // Safety: UPDATING_WAKER excludes the producer from
                        // reading the waiter cell while the consumer replaces it.
                        unsafe {
                            (*self.waiter.get()).assume_init_drop();
                            (*self.waiter.get()).write(waker.clone());
                        }
                        self.state.store(ASYNC_RESULT_WAITING, Ordering::Release);
                        return;
                    }
                }
                ASYNC_RESULT_UPDATING_WAKER | ASYNC_RESULT_WRITING => core::hint::spin_loop(),
                _ => return,
            }
        }
    }

    fn begin_completion(&self) -> Option<bool> {
        loop {
            match self.state.load(Ordering::Acquire) {
                ASYNC_RESULT_PENDING => {
                    if self
                        .state
                        .compare_exchange(
                            ASYNC_RESULT_PENDING,
                            ASYNC_RESULT_WRITING,
                            Ordering::Relaxed,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return Some(false);
                    }
                }
                ASYNC_RESULT_WAITING => {
                    if self
                        .state
                        .compare_exchange(
                            ASYNC_RESULT_WAITING,
                            ASYNC_RESULT_WRITING,
                            Ordering::Acquire,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return Some(true);
                    }
                }
                ASYNC_RESULT_UPDATING_WAKER => core::hint::spin_loop(),
                _ => return None,
            }
        }
    }
}

impl<T> Drop for AsyncResultSlot<T> {
    fn drop(&mut self) {
        match self.state.load(Ordering::Acquire) {
            ASYNC_RESULT_READY => {
                // Safety: READY means the result cell is initialized and no
                // consumer took it because drop has exclusive access.
                unsafe {
                    self.result.get_mut().assume_init_drop();
                }
            }
            ASYNC_RESULT_WAITING => {
                // Safety: WAITING means the waiter cell is initialized and no
                // producer can access it because drop has exclusive access.
                unsafe {
                    self.waiter.get_mut().assume_init_drop();
                }
            }
            _ => {}
        }
    }
}
