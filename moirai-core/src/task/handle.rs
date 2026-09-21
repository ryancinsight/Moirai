use crate::error::TaskError;

use super::id_and_context::TaskId;

// ── std-only block ────────────────────────────────────────────────────────────

#[cfg(feature = "std")]
use core::mem::ManuallyDrop;

#[cfg(feature = "std")]
use std::sync::{Arc, atomic::AtomicU8};
#[cfg(feature = "std")]
use std::thread;

#[cfg(feature = "std")]
use moirai_utils::{CacheAligned, ResultCell};

// ── ResultWaitPolicy sealed module ────────────────────────────────────────────

#[cfg(feature = "std")]
pub(super) mod result_wait {
    pub(super) mod sealed {
        pub trait Sealed {}
    }

    /// Compile-time wait policy for task result handoff.
    ///
    /// Implementors are zero-sized marker types. `TaskResultSlot` receives the
    /// policy as a generic parameter, so the spin budget is const-folded and no
    /// runtime policy value is stored in the handle or slot.
    pub trait ResultWaitPolicy: sealed::Sealed {
        /// Maximum number of spin-loop iterations before parking the thread.
        const SPIN_ATTEMPTS: usize;
    }

    /// Zero-sized blocking wait policy: spins up to `MAX_SPIN_ATTEMPTS` then parks.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct BlockingResultWait;

    impl sealed::Sealed for BlockingResultWait {}

    impl ResultWaitPolicy for BlockingResultWait {
        const SPIN_ATTEMPTS: usize = super::super::MAX_SPIN_ATTEMPTS;
    }
}

#[cfg(feature = "std")]
pub use result_wait::{BlockingResultWait, ResultWaitPolicy};

// ── TaskResultSlot (private) ──────────────────────────────────────────────────

/// One-shot cell carrying the result of a single scheduled task.
///
/// # Cache-line layout
///
/// The `state` field is the synchronisation point between the *producer*
/// (the worker that executes the task and writes the result) and the
/// *consumer* (the thread that called `JoinHandle::wait`).  Keeping
/// `state` on its own sector prevents false sharing: the producer
/// invalidates only that sector when storing `RESULT_READY`, and the
/// consumer accesses `result`/`waiter` beyond it.
///
/// [`CacheAligned`] supplies both halves of that layout — it aligns the slot
/// (and therefore the cell's `state` word) to
/// `moirai_utils::DESTRUCTIVE_INTERFERENCE_SIZE`, and its own size pushes
/// `result`/`waiter` past that boundary. The separation is 128 bytes on
/// x86-64/aarch64, where the adjacent-line prefetcher makes 64 too narrow;
/// the per-target value lives in `moirai-utils`, not in a literal here.
#[cfg(feature = "std")]
struct TaskResultSlot<T> {
    cell: ResultCell<Result<T, TaskError>, thread::Thread, CacheAligned<AtomicU8>>,
}

/// The blocking side of the completion path.
///
/// The protocol — the state machine, its cell-access invariants and its ordering
/// argument — lives with [`ResultCell`] in `moirai-utils`, and the async handle
/// runs it too. This type adds only what blocking needs: a parked
/// [`thread::Thread`] as the waiter, the cache-aligned state word its layout note
/// above describes, and [`wait`](Self::wait)'s spin-then-park loop, which needs
/// `thread::park` and so does not belong in the shared cell.
#[cfg(feature = "std")]
impl<T> TaskResultSlot<T> {
    fn new() -> Self {
        Self {
            cell: ResultCell::new(),
        }
    }

    fn complete(&self, result: Result<T, TaskError>) {
        self.cell.complete(result);
    }

    fn wait<P>(&self) -> Result<T, TaskError>
    where
        P: ResultWaitPolicy,
    {
        if let Some(result) = self.try_take_ready() {
            return result;
        }

        for _ in 0..P::SPIN_ATTEMPTS {
            if let Some(result) = self.try_take_observed_ready() {
                return result;
            }
            core::hint::spin_loop();
        }

        self.register_waiter();

        loop {
            if let Some(result) = self.try_take_observed_ready() {
                return result;
            }

            thread::park();
        }
    }

    fn is_completed(&self) -> bool {
        self.cell.is_completed()
    }

    fn try_take_ready(&self) -> Option<Result<T, TaskError>> {
        self.cell.try_take_ready()
    }

    fn try_take_observed_ready(&self) -> Option<Result<T, TaskError>> {
        self.cell.try_take_observed_ready()
    }

    fn register_waiter(&self) {
        self.cell.register(&thread::current());
    }
}
// ── Diagnostic helpers (feature = "result-diagnostics") ──────────────────────

#[cfg(all(feature = "std", feature = "result-diagnostics"))]
const DIAGNOSTIC_READY_VALUE: usize = 42;

/// Diagnostic-only ready result-slot take path for benchmark attribution.
#[cfg(all(feature = "std", feature = "result-diagnostics"))]
#[doc(hidden)]
pub fn diagnostic_result_slot_ready_take() -> usize {
    let slot = TaskResultSlot::new();
    slot.complete(Ok(DIAGNOSTIC_READY_VALUE));
    match slot.try_take_ready() {
        Some(Ok(value)) => value,
        _ => 0,
    }
}

/// Diagnostic-only pending spin miss path for benchmark attribution.
#[cfg(all(feature = "std", feature = "result-diagnostics"))]
#[doc(hidden)]
pub fn diagnostic_result_slot_spin_miss() -> usize {
    let slot = TaskResultSlot::<usize>::new();
    let mut misses = 0usize;
    for _ in 0..BlockingResultWait::SPIN_ATTEMPTS {
        if slot.try_take_observed_ready().is_none() {
            misses = misses.wrapping_add(1);
        }
        core::hint::spin_loop();
    }
    misses
}

/// Diagnostic-only waiter registration path for benchmark attribution.
#[cfg(all(feature = "std", feature = "result-diagnostics"))]
#[doc(hidden)]
pub fn diagnostic_result_slot_register_waiter() -> usize {
    let slot = TaskResultSlot::<usize>::new();
    slot.register_waiter();
    usize::from(slot.state.load(Ordering::Acquire) == RESULT_WAITING)
}

/// Diagnostic-only waiting-result completion path for benchmark attribution.
#[cfg(all(feature = "std", feature = "result-diagnostics"))]
#[doc(hidden)]
pub fn diagnostic_result_slot_complete_waiting() -> usize {
    let slot = TaskResultSlot::new();
    slot.register_waiter();
    slot.complete(Ok(DIAGNOSTIC_READY_VALUE));
    match slot.try_take_ready() {
        Some(Ok(value)) => value,
        _ => 0,
    }
}

// ── TaskHandle (std) ──────────────────────────────────────────────────────────

/// A handle to a task that may be running on another thread.
#[cfg(feature = "std")]
#[allow(clippy::module_name_repetitions)]
pub struct TaskHandle<T> {
    id: TaskId,
    result_slot: Option<Arc<TaskResultSlot<T>>>,
}

#[cfg(feature = "std")]
impl<T> TaskHandle<T> {
    /// Creates a new pending task handle and its completion sender.
    #[must_use]
    pub fn new_pending(id: TaskId) -> (Self, TaskResultSender<T>) {
        let slot = Arc::new(TaskResultSlot::new());
        (
            Self {
                id,
                result_slot: Some(Arc::clone(&slot)),
            },
            TaskResultSender { slot: Some(slot) },
        )
    }

    /// Creates a new task handle from an existing result.
    #[must_use]
    pub fn ready(id: TaskId, result: Result<T, TaskError>) -> Self {
        let slot = Arc::new(TaskResultSlot::new());
        slot.complete(result);
        Self {
            id,
            result_slot: Some(slot),
        }
    }

    /// Creates a new detached task handle (no result channel).
    ///
    /// # Arguments
    /// * `id` - The unique identifier for this task
    ///
    /// # Returns
    /// A new detached task handle instance
    #[must_use]
    pub fn new_detached(id: TaskId) -> Self {
        Self {
            id,
            result_slot: None,
        }
    }

    /// Returns the task ID.
    ///
    /// # Returns
    /// The unique identifier for this task
    #[must_use]
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Waits for the task to complete and returns the result.
    ///
    /// # Returns
    /// - `Some(Ok(result))` if the task completed successfully
    /// - `Some(Err(error))` if the task failed with an error
    /// - `None` if the task was detached
    #[must_use]
    pub fn join(mut self) -> Option<Result<T, TaskError>> {
        self.result_slot
            .take()
            .map(|slot| slot.wait::<BlockingResultWait>())
    }

    /// Checks if the task has finished execution.
    ///
    /// # Returns
    /// `true` if the task has completed (successfully or with error), `false` if still running
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.result_slot
            .as_ref()
            .is_some_and(|slot| slot.is_completed())
    }
}

// ── TaskResultSender (std) ────────────────────────────────────────────────────

/// Single-producer completion endpoint for a task result.
#[cfg(feature = "std")]
#[allow(clippy::module_name_repetitions)]
pub struct TaskResultSender<T> {
    slot: Option<Arc<TaskResultSlot<T>>>,
}

#[cfg(feature = "std")]
impl<T> TaskResultSender<T> {
    /// Complete the task result and wake any waiter.
    pub fn send(self, result: Result<T, TaskError>) {
        let mut sender = ManuallyDrop::new(self);
        if let Some(slot) = sender.slot.take() {
            slot.complete(result);
        }
    }
}

#[cfg(feature = "std")]
impl<T> Drop for TaskResultSender<T> {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            slot.complete(Err(TaskError::Cancelled));
        }
    }
}

// ── TaskHandle (no_std) ───────────────────────────────────────────────────────

// For no_std environments, provide a simpler handle
#[cfg(not(feature = "std"))]
pub struct TaskHandle<T> {
    id: TaskId,
    _phantom: core::marker::PhantomData<T>,
}

#[cfg(not(feature = "std"))]
impl<T> TaskHandle<T> {
    /// Create a new task handle.
    pub fn new(id: TaskId) -> Self {
        Self {
            id,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Create a new detached task handle (alias for new in no_std).
    pub fn new_detached(id: TaskId) -> Self {
        Self::new(id)
    }

    /// Get the task ID.
    pub fn id(&self) -> TaskId {
        self.id
    }
}
