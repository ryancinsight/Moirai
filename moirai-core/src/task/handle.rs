use crate::error::TaskError;

use super::id_and_context::TaskId;
use super::traits::Task;

// ── std-only block ────────────────────────────────────────────────────────────

#[cfg(feature = "std")]
use core::cell::UnsafeCell;
#[cfg(feature = "std")]
use core::mem::{ManuallyDrop, MaybeUninit};

#[cfg(feature = "std")]
use std::sync::{
    atomic::{AtomicU8, Ordering},
    mpsc, Arc,
};
#[cfg(feature = "std")]
use std::thread;

#[cfg(feature = "std")]
use super::id_and_context::TaskContext;

// State constants for TaskResultSlot
#[cfg(feature = "std")]
const RESULT_PENDING: u8 = 0;
#[cfg(feature = "std")]
const RESULT_WRITING: u8 = 1;
#[cfg(feature = "std")]
const RESULT_READY: u8 = 2;
#[cfg(feature = "std")]
const RESULT_TAKEN: u8 = 3;
#[cfg(feature = "std")]
const RESULT_WAITING: u8 = 4;

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
        const SPIN_ATTEMPTS: usize = crate::constants::MAX_SPIN_ATTEMPTS;
    }
}

#[cfg(feature = "std")]
pub use result_wait::{BlockingResultWait, ResultWaitPolicy};

// ── TaskWrapper ───────────────────────────────────────────────────────────────

/// A wrapper that adds result channel support to any task.
#[cfg(feature = "std")]
pub struct TaskWrapper<T: Task> {
    task: T,
    result_sender: Option<mpsc::Sender<Result<T::Output, TaskError>>>,
    completion_sender: Option<mpsc::Sender<()>>,
}

#[cfg(feature = "std")]
impl<T: Task> TaskWrapper<T> {
    /// Create a new task wrapper.
    pub fn new(task: T) -> Self {
        Self {
            task,
            result_sender: None,
            completion_sender: None,
        }
    }

    /// Create a new task wrapper with result and completion senders.
    pub fn with_result_sender(
        task: T,
        result_sender: mpsc::Sender<Result<T::Output, TaskError>>,
        completion_sender: mpsc::Sender<()>,
    ) -> Self {
        Self {
            task,
            result_sender: Some(result_sender),
            completion_sender: Some(completion_sender),
        }
    }
}

#[cfg(feature = "std")]
impl<T: Task> Task for TaskWrapper<T>
where
    T::Output: Clone,
{
    type Output = T::Output;

    fn execute(self) -> Self::Output {
        let result = self.task.execute();

        // Send result through channel if available
        if let Some(sender) = self.result_sender {
            let _ = sender.send(Ok(result.clone()));
        }

        // Signal completion
        if let Some(sender) = self.completion_sender {
            let _ = sender.send(());
        }

        result
    }

    fn context(&self) -> &TaskContext {
        self.task.context()
    }
}

// ── TaskResultSlot (private) ──────────────────────────────────────────────────

/// One-shot cell carrying the result of a single scheduled task.
///
/// # Cache-line layout
///
/// The `state` field is the synchronisation point between the *producer*
/// (the worker that executes the task and writes the result) and the
/// *consumer* (the thread that called `JoinHandle::wait`).  Keeping
/// `state` on its own 64-byte cache line prevents false sharing:
/// the producer invalidates only its line when storing `RESULT_READY`,
/// and the consumer accesses `result`/`waiter` on a separate line.
///
/// `#[repr(align(64))]` on `TaskResultSlot` itself aligns the start of
/// the struct to a cache line; `_pad` pushes `result` and `waiter` past
/// the first 64 bytes so they land on a second line.
#[cfg(feature = "std")]
#[repr(align(64))]
struct TaskResultSlot<T> {
    /// Synchronisation state — written by the producer, read by consumer.
    /// Placed first so it occupies the beginning of the first cache line.
    state: AtomicU8,
    /// Padding to push `result` and `waiter` onto a separate cache line,
    /// eliminating producer-consumer false sharing on the `state` field.
    _pad: [u8; 63],
    result: UnsafeCell<MaybeUninit<Result<T, TaskError>>>,
    waiter: UnsafeCell<MaybeUninit<thread::Thread>>,
}

// Safety: the slot is a single-producer/single-consumer one-shot cell.
// `complete` wins the PENDING/WAITING -> WRITING transition before writing.
// WAITING is entered only after the waiter thread is stored in the waiter cell,
// and `wait` takes the value only after an acquire READY -> TAKEN transition.
#[cfg(feature = "std")]
unsafe impl<T: Send> Send for TaskResultSlot<T> {}

#[cfg(feature = "std")]
unsafe impl<T: Send> Sync for TaskResultSlot<T> {}

#[cfg(feature = "std")]
impl<T> TaskResultSlot<T> {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(RESULT_PENDING),
            _pad: [0u8; 63],
            result: UnsafeCell::new(MaybeUninit::uninit()),
            waiter: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    fn complete(&self, result: Result<T, TaskError>) {
        let Some(waiting) = self.begin_completion() else {
            return;
        };

        // Safety: the WRITING state is reachable only through
        // `begin_completion`, so no other thread can read, write, or drop the
        // result cell until READY publishes.
        unsafe {
            (*self.result.get()).write(result);
        }

        self.state.store(RESULT_READY, Ordering::Release);

        if waiting {
            // Safety: WAITING is reachable only after `register_waiter` writes
            // the thread handle and publishes it with a release CAS.
            let thread = unsafe { (*self.waiter.get()).assume_init_read() };
            thread.unpark();
        }
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
        self.state.load(Ordering::Acquire) == RESULT_READY
    }

    fn try_take_ready(&self) -> Option<Result<T, TaskError>> {
        if self
            .state
            .compare_exchange(
                RESULT_READY,
                RESULT_TAKEN,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            // Safety: READY is published only after `complete` initializes the
            // cell. The READY -> TAKEN transition is unique, so this read moves
            // the result exactly once.
            Some(unsafe { (*self.result.get()).assume_init_read() })
        } else {
            None
        }
    }

    fn try_take_observed_ready(&self) -> Option<Result<T, TaskError>> {
        if self.state.load(Ordering::Relaxed) == RESULT_READY {
            self.try_take_ready()
        } else {
            None
        }
    }

    fn register_waiter(&self) {
        loop {
            match self.state.load(Ordering::Acquire) {
                RESULT_PENDING => {
                    // Safety: there is only one consumer. If the publish CAS
                    // fails, this local thread handle is dropped before retry.
                    unsafe {
                        (*self.waiter.get()).write(thread::current());
                    }

                    if self
                        .state
                        .compare_exchange(
                            RESULT_PENDING,
                            RESULT_WAITING,
                            Ordering::Release,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return;
                    }

                    // Safety: the CAS failed, so no producer can observe this
                    // waiter cell as initialized through the WAITING state.
                    unsafe {
                        (*self.waiter.get()).assume_init_drop();
                    }
                }
                RESULT_WRITING => core::hint::spin_loop(),
                _ => return,
            }
        }
    }

    fn begin_completion(&self) -> Option<bool> {
        match self.state.compare_exchange(
            RESULT_PENDING,
            RESULT_WRITING,
            Ordering::Relaxed,
            Ordering::Acquire,
        ) {
            Ok(_) => Some(false),
            Err(RESULT_WAITING) => {
                if self
                    .state
                    .compare_exchange(
                        RESULT_WAITING,
                        RESULT_WRITING,
                        Ordering::Acquire,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    Some(true)
                } else {
                    None
                }
            }
            Err(_) => None,
        }
    }
}

#[cfg(feature = "std")]
impl<T> Drop for TaskResultSlot<T> {
    fn drop(&mut self) {
        let state = *self.state.get_mut();
        if state == RESULT_READY {
            // Safety: READY means the cell is initialized and no consuming join
            // took it because `drop` has exclusive access to the slot.
            unsafe {
                self.result.get_mut().assume_init_drop();
            }
        } else if state == RESULT_WAITING {
            // Safety: WAITING means the waiter thread handle is initialized and
            // no producer unparked it because `drop` has exclusive access.
            unsafe {
                self.waiter.get_mut().assume_init_drop();
            }
        }
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
