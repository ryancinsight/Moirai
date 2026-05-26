//! Type-erased scheduler task storage.
//!
//! `ScheduledTask` is the scheduler-facing heterogeneous task container. It
//! keeps fitting task values inline and erases only through monomorphized
//! function pointers, so queue storage does not require a task trait object.

use crate::platform::{align_of, ptr, size_of, Box, MaybeUninit, UnsafeCell};
use crate::task::{Task, TaskContext};

/// Inline scheduler task payload budget.
///
/// Fourteen machine words plus the container metadata keeps the payload within
/// the same two-cache-line budget used by executor scheduled jobs on 64-bit
/// targets.
pub const INLINE_SCHEDULED_TASK_WORDS: usize = 14;

#[repr(C, align(64))]
struct ScheduledTaskStorage {
    words: [MaybeUninit<usize>; INLINE_SCHEDULED_TASK_WORDS],
}

impl ScheduledTaskStorage {
    fn uninit() -> Self {
        Self {
            words: [MaybeUninit::uninit(); INLINE_SCHEDULED_TASK_WORDS],
        }
    }
}

/// Scheduler-owned task value with inline storage and monomorphized erasure.
pub struct ScheduledTask {
    storage: UnsafeCell<ScheduledTaskStorage>,
    execute: unsafe fn(*mut ScheduledTaskStorage),
    drop_task: unsafe fn(*mut ScheduledTaskStorage),
    context: unsafe fn(*const ScheduledTaskStorage) -> *const TaskContext,
    present: bool,
}

// Safety: a `ScheduledTask` is created only from `T: Task`, and `Task` requires
// `Send + 'static`. The task is moved between queues and executed once.
unsafe impl Send for ScheduledTask {}

// Safety: shared access exposes only `context`, which reads immutable task
// metadata while the task remains present. Execution consumes `self`.
unsafe impl Sync for ScheduledTask {}

impl ScheduledTask {
    /// Creates a scheduler task from a concrete task type.
    pub fn new<T>(task: T) -> Self
    where
        T: Task,
    {
        if scheduled_task_fits::<T>() {
            Self::new_inline(task)
        } else {
            Self::new_boxed(task)
        }
    }

    fn new_inline<T>(task: T) -> Self
    where
        T: Task,
    {
        let erased = Self {
            storage: UnsafeCell::new(ScheduledTaskStorage::uninit()),
            execute: execute_inline_task::<T>,
            drop_task: drop_inline_task::<T>,
            context: context_inline_task::<T>,
            present: true,
        };

        // Safety: `scheduled_task_fits::<T>()` was checked by `new`, and
        // `ScheduledTaskStorage` has enough size and alignment for `T`.
        unsafe {
            erased.storage.get().cast::<T>().write(task);
        }

        erased
    }

    fn new_boxed<T>(task: T) -> Self
    where
        T: Task,
    {
        let erased = Self {
            storage: UnsafeCell::new(ScheduledTaskStorage::uninit()),
            execute: execute_boxed_task::<T>,
            drop_task: drop_boxed_task::<T>,
            context: context_boxed_task::<T>,
            present: true,
        };

        // Safety: a `Box<T>` is pointer-sized and fits the inline storage.
        unsafe {
            erased.storage.get().cast::<Box<T>>().write(Box::new(task));
        }

        erased
    }

    /// Returns the task context while the task is queued.
    #[must_use]
    pub fn context(&self) -> &TaskContext {
        assert!(
            self.present,
            "scheduled task context requested after execute"
        );
        // Safety: `present` means the storage contains the variant selected by
        // `context`, and the returned reference is tied to `&self`.
        unsafe { &*(self.context)(self.storage.get().cast_const()) }
    }

    /// Executes the task exactly once and drops its output.
    pub fn execute(mut self) {
        if self.present {
            self.present = false;
            // Safety: `present` was true, so storage contains a valid task for
            // the selected monomorphized execute function. The function consumes
            // the stored value.
            unsafe {
                (self.execute)(self.storage.get());
            }
        }
    }

    /// Returns whether `T` is stored inline by `ScheduledTask::new`.
    #[must_use]
    pub const fn stores_inline<T>() -> bool {
        scheduled_task_fits::<T>()
    }
}

impl Drop for ScheduledTask {
    fn drop(&mut self) {
        if self.present {
            self.present = false;
            // Safety: `present` means storage contains a valid task value for
            // the selected monomorphized drop function.
            unsafe {
                (self.drop_task)(self.storage.get());
            }
        }
    }
}

const fn scheduled_task_fits<T>() -> bool {
    size_of::<T>() <= size_of::<ScheduledTaskStorage>()
        && align_of::<T>() <= align_of::<ScheduledTaskStorage>()
}

unsafe fn execute_inline_task<T>(storage: *mut ScheduledTaskStorage)
where
    T: Task,
{
    let task = storage.cast::<T>().read();
    drop(task.execute());
}

unsafe fn drop_inline_task<T>(storage: *mut ScheduledTaskStorage)
where
    T: Task,
{
    ptr::drop_in_place(storage.cast::<T>());
}

unsafe fn context_inline_task<T>(storage: *const ScheduledTaskStorage) -> *const TaskContext
where
    T: Task,
{
    let task = &*storage.cast::<T>();
    task.context()
}

unsafe fn execute_boxed_task<T>(storage: *mut ScheduledTaskStorage)
where
    T: Task,
{
    let task = storage.cast::<Box<T>>().read();
    drop(task.execute());
}

unsafe fn drop_boxed_task<T>(storage: *mut ScheduledTaskStorage)
where
    T: Task,
{
    ptr::drop_in_place(storage.cast::<Box<T>>());
}

unsafe fn context_boxed_task<T>(storage: *const ScheduledTaskStorage) -> *const TaskContext
where
    T: Task,
{
    let task = &*storage.cast::<Box<T>>();
    task.context()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::{Priority, TaskId};
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingTask {
        context: TaskContext,
        counter: Arc<AtomicUsize>,
        value: usize,
    }

    impl CountingTask {
        fn new(id: u64, counter: Arc<AtomicUsize>, value: usize) -> Self {
            Self {
                context: TaskContext::new(TaskId::new(id)).with_priority(Priority::High),
                counter,
                value,
            }
        }
    }

    impl Task for CountingTask {
        type Output = usize;

        fn execute(self) -> Self::Output {
            self.counter.fetch_add(self.value, Ordering::Relaxed);
            self.value
        }

        fn context(&self) -> &TaskContext {
            &self.context
        }
    }

    struct OversizedTask {
        context: TaskContext,
        counter: Arc<AtomicUsize>,
        payload: [usize; INLINE_SCHEDULED_TASK_WORDS + 2],
    }

    impl OversizedTask {
        fn new(counter: Arc<AtomicUsize>) -> Self {
            Self {
                context: TaskContext::new(TaskId::new(99)),
                counter,
                payload: [1; INLINE_SCHEDULED_TASK_WORDS + 2],
            }
        }
    }

    impl Task for OversizedTask {
        type Output = usize;

        fn execute(self) -> Self::Output {
            let sum = self.payload.iter().sum::<usize>();
            self.counter.fetch_add(sum, Ordering::Relaxed);
            sum
        }

        fn context(&self) -> &TaskContext {
            &self.context
        }
    }

    struct DropTask {
        context: TaskContext,
        drops: Arc<AtomicUsize>,
    }

    impl Task for DropTask {
        type Output = ();

        fn execute(self) -> Self::Output {}

        fn context(&self) -> &TaskContext {
            &self.context
        }
    }

    impl Drop for DropTask {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn scheduled_task_storage_budget_is_static_and_bounded() {
        assert!(ScheduledTask::stores_inline::<CountingTask>());
        assert!(!ScheduledTask::stores_inline::<OversizedTask>());
        assert_eq!(align_of::<ScheduledTaskStorage>(), 64);
        assert!(size_of::<ScheduledTaskStorage>() <= 128);
        assert!(
            size_of::<ScheduledTaskStorage>() >= INLINE_SCHEDULED_TASK_WORDS * size_of::<usize>()
        );
    }

    #[test]
    fn scheduled_task_executes_inline_and_oversized_tasks() {
        let counter = Arc::new(AtomicUsize::new(0));
        ScheduledTask::new(CountingTask::new(7, Arc::clone(&counter), 5)).execute();
        ScheduledTask::new(OversizedTask::new(Arc::clone(&counter))).execute();

        assert_eq!(
            counter.load(Ordering::Relaxed),
            5 + INLINE_SCHEDULED_TASK_WORDS + 2
        );
    }

    #[test]
    fn scheduled_task_exposes_context_before_execution() {
        let counter = Arc::new(AtomicUsize::new(0));
        let task = ScheduledTask::new(CountingTask::new(11, counter, 1));

        assert_eq!(task.context().id, TaskId::new(11));
        assert_eq!(task.context().priority, Priority::High);
    }

    #[test]
    fn scheduled_task_drops_unexecuted_inline_and_oversized_tasks() {
        let inline_drops = Arc::new(AtomicUsize::new(0));
        let boxed_drops = Arc::new(AtomicUsize::new(0));

        drop(ScheduledTask::new(DropTask {
            context: TaskContext::new(TaskId::new(1)),
            drops: Arc::clone(&inline_drops),
        }));
        drop(ScheduledTask::new(OversizedTask {
            context: TaskContext::new(TaskId::new(2)),
            counter: Arc::clone(&boxed_drops),
            payload: [1; INLINE_SCHEDULED_TASK_WORDS + 2],
        }));

        assert_eq!(inline_drops.load(Ordering::Relaxed), 1);
        assert_eq!(boxed_drops.load(Ordering::Relaxed), 0);
    }
}
