//! Scheduled job representation.

use core::{
    mem::{align_of, size_of, MaybeUninit},
    ptr,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

const INLINE_JOB_WORDS: usize = 14;

/// Job stored inside worker queues.
pub(crate) struct ScheduledJob {
    job: InlineJob,
}

#[repr(C, align(64))]
pub(crate) struct InlineJob {
    storage: InlineJobStorage,
    execute: unsafe fn(*mut InlineJobStorage, usize) -> bool,
    drop: unsafe fn(*mut InlineJobStorage),
}

#[repr(C)]
struct InlineJobStorage {
    words: [MaybeUninit<usize>; INLINE_JOB_WORDS],
}

impl ScheduledJob {
    /// Create a scheduled job from a worker-aware closure.
    pub(crate) fn new<F>(task: F) -> Self
    where
        F: FnOnce(usize) + Send + 'static,
    {
        if inline_job_fits::<F>() {
            Self {
                job: InlineJob::new(task),
            }
        } else {
            Self {
                job: InlineJob::new(boxed_job(task)),
            }
        }
    }

    /// Create a scheduled job from a scoped worker-aware closure.
    ///
    /// # Safety
    ///
    /// The caller must prove that every queued job is either executed or
    /// dropped before the borrowed scope ends.
    pub(crate) unsafe fn new_scoped<'scope, F>(task: F) -> Self
    where
        F: FnOnce(usize) + Send + 'scope,
    {
        if inline_job_fits::<F>() {
            Self {
                job: InlineJob::new(task),
            }
        } else {
            Self {
                job: InlineJob::new(boxed_job(task)),
            }
        }
    }

    /// Execute the job exactly once.
    pub(crate) fn execute(mut self, worker_id: usize) -> bool {
        self.job.execute(worker_id)
    }
}

impl InlineJob {
    fn new<F>(task: F) -> Self
    where
        F: FnOnce(usize) + Send,
    {
        debug_assert!(inline_job_fits::<F>());
        let mut job = Self {
            storage: InlineJobStorage::new(),
            execute: execute_inline::<F>,
            drop: drop_inline::<F>,
        };

        // Safety: `inline_job_fits` proves size and alignment fit the storage.
        // The enclosing `InlineJob` supplies cache-line alignment and the
        // storage field is at offset zero, so the field address has that
        // alignment even though the storage type itself remains compact.
        unsafe {
            job.storage.as_mut_ptr::<F>().write(task);
        }

        job
    }

    fn execute(&mut self, worker_id: usize) -> bool {
        self.drop = drop_consumed;
        // Safety: `InlineJob::new` initialized storage as the same `F` used to
        // create this function pointer. Replacing `drop` before execution
        // prevents a second drop after `execute_inline` moves the closure out.
        unsafe { (self.execute)(&mut self.storage, worker_id) }
    }
}

impl Drop for InlineJob {
    fn drop(&mut self) {
        // Safety: storage contains the initialized closure until `execute`
        // swaps this function pointer to `drop_consumed`.
        unsafe {
            (self.drop)(&mut self.storage);
        }
    }
}

impl InlineJobStorage {
    fn new() -> Self {
        Self {
            words: [MaybeUninit::uninit(); INLINE_JOB_WORDS],
        }
    }

    fn as_mut_ptr<T>(&mut self) -> *mut T {
        self.words.as_mut_ptr().cast::<T>()
    }
}

fn inline_job_fits<F>() -> bool {
    size_of::<F>() <= size_of::<InlineJobStorage>() && align_of::<F>() <= align_of::<InlineJob>()
}

fn boxed_job<F>(task: F) -> impl FnOnce(usize) + Send
where
    F: FnOnce(usize) + Send,
{
    let task = Box::new(task);
    move |worker_id| task(worker_id)
}

unsafe fn execute_inline<F>(storage: *mut InlineJobStorage, worker_id: usize) -> bool
where
    F: FnOnce(usize) + Send,
{
    // Safety: `InlineJob::new` initialized this storage as `F`; execute reads
    // it exactly once before marking the inline job consumed.
    let task = unsafe { ptr::read((*storage).as_mut_ptr::<F>()) };
    catch_unwind(AssertUnwindSafe(|| task(worker_id))).is_ok()
}

unsafe fn drop_inline<F>(storage: *mut InlineJobStorage)
where
    F: FnOnce(usize) + Send,
{
    // Safety: called only when the inline job was not consumed.
    unsafe {
        ptr::drop_in_place((*storage).as_mut_ptr::<F>());
    }
}

unsafe fn drop_consumed(_: *mut InlineJobStorage) {}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use super::{InlineJob, InlineJobStorage, ScheduledJob, INLINE_JOB_WORDS};
    use moirai_core::constants::CACHE_LINE_SIZE;

    #[test]
    fn inline_job_uses_two_cache_line_budget() {
        assert_eq!(core::mem::size_of::<InlineJob>(), CACHE_LINE_SIZE * 2);
        assert_eq!(core::mem::align_of::<InlineJob>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn small_job_uses_inline_storage() {
        let observed = Arc::new(AtomicUsize::new(0));
        let job = {
            let observed = Arc::clone(&observed);
            ScheduledJob::new(move |worker_id| {
                observed.store(worker_id + 1, Ordering::Relaxed);
            })
        };

        assert!(job.execute(6));
        assert_eq!(observed.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn oversized_job_uses_boxed_inline_trampoline() {
        let observed = Arc::new(AtomicUsize::new(0));
        let job = {
            let observed = Arc::clone(&observed);
            let values = [1usize; INLINE_JOB_WORDS + 1];
            assert!(core::mem::size_of_val(&values) > core::mem::size_of::<InlineJobStorage>());
            ScheduledJob::new(move |worker_id| {
                observed.store(values[0] + worker_id, Ordering::Relaxed);
            })
        };

        assert!(job.execute(4));
        assert_eq!(observed.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn maximum_two_cache_line_job_uses_inline_storage() {
        let values = [1usize; INLINE_JOB_WORDS];
        let job = ScheduledJob::new(move |worker_id| {
            assert_eq!(values[worker_id], 1);
        });

        assert!(job.execute(0));
    }

    #[test]
    fn inline_job_reports_panic_as_failure() {
        let job = ScheduledJob::new(|_| panic!("job panic"));

        assert!(!job.execute(0));
    }
}
