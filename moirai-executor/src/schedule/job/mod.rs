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

#[repr(C)]
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
        // Checked in release too, not just under `debug_assert`: this is a safe
        // function whose precondition is UB-critical, since an `F` too large for
        // the storage would be written past it, and a debug-only guard
        // disappears exactly where the consequence stops being a panic.
        //
        // It costs nothing. Both operands are compile-time constants, so for a
        // type that fits this folds to `assert!(true)` and leaves no code.
        //
        // A `const` assertion cannot be used here even though the operands are
        // constant: `ScheduledJob::new` picks between the inline and boxed forms
        // at runtime, and the inline branch is still *instantiated* for an
        // oversized closure it never takes. Asserting at compile time therefore
        // rejects the boxed path's own callers.
        assert!(inline_job_fits::<F>());
        let mut job = Self {
            storage: InlineJobStorage::new(),
            execute: execute_inline::<F>,
            drop: drop_inline::<F>,
        };

        // Safety: `inline_job_fits` proves size and alignment fit the storage.
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

const fn inline_job_fits<F>() -> bool {
    size_of::<F>() <= size_of::<InlineJobStorage>()
        && align_of::<F>() <= align_of::<InlineJobStorage>()
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

    use super::{inline_job_fits, InlineJob, InlineJobStorage, ScheduledJob, INLINE_JOB_WORDS};

    struct DropCounter {
        drops: Arc<AtomicUsize>,
    }

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    struct OversizedCapture {
        counter: DropCounter,
        padding: [usize; INLINE_JOB_WORDS],
    }

    impl OversizedCapture {
        fn run(self, worker_id: usize) {
            assert_eq!(self.padding[0], worker_id);
            drop(self.counter);
        }
    }

    #[repr(align(16))]
    struct OverAlignedCapture {
        counter: DropCounter,
    }

    impl OverAlignedCapture {
        fn run(self, worker_id: usize) {
            assert_eq!(worker_id, 0);
            drop(self.counter);
        }
    }

    enum ExpectedStorage {
        Inline,
        Boxed,
    }

    fn assert_dropped_once_before_and_after_execution<F, M>(expected: ExpectedStorage, make_task: M)
    where
        F: FnOnce(usize) + Send + 'static,
        M: Fn(Arc<AtomicUsize>) -> F,
    {
        match expected {
            ExpectedStorage::Inline => assert!(inline_job_fits::<F>()),
            ExpectedStorage::Boxed => assert!(!inline_job_fits::<F>()),
        }

        let unexecuted_drops = Arc::new(AtomicUsize::new(0));
        drop(ScheduledJob::new(make_task(Arc::clone(&unexecuted_drops))));
        assert_eq!(unexecuted_drops.load(Ordering::Relaxed), 1);

        let executed_drops = Arc::new(AtomicUsize::new(0));
        let job = ScheduledJob::new(make_task(Arc::clone(&executed_drops)));
        assert!(job.execute(0));
        assert_eq!(executed_drops.load(Ordering::Relaxed), 1);
    }

    fn scheduled_job_requiring_box<F>(task: F) -> ScheduledJob
    where
        F: FnOnce(usize) + Send + 'static,
    {
        assert!(!inline_job_fits::<F>());
        ScheduledJob::new(task)
    }

    #[test]
    fn inline_job_uses_natural_alignment_with_same_capacity() {
        let word_size = core::mem::size_of::<usize>();

        assert_eq!(
            core::mem::size_of::<InlineJobStorage>(),
            INLINE_JOB_WORDS * word_size
        );
        assert_eq!(
            core::mem::size_of::<InlineJob>(),
            (INLINE_JOB_WORDS + 2) * word_size
        );
        assert_eq!(
            core::mem::size_of::<ScheduledJob>(),
            core::mem::size_of::<InlineJob>()
        );
        assert_eq!(
            core::mem::align_of::<InlineJob>(),
            core::mem::align_of::<InlineJobStorage>()
        );
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
    fn maximum_inline_capacity_job_uses_inline_storage() {
        let values = [1usize; INLINE_JOB_WORDS];
        let job = ScheduledJob::new(move |worker_id| {
            assert_eq!(values[worker_id], 1);
        });

        assert!(job.execute(0));
    }

    #[test]
    fn over_aligned_job_uses_typed_boxed_trampoline() {
        let observed = Arc::new(AtomicUsize::new(0));
        #[repr(align(16))]
        struct ObservedCapture(Arc<AtomicUsize>);

        impl ObservedCapture {
            fn record(self, worker_id: usize) {
                self.0.store(worker_id + 1, Ordering::Relaxed);
            }
        }

        let capture = ObservedCapture(Arc::clone(&observed));
        let job = scheduled_job_requiring_box(move |worker_id| capture.record(worker_id));

        assert!(job.execute(8));
        assert_eq!(observed.load(Ordering::Relaxed), 9);
    }

    #[test]
    fn inline_job_drops_capture_once_before_and_after_execution() {
        assert_dropped_once_before_and_after_execution(ExpectedStorage::Inline, |drops| {
            let counter = DropCounter { drops };
            move |_| drop(counter)
        });
    }

    #[test]
    fn oversized_job_drops_capture_once_before_and_after_execution() {
        assert_dropped_once_before_and_after_execution(ExpectedStorage::Boxed, |drops| {
            let capture = OversizedCapture {
                counter: DropCounter { drops },
                padding: [0; INLINE_JOB_WORDS],
            };
            move |worker_id| capture.run(worker_id)
        });
    }

    #[test]
    fn over_aligned_job_drops_capture_once_before_and_after_execution() {
        assert_dropped_once_before_and_after_execution(ExpectedStorage::Boxed, |drops| {
            let capture = OverAlignedCapture {
                counter: DropCounter { drops },
            };
            move |worker_id| capture.run(worker_id)
        });
    }

    #[test]
    fn inline_job_reports_panic_as_failure() {
        let job = ScheduledJob::new(|_| panic!("job panic"));

        assert!(!job.execute(0));
    }
}
