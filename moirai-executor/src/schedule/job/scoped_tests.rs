use std::{
    mem::{size_of, size_of_val},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use super::{INLINE_JOB_WORDS, InlineJobStorage, ScheduledJob};

struct DropCounter {
    drops: Arc<AtomicUsize>,
}

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

struct OversizedTask {
    capture: DropCounter,
    padding: [usize; INLINE_JOB_WORDS],
}

impl OversizedTask {
    fn run(self) {
        let Self { capture, padding } = self;
        std::hint::black_box(padding);
        drop(capture);
    }
}

#[test]
fn scoped_job_drops_task_captures_before_completion() {
    let task_drops = Arc::new(AtomicUsize::new(0));
    let completion_calls = Arc::new(AtomicUsize::new(0));
    let task_drop_view = Arc::clone(&task_drops);
    let task_capture = DropCounter { drops: task_drops };
    let complete = {
        let completion_calls = Arc::clone(&completion_calls);
        move |succeeded: bool| {
            assert!(succeeded);
            assert_eq!(task_drop_view.load(Ordering::Relaxed), 1);
            completion_calls.fetch_add(1, Ordering::Relaxed);
        }
    };

    // SAFETY: the job executes synchronously while every capture is live.
    let job = unsafe {
        ScheduledJob::new_scoped_with_completion(
            move |_| {
                std::hint::black_box(&task_capture);
            },
            complete,
        )
    };

    assert!(job.execute(0));
    assert_eq!(completion_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn boxed_scoped_job_drops_task_captures_before_completion() {
    let task_drops = Arc::new(AtomicUsize::new(0));
    let task_drop_view = Arc::clone(&task_drops);
    let oversized = OversizedTask {
        capture: DropCounter { drops: task_drops },
        padding: [7; INLINE_JOB_WORDS],
    };
    let task = move |_| oversized.run();
    assert!(size_of_val(&task) > size_of::<InlineJobStorage>());

    // SAFETY: the job executes synchronously while every capture is live.
    let job = unsafe {
        ScheduledJob::new_scoped_with_completion(task, move |succeeded| {
            assert!(succeeded);
            assert_eq!(task_drop_view.load(Ordering::Relaxed), 1);
        })
    };

    assert!(job.execute(0));
}

#[test]
fn dropped_scoped_job_releases_task_and_completion_without_calling() {
    let task_drops = Arc::new(AtomicUsize::new(0));
    let completion_drops = Arc::new(AtomicUsize::new(0));
    let completion_calls = Arc::new(AtomicUsize::new(0));
    let task_capture = DropCounter {
        drops: Arc::clone(&task_drops),
    };
    let completion_capture = DropCounter {
        drops: Arc::clone(&completion_drops),
    };
    let complete = {
        let completion_calls = Arc::clone(&completion_calls);
        move |_| {
            completion_calls.fetch_add(1, Ordering::Relaxed);
            drop(completion_capture);
        }
    };

    // SAFETY: the job is dropped synchronously while every capture is live.
    let job = unsafe {
        ScheduledJob::new_scoped_with_completion(
            move |_| {
                std::hint::black_box(&task_capture);
            },
            complete,
        )
    };
    drop(job);

    assert_eq!(task_drops.load(Ordering::Relaxed), 1);
    assert_eq!(completion_drops.load(Ordering::Relaxed), 1);
    assert_eq!(completion_calls.load(Ordering::Relaxed), 0);
}

#[test]
fn scoped_job_reports_panic_before_failed_completion() {
    let completion_outcome = Arc::new(AtomicUsize::new(0));
    let complete = {
        let completion_outcome = Arc::clone(&completion_outcome);
        move |succeeded: bool| {
            completion_outcome.store(if succeeded { 1 } else { 2 }, Ordering::Relaxed);
        }
    };

    // SAFETY: the job executes synchronously and borrows no external data.
    let job = unsafe {
        ScheduledJob::new_scoped_with_completion(|_| panic!("scoped job panic"), complete)
    };

    assert!(!job.execute(0));
    assert_eq!(completion_outcome.load(Ordering::Relaxed), 2);
}

#[test]
fn the_borrowed_frame_may_be_freed_while_the_completion_returns() {
    // A plain integer, not an atomic: a shared reference to it is read-only for
    // its whole extent, which is the retag that must not outlive the release.
    let frame = Box::new(7usize);
    let observed = Arc::new(AtomicUsize::new(0));
    // SAFETY: the reference is only read while the task runs, which finishes
    // before `complete` sends the release; `frame` is freed after that send
    // and the extended lifetime is never used again.
    let borrowed: &'static usize = unsafe { &*std::ptr::from_ref(&*frame) };
    let (release, released) = std::sync::mpsc::sync_channel::<()>(1);
    let (freed_tx, freed) = std::sync::mpsc::sync_channel::<()>(1);
    // Completion stays inside `execute` until the frame is gone, so the free
    // provably happens while the job's own call is still on the stack.
    let complete = move |succeeded: bool| {
        assert!(succeeded);
        release
            .send(())
            .expect("the frame owner waits for the release");
        freed.recv().expect("the frame owner reports the free");
    };

    // SAFETY: the task's borrow is dead once it returns, and `complete` runs
    // strictly after that; `frame` outlives the task's use of `borrowed`.
    let job = unsafe {
        let observed = Arc::clone(&observed);
        ScheduledJob::new_scoped_with_completion(
            move |_| observed.store(*borrowed, Ordering::Relaxed),
            complete,
        )
    };
    let runner = std::thread::spawn(move || job.execute(0));

    released.recv().expect("completion sends the release");
    drop(frame);
    freed_tx.send(()).expect("completion waits for the free");
    assert!(runner.join().expect("the runner does not panic"));
    assert_eq!(observed.load(Ordering::Relaxed), 7);
}
