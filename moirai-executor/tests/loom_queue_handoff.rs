//! loom model of the injector handoff between single and batched thieves.
//!
//! `WorkerQueues::steal_one` and `WorkerQueueOwner::steal_batch` both claim
//! jobs from the target injector. The batch path claims its first job, drains
//! more jobs into the thief's private queues, and only then decrements the
//! target's advisory `len` by the whole batch. A single thief can therefore
//! observe the target between the first dequeue and that deferred decrement.
//!
//! This model keeps the production operation boundaries and orderings while
//! replacing the unsafe bounded MPMC storage with a mutex-backed linearizable
//! injector. The storage abstraction is deliberately narrower than the
//! production queue: the property under test is the interaction between a
//! successful claim and the deferred `len` update. Each model run asserts that
//! every published job is claimed exactly once and that the advisory count
//! returns to zero. The production queue's slot protocol is covered separately
//! by its own queue tests; this model does not claim to prove that storage.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo nextest run --locked -p moirai-executor
//! --test loom_queue_handoff --release`

#![cfg(loom)]
#![allow(
    clippy::unwrap_used,
    reason = "loom model lock and thread failures are test assertions"
)]

use std::collections::VecDeque;

use loom::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;

const JOBS: usize = 3;
const BATCH_LIMIT: usize = 2;

/// Linearizable injector abstraction used by this ordering model.
struct Injector {
    jobs: Mutex<VecDeque<usize>>,
    queued: AtomicUsize,
    marks: [AtomicU8; JOBS],
}

impl Injector {
    fn new() -> Self {
        Self {
            jobs: Mutex::new((0..JOBS).collect()),
            queued: AtomicUsize::new(JOBS),
            marks: std::array::from_fn(|_| AtomicU8::new(0)),
        }
    }

    /// Mirrors the successful `try_dequeue` claim; the queue lock only stands
    /// in for the production MPMC queue's linearization point.
    fn try_dequeue(&self) -> Option<usize> {
        self.jobs.lock().unwrap().pop_front()
    }

    /// Record a claim before the consumer's deferred accounting update.
    fn record_claim(&self, id: usize) {
        let previous = self.marks[id].fetch_add(1, Ordering::Relaxed);
        assert_eq!(previous, 0, "job claimed more than once: {id}");
    }

    fn queued(&self) -> usize {
        self.queued.load(Ordering::Relaxed)
    }
}

/// Single-item thief, matching `WorkerQueues::steal_one`'s injector path.
fn steal_one(injector: &Injector) {
    if injector.queued.load(Ordering::Relaxed) == 0 {
        return;
    }

    if let Some(id) = injector.try_dequeue() {
        injector.record_claim(id);
        // The decrement follows the successful dequeue, as in production.
        injector.queued.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Batched thief, matching `WorkerQueueOwner::steal_batch`'s injector path.
fn steal_batch(injector: &Injector) {
    if injector.queued.load(Ordering::Relaxed) == 0 {
        return;
    }

    let Some(first) = injector.try_dequeue() else {
        return;
    };
    injector.record_claim(first);

    let mut retained = 0;
    while retained < BATCH_LIMIT - 1 {
        let Some(id) = injector.try_dequeue() else {
            break;
        };
        injector.record_claim(id);
        retained += 1;
        // The real implementation pushes this job into the thief's private
        // local queue. It remains claimed, while target accounting is deferred.
    }

    // Keep the handoff window explicit: another consumer may claim a job before
    // this deferred decrement, exactly as it may between production dequeues.
    thread::yield_now();
    injector.queued.fetch_sub(retained + 1, Ordering::Relaxed);
}

#[test]
fn mixed_single_and_batch_handoff_conserves_jobs() {
    loom::model(|| {
        let injector = Arc::new(Injector::new());

        let batch_injector = Arc::clone(&injector);
        let batch = thread::spawn(move || steal_batch(&batch_injector));

        let single_injector = Arc::clone(&injector);
        let single = thread::spawn(move || steal_one(&single_injector));

        batch.join().unwrap();
        single.join().unwrap();

        for (id, mark) in injector.marks.iter().enumerate() {
            assert_eq!(
                mark.load(Ordering::Relaxed),
                1,
                "published job was not claimed exactly once: {id}"
            );
        }
        assert_eq!(injector.queued(), 0, "deferred accounting must settle");
        assert!(injector.jobs.lock().unwrap().is_empty());
    });
}
