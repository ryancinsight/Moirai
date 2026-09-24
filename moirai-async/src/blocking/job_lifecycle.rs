//! Checks every [`BlockingPool`] must pass, written once and instantiated per
//! pool by the modules that own one: a dropped operation releases its slot at
//! every stage, the thread count never exceeds the bound, and a panic fails
//! only its own job. The gate hook pins each stage, so no check depends on
//! timing.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::{Arc, mpsc};
use std::task::{Context, Poll, Wake, Waker};

use futures::executor::block_on;

use super::BlockingPool;
use super::test_hooks::{self, STAGE_LIMIT};

/// One operation on the pool under test: a real public entry point that runs
/// a single [`Abandoned::Skip`](super::Abandoned::Skip) job and succeeds.
pub(crate) type Operation = Pin<Box<dyn Future<Output = io::Result<()>> + Send>>;

/// The pool under test and a way to start one operation on it.
#[derive(Clone, Copy)]
pub(crate) struct Subject {
    pub(crate) pool: fn() -> &'static BlockingPool,
    pub(crate) operation: fn() -> Operation,
}

impl Subject {
    fn pool(self) -> &'static BlockingPool {
        (self.pool)()
    }

    fn start(self) -> Operation {
        let mut operation = (self.operation)();
        let waker = futures::task::noop_waker();
        assert!(
            matches!(
                operation.as_mut().poll(&mut Context::from_waker(&waker)),
                Poll::Pending
            ),
            "a gated operation cannot complete"
        );
        operation
    }

    /// Run one operation on a helper thread and wait at most [`STAGE_LIMIT`],
    /// so a pool with no live worker fails the check instead of hanging it.
    fn run_within_limit(self) -> io::Result<()> {
        let (sender, outcome) = mpsc::channel();
        std::thread::spawn(move || {
            sender
                .send(block_on((self.operation)()))
                .expect("the check awaits the outcome");
        });
        outcome
            .recv_timeout(STAGE_LIMIT)
            .expect("an operation must finish; a pool with no live worker hangs it")
    }

    /// Fill every worker with a gated job and wait until all of them run.
    fn occupy_workers(self, started: usize) -> Vec<Operation> {
        let workers = self.pool().workers_bound();
        let running: Vec<Operation> = (0..workers).map(|_| self.start()).collect();
        let progress = self
            .pool()
            .hooks()
            .wait_until(|progress| progress.started == started + workers);
        assert_eq!(progress.live, workers);
        running
    }
}

fn finish(operation: Operation) {
    block_on(operation).expect("a released operation must succeed");
}

/// A future dropped while it waits for admission takes no slot, and the next
/// waiter is admitted in its place.
pub(crate) fn dropped_admission_waiter_returns_no_slot(subject: Subject) {
    let _exclusive = test_hooks::exclusive();
    let (pool, hooks) = (subject.pool(), subject.pool().hooks());
    let baseline = hooks.progress();
    hooks.set_gate_closed(true);

    let mut admitted = subject.occupy_workers(baseline.started);
    admitted.extend((pool.workers_bound()..pool.admissions()).map(|_| subject.start()));
    assert_eq!(pool.free_admissions(), 0);

    let abandoned = subject.start();
    let successor = subject.start();
    drop(abandoned);

    hooks.set_gate_closed(false);
    admitted.into_iter().for_each(finish);
    finish(successor);

    let settled = pool.admissions() + 1;
    let progress = hooks.wait_until(|progress| progress.disposed == baseline.disposed + settled);
    assert_eq!(progress.disposed - baseline.disposed, settled);
    assert_eq!(
        progress.started - baseline.started,
        settled,
        "the abandoned waiter was never admitted, so it never ran"
    );
    assert_eq!(
        pool.free_admissions(),
        pool.admissions(),
        "a slot granted to the dropped waiter was lost"
    );
}

/// A future dropped while its job is queued: the worker skips the job and
/// releases its slot.
pub(crate) fn dropped_queued_job_is_skipped(subject: Subject) {
    let _exclusive = test_hooks::exclusive();
    let (pool, hooks) = (subject.pool(), subject.pool().hooks());
    let baseline = hooks.progress();
    hooks.set_gate_closed(true);

    let running = subject.occupy_workers(baseline.started);
    let abandoned = subject.start();
    let kept = subject.start();
    assert_eq!(
        pool.free_admissions(),
        pool.admissions() - pool.workers_bound() - 2
    );
    drop(abandoned);

    hooks.set_gate_closed(false);
    running.into_iter().for_each(finish);
    finish(kept);

    let settled = pool.workers_bound() + 2;
    let progress = hooks.wait_until(|progress| progress.disposed == baseline.disposed + settled);
    assert_eq!(progress.disposed - baseline.disposed, settled);
    assert_eq!(
        progress.started - baseline.started,
        settled - 1,
        "the dropped queued job must not run"
    );
    assert_eq!(pool.free_admissions(), pool.admissions());
}

/// A future dropped while its job runs: the slot stays taken until the job
/// returns, then comes back.
pub(crate) fn dropped_running_job_releases_its_slot_on_return(subject: Subject) {
    let _exclusive = test_hooks::exclusive();
    let (pool, hooks) = (subject.pool(), subject.pool().hooks());
    let baseline = hooks.progress();
    hooks.set_gate_closed(true);

    let abandoned = subject.start();
    hooks.wait_until(|progress| progress.started == baseline.started + 1);
    drop(abandoned);
    assert_eq!(
        pool.free_admissions(),
        pool.admissions() - 1,
        "a running job holds its slot until it returns"
    );

    hooks.set_gate_closed(false);
    let progress = hooks.wait_until(|progress| progress.disposed == baseline.disposed + 1);
    assert_eq!(progress.disposed - baseline.disposed, 1);
    assert_eq!(progress.started - baseline.started, 1);
    assert_eq!(pool.free_admissions(), pool.admissions());
}

/// Panics in more jobs than there are workers each fail only their own
/// operation; the pool keeps serving at full size.
pub(crate) fn panicking_jobs_fail_alone(subject: Subject) {
    let _exclusive = test_hooks::exclusive();
    let (pool, hooks) = (subject.pool(), subject.pool().hooks());
    let baseline = hooks.progress();
    let panics = pool.workers_bound() + 1;
    hooks.inject_panics(panics);

    for _ in 0..panics {
        let failure = subject
            .run_within_limit()
            .expect_err("an injected panic fails its operation");
        assert_eq!(failure.kind(), io::ErrorKind::Other);
    }
    subject
        .run_within_limit()
        .expect("the pool must still serve operations");

    let progress = hooks.wait_until(|progress| progress.disposed == baseline.disposed + panics + 1);
    assert_eq!(progress.disposed - baseline.disposed, panics + 1);
    assert_eq!(pool.workers(), pool.workers_bound());
    assert_eq!(pool.free_admissions(), pool.admissions());
}

/// A waker that panics when woken, as a buggy executor's might.
struct PanickingWaker;

impl Wake for PanickingWaker {
    fn wake(self: Arc<Self>) {
        panic!("injected waker panic");
    }
}

/// A caller's waker that panics when the reply wakes it does not take a
/// worker down.
pub(crate) fn panicking_waker_leaves_the_worker_serving(subject: Subject) {
    let _exclusive = test_hooks::exclusive();
    let hooks = subject.pool().hooks();
    let baseline = hooks.progress();
    let waker = Waker::from(Arc::new(PanickingWaker));
    let rounds = subject.pool().workers_bound() + 1;
    for done in 0..rounds {
        // The gate holds the worker inside the job until the panicking waker
        // is registered, so the worker's reply is what wakes it.
        hooks.set_gate_closed(true);
        let mut operation = (subject.operation)();
        assert!(matches!(
            operation.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Pending
        ));
        hooks.wait_until(|progress| progress.started == baseline.started + done + 1);
        hooks.set_gate_closed(false);
        hooks.wait_until(|progress| progress.disposed == baseline.disposed + done + 1);
        finish(operation);
    }
    subject
        .run_within_limit()
        .expect("the pool must still serve operations");
}
