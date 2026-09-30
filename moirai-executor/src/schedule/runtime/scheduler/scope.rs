//! SchedulerScope implementation.

use std::{
    marker::PhantomData,
    mem,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr::NonNull,
    sync::atomic::Ordering,
};

use moirai_core::{
    Priority,
    error::{ExecutorError, ExecutorResult},
};

use super::super::super::{class::WorkClass, job::ScheduledJob};
use super::super::scope_state::{SchedulerScopeState, ScopedTaskCompletion};
use super::super::types::{SchedulerScope, ThreadScheduler, get_current_worker_id};
use super::super::worker::{execute_job, lock_mutex, next_shared_job};

impl<'scope, C, const BLOCKING_QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>
    SchedulerScope<'scope, C, BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>
where
    C: WorkClass,
{
    /// Spawn a job into this scope.
    ///
    /// The job may borrow values that outlive the scope call. Scoped jobs are
    /// coalesced into worker-sized scheduler batches and complete before
    /// `ThreadScheduler::scope` returns. Jobs are not guaranteed to start while
    /// the scope body is still registering work.
    ///
    /// The `usize` the job receives identifies the lane running it. Worker
    /// lanes are `0..worker_count()`; a job the admission queue turned away
    /// runs on the calling lane, identified as `worker_count()`. It is a lane
    /// identity, not an index into the worker set.
    pub fn spawn<F>(&self, task: F) -> ExecutorResult<()>
    where
        F: FnOnce(usize) + Send + 'scope,
    {
        self.state().register_task();
        let completion = ScopedTaskCompletion::new(self.state());
        let complete = move |succeeded: bool| completion.finish(succeeded);

        // SAFETY: `ThreadScheduler::scope` waits for every scheduled scoped
        // job and drops unscheduled buffered jobs before borrowed scope data
        // can expire.
        let job = unsafe { ScheduledJob::new_scoped_with_completion(task, complete) };
        self.jobs.borrow_mut().push(job);
        Ok(())
    }

    /// Schedule all jobs currently buffered in this scope.
    ///
    /// `ThreadScheduler::scope` calls this before waiting, so most callers do
    /// not need to invoke it directly. It is exposed for two-lane fork/join
    /// shapes where one branch should enter the scheduler before the caller
    /// executes the second branch locally. The scope still waits for every
    /// flushed job before returning, so borrowed data cannot escape.
    pub fn flush(&self) -> ExecutorResult<()> {
        let jobs = mem::take(&mut *self.jobs.borrow_mut());
        if jobs.is_empty() {
            return Ok(());
        }

        if jobs.len() == 1 {
            let job = jobs
                .into_iter()
                .next()
                .expect("single scoped job must exist");
            return self.schedule_single(job);
        }

        let worker_count = self.scheduler.worker_count();
        let chunk_count = jobs.len().min(worker_count.max(1));
        let chunk_size = jobs.len().div_ceil(chunk_count);
        let spread_start = self
            .locality_hint
            .is_none()
            .then(|| self.scheduler.select_worker::<C>(self.priority, None));
        let mut pending_jobs = jobs.into_iter();

        for chunk_index in 0..chunk_count {
            let mut chunk = Vec::with_capacity(chunk_size);
            for _ in 0..chunk_size {
                if let Some(job) = pending_jobs.next() {
                    chunk.push(job);
                }
            }

            if chunk.is_empty() {
                break;
            }

            // Select the unhinted batch base once, then distribute physical
            // batches across distinct workers. Re-running selection after each
            // admission lets a fast first batch change pending/active state and
            // route later batches back to its occupied lane, defeating the
            // worker-sized coalescing contract and deadlocking saturated joins.
            let locality_hint = self.locality_hint.or_else(|| {
                spread_start.map(|start| start.wrapping_add(chunk_index) % worker_count)
            });
            self.schedule_chunk(chunk, locality_hint)?;
        }

        Ok(())
    }

    fn schedule_single(&self, job: ScheduledJob) -> ExecutorResult<()> {
        self.schedule_job(job, self.locality_hint)
    }

    fn schedule_job(&self, job: ScheduledJob, locality_hint: Option<usize>) -> ExecutorResult<()> {
        let mut job = Some(job);
        let admitted = self
            .scheduler
            .admit_job::<C>(self.priority, locality_hint, &mut job);
        self.run_if_refused(admitted, job)
    }

    fn schedule_chunk(
        &self,
        jobs: Vec<ScheduledJob>,
        locality_hint: Option<usize>,
    ) -> ExecutorResult<()> {
        let scoped_job = move |worker_id| {
            for job in jobs {
                let _ = job.execute(worker_id);
            }
        };

        // Safety: `ThreadScheduler::scope` waits for every scheduled scoped job
        // and drops unscheduled buffered jobs before borrowed scope data can
        // expire. A refused job runs below, inside the same scope, so it
        // observes the same live borrows.
        let job = unsafe { ScheduledJob::new_scoped(scoped_job) };
        self.schedule_job(job, locality_hint)
    }

    /// Run a job the scheduler refused on the calling lane.
    ///
    /// A scope promises its caller that every spawned job runs before the scope
    /// returns. Dropping a job the admission queue rejected breaks that promise
    /// silently: the caller blocks until the scope joins and then continues as
    /// though the work happened. Running it here keeps the promise at the cost
    /// of the parallelism that job would have had — the same trade
    /// `for_each_indexed` already makes with a rejected chunk, counted by the
    /// same `admission_caller_runs` surface.
    ///
    /// Shutdown is not backpressure and is not absorbed: a scheduler that is
    /// going away refuses the work, and the error reaches the caller.
    fn run_if_refused(
        &self,
        admitted: ExecutorResult<()>,
        refused: Option<ScheduledJob>,
    ) -> ExecutorResult<()> {
        match (admitted, refused) {
            (Ok(()), _) => Ok(()),
            (Err(ExecutorError::ResourceExhausted(_)), Some(job)) => {
                self.scheduler.record_admission_caller_run();
                // `execute` contains its own unwind boundary, so a panicking
                // job marks its completion token failed exactly as it would on
                // a worker instead of unwinding through the scope body.
                let _ = job.execute(self.scheduler.caller_lane_id());
                Ok(())
            }
            (Err(error), _) => Err(error),
        }
    }

    fn state(&self) -> &SchedulerScopeState {
        // Safety: `ThreadScheduler::scope` creates this pointer from a local
        // state value and waits for every scheduled scoped job before returning.
        unsafe { self.state.as_ref() }
    }
}

/// Busy-spin iterations a worker-thread scope waiter performs after exhausting
/// runnable work before it parks on the scope condvar. The waiter only reaches
/// this path when its remaining scoped jobs are actively executing on other
/// workers (nothing left to steal), so a short spin absorbs the common
/// finish-imminently case without an OS park round-trip; the timed park below
/// then bounds idle-CPU while `complete_task` provides the real wakeup.
const SCOPE_HELP_SPIN_LIMIT: usize = 64;

impl<const BLOCKING_QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>
    ThreadScheduler<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>
{
    /// Run a borrowing job scope on the scheduler and wait for all spawned jobs.
    ///
    /// This is the scheduler-equivalent of a scoped fan-out. It avoids per-task
    /// result storage when the caller only needs completion, while preserving the
    /// invariant that borrowed data cannot outlive the scope.
    ///
    /// Jobs may borrow data that outlives this call:
    ///
    /// ```
    /// use moirai_core::Priority;
    /// use moirai_executor::{SyncTask, ThreadScheduler};
    ///
    /// let scheduler = ThreadScheduler::new(2, "scope-doc").unwrap();
    /// let total = std::sync::atomic::AtomicU64::new(0);
    /// let values = [1_u64, 2, 3];
    /// scheduler
    ///     .scope::<SyncTask, _>(Priority::Normal, None, |scope| {
    ///         for value in &values {
    ///             scope.spawn(|_| {
    ///                 total.fetch_add(*value, std::sync::atomic::Ordering::Relaxed);
    ///             })?;
    ///         }
    ///         Ok(())
    ///     })
    ///     .unwrap();
    /// assert_eq!(total.into_inner(), 6);
    /// ```
    ///
    /// A job cannot borrow a value local to the body. The body returns and drops
    /// it before the buffered job is scheduled:
    ///
    /// ```compile_fail,E0597
    /// use moirai_core::Priority;
    /// use moirai_executor::{SyncTask, ThreadScheduler};
    ///
    /// let scheduler = ThreadScheduler::new(2, "scope-doc").unwrap();
    /// scheduler
    ///     .scope::<SyncTask, _>(Priority::Normal, None, |scope| {
    ///         let local = vec![7_u8; 64];
    ///         let borrowed: &Vec<u8> = &local;
    ///         scope.spawn(move |_| assert_eq!(borrowed[0], 7))
    ///     })
    ///     .unwrap();
    /// ```
    pub fn scope<'scope, C, F>(
        &'scope self,
        priority: Priority,
        locality_hint: Option<usize>,
        body: F,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(
            &SchedulerScope<'scope, C, BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>,
        ) -> ExecutorResult<()>,
    {
        if self.inner.shutdown.load(Ordering::Acquire) {
            return Err(ExecutorError::ShuttingDown);
        }

        let state = SchedulerScopeState::new();
        let scope = SchedulerScope {
            scheduler: self,
            state: NonNull::from(&state),
            priority,
            locality_hint,
            jobs: std::cell::RefCell::new(Vec::new()),
            _scope: PhantomData,
            _class: PhantomData,
        };

        let body_result = catch_unwind(AssertUnwindSafe(|| body(&scope)));
        // `flush` may enqueue lifetime-erased borrowing jobs before an internal
        // unwind. Catch it so the scope state remains live through the drain.
        let flush_result = catch_unwind(AssertUnwindSafe(|| scope.flush()));
        // A panic here would unwind `scope` while buffered or running jobs still
        // hold `state` and the caller's borrows, so it cannot be allowed to
        // escape: the process aborts, as `std::thread::scope` does.
        if catch_unwind(AssertUnwindSafe(|| self.drain_scope(&state))).is_err() {
            std::process::abort();
        }

        match body_result {
            Err(payload) => std::panic::resume_unwind(payload),
            Ok(body_result) => match flush_result {
                Err(payload) => std::panic::resume_unwind(payload),
                Ok(flush_result) => match body_result {
                    Ok(()) if state.has_panicked() => Err(ExecutorError::SpawnFailed(
                        moirai_core::error::TaskError::Panicked,
                    )),
                    Ok(()) => flush_result.and_then(|()| state.unrun_job_result()),
                    Err(error) => Err(error),
                },
            },
        }
    }

    /// Wait for every job registered on `state` to complete.
    ///
    /// If the caller is itself a scheduler worker, it participates in work
    /// stealing instead of parking: a worker that blocks inside `scope` while
    /// its nested scoped jobs sit unrun would otherwise remove itself from the
    /// pool and deadlock the fork-join (provably so on a single-worker pool, and
    /// a source of use-after-free on the scope's stack-owned state under
    /// concurrent nesting). Running its own queue via `next_job` keeps the pool
    /// making progress, so nesting is deadlock-free and the scope state stays
    /// live until every borrowing job has completed. `next_job(worker_id)` only
    /// pops this worker's own deque and steals into it, so the aliasing rules of
    /// the single-owner Chase–Lev deques are preserved.
    ///
    /// A non-worker caller parks (`SchedulerScopeState::wait`): the worker pool
    /// drains its scoped jobs, so it never starves anything by blocking. A
    /// caller that helped from its own lane crashed consumers (moirai-iter's
    /// nested iteration on CI, kwavers' 3-D FFT at 32³ and above); until the
    /// race is found the join waits as before.
    pub(super) fn drain_scope(&self, state: &SchedulerScopeState) {
        let Some(worker_id) = self.own_worker_id() else {
            state.wait();
            return;
        };

        let inner = &self.inner;
        let mut idle_spins = 0usize;
        loop {
            if state.pending_tasks.load(Ordering::Acquire) == 0 {
                state.wait();
                return;
            }

            if let Some(job) = next_shared_job(inner, worker_id) {
                execute_job(inner, worker_id, job);
                idle_spins = 0;
                continue;
            }

            // Scope still pending but nothing runnable: the remaining scoped jobs
            // are executing on other workers. Spin briefly, then park on the
            // scope condvar with a timeout so `complete_task` wakes us while we
            // still periodically re-probe for freshly stealable work.
            if idle_spins < SCOPE_HELP_SPIN_LIMIT {
                idle_spins += 1;
                core::hint::spin_loop();
                continue;
            }
            idle_spins = 0;

            let guard = lock_mutex(&state.wait_lock);
            if state.pending_tasks.load(Ordering::Acquire) != 0 {
                let _ = state
                    .wait_signal
                    .wait_timeout(guard, std::time::Duration::from_micros(50))
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        }
    }

    /// Index of the calling thread among this scheduler's own workers.
    ///
    /// The worker-id thread cache is process-wide, so it names whichever
    /// scheduler owns the thread. Indexing this scheduler's worker table with
    /// a foreign id panics or drains the wrong deques, so membership is
    /// confirmed against the thread each worker registers when it starts.
    fn own_worker_id(&self) -> Option<usize> {
        let id = get_current_worker_id()?;
        let registered = self.inner.workers.get(id)?.thread.get()?;
        (registered.id() == std::thread::current().id()).then_some(id)
    }
}
