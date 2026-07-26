//! SchedulerScope implementation.

use std::{
    marker::PhantomData,
    mem,
    panic::{catch_unwind, AssertUnwindSafe},
};

use moirai_core::error::{ExecutorError, ExecutorResult};

use super::super::super::{class::WorkClass, job::ScheduledJob};
use super::super::types::{SchedulerScope, SchedulerScopeState, ScopedTaskCompletion};

impl<'scope, C, const QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>
    SchedulerScope<'scope, C, QUEUE_CAPACITY, SPIN_LIMIT>
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
        let completion = ScopedTaskCompletion {
            state: self.state,
            _state: PhantomData,
        };
        let scoped_task = move |worker_id| {
            let _completion = completion;
            let result = catch_unwind(AssertUnwindSafe(|| task(worker_id)));
            if result.is_err() {
                _completion.mark_failed();
            }
        };

        // Safety: `ThreadScheduler::scope` waits for every scheduled scoped
        // job and drops unscheduled buffered jobs before borrowed scope data
        // can expire.
        let job = unsafe { ScheduledJob::new_scoped(scoped_task) };
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
        let mut pending_jobs = jobs.into_iter();

        for _ in 0..chunk_count {
            let mut chunk = Vec::with_capacity(chunk_size);
            for _ in 0..chunk_size {
                if let Some(job) = pending_jobs.next() {
                    chunk.push(job);
                }
            }

            if chunk.is_empty() {
                break;
            }

            self.schedule_chunk(chunk)?;
        }

        Ok(())
    }

    fn schedule_single(&self, job: ScheduledJob) -> ExecutorResult<()> {
        let mut job = Some(job);
        let admitted = self
            .scheduler
            .admit_job::<C>(self.priority, self.locality_hint, &mut job);
        self.run_if_refused(admitted, job)
    }

    fn schedule_chunk(&self, jobs: Vec<ScheduledJob>) -> ExecutorResult<()> {
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
        self.schedule_single(job)
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
