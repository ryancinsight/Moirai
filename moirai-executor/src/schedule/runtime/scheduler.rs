//! `ThreadScheduler` and `SchedulerScope` impl blocks.

use std::{
    marker::PhantomData,
    mem,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr::NonNull,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
};

use moirai_core::{
    error::{ExecutorError, ExecutorResult},
    Priority,
};

use moirai_utils::cache::CacheAligned;

use super::super::{class::WorkClass, job::ScheduledJob, reduce::inline_reduction_limit};

use super::types::{
    get_current_worker_id, BoundedContendedWake, SchedulerInner, SchedulerScope,
    SchedulerScopeState, ScopedTaskCompletion, SharedScopedTaskCompletion, ThreadScheduler,
};

use super::worker::{
    indexed_chunk_count, indexed_reduce_chunk_count, inline_map_reduce, is_quiescent, lock_mutex,
    map_reduce_range, priority_weight, wake_all_workers, wake_contended_workers, wake_worker,
    JOIN_FAST_SPIN_ATTEMPTS,
};

#[cfg(feature = "scheduler-diagnostics")]
mod diagnostics;

impl ThreadScheduler<256, 256> {
    /// Start a scheduler with one worker set for all work classes.
    pub fn new(worker_count: usize, thread_name_prefix: &str) -> ExecutorResult<Self> {
        Self::new_with_config(worker_count, thread_name_prefix)
    }
}

impl<const QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>
    ThreadScheduler<QUEUE_CAPACITY, SPIN_LIMIT>
{
    /// Start a scheduler with custom configurations.
    pub fn new_with_config(worker_count: usize, thread_name_prefix: &str) -> ExecutorResult<Self> {
        let worker_count = worker_count.max(1);
        let workers = (0..worker_count)
            .map(|id| Arc::new(super::types::WorkerState::new(id)))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let inner = Arc::new(SchedulerInner {
            workers,
            handles: std::sync::Mutex::new(Vec::with_capacity(worker_count)),
            next_worker: CacheAligned::new(AtomicUsize::new(0)),
            pending_tasks: CacheAligned::new(AtomicUsize::new(0)),
            active_workers: CacheAligned::new(AtomicUsize::new(0)),
            completed_tasks: CacheAligned::new(std::sync::atomic::AtomicU64::new(0)),
            failed_tasks: CacheAligned::new(std::sync::atomic::AtomicU64::new(0)),
            shutdown: CacheAligned::new(std::sync::atomic::AtomicBool::new(false)),
            join_waiters: CacheAligned::new(AtomicUsize::new(0)),
            wait_lock: std::sync::Mutex::new(()),
            wait_signal: std::sync::Condvar::new(),
            idle_workers: CacheAligned::new(std::sync::atomic::AtomicU64::new(0)),
        });

        for worker_id in 0..worker_count {
            let worker_inner = Arc::clone(&inner);
            let thread_name = format!("{thread_name_prefix}-{worker_id}");
            let handle = thread::Builder::new()
                .name(thread_name)
                .spawn(move || {
                    super::worker::worker_loop::<QUEUE_CAPACITY, SPIN_LIMIT>(
                        worker_inner,
                        worker_id,
                    )
                })
                .map_err(|_| ExecutorError::ThreadPoolCreationFailed)?;

            lock_mutex(&inner.handles).push(handle);
        }

        Ok(Self { inner })
    }

    /// Schedule a job for a compile-time work class.
    pub fn schedule<C, F>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        task: F,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(usize) + Send + 'static,
    {
        let job = ScheduledJob::new(task);
        self.schedule_job::<C>(priority, locality_hint, job)
    }

    /// Run a borrowing job scope on the scheduler and wait for all spawned jobs.
    ///
    /// This is the scheduler-equivalent of a scoped fan-out. It avoids per-task
    /// result storage when the caller only needs completion, while preserving the
    /// invariant that borrowed data cannot outlive the scope.
    pub fn scope<'scope, C, F>(
        &'scope self,
        priority: Priority,
        locality_hint: Option<usize>,
        body: F,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(&SchedulerScope<'scope, C, QUEUE_CAPACITY, SPIN_LIMIT>) -> ExecutorResult<()>,
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
            _state: PhantomData,
            _class: PhantomData,
        };

        let body_result = catch_unwind(AssertUnwindSafe(|| body(&scope)));
        let flush_result = scope.flush();
        state.wait();

        match body_result {
            Ok(Ok(())) if state.failed_tasks.load(Ordering::Acquire) => Err(
                ExecutorError::SpawnFailed(moirai_core::error::TaskError::Panicked),
            ),
            Ok(Ok(())) => flush_result,
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Run an indexed scoped fan-out with worker-sized scheduler chunks.
    ///
    /// This path is for data-parallel work where the caller needs completion,
    /// not one task handle per logical item. It schedules at most one erased
    /// scheduler job per worker, while the item closure remains statically
    /// typed and shared by reference across chunks.
    pub fn for_each_indexed<C, F>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        count: usize,
        task: F,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: Fn(usize) + Send + Sync,
    {
        if self.inner.shutdown.load(Ordering::Acquire) {
            return Err(ExecutorError::ShuttingDown);
        }

        if count == 0 {
            return Ok(());
        }

        let chunk_count = indexed_chunk_count(count, self.worker_count());
        let chunk_size = count.div_ceil(chunk_count);
        let caller_end = chunk_size.min(count);
        if chunk_count == 1 {
            return catch_unwind(AssertUnwindSafe(|| {
                for index in 0..caller_end {
                    task(index);
                }
            }))
            .map_err(|_| ExecutorError::SpawnFailed(moirai_core::error::TaskError::Panicked));
        }

        let state = Arc::new(SchedulerScopeState::new());
        let task = &task;
        let mut schedule_result = Ok(());

        for chunk_index in 1..chunk_count {
            let start = chunk_index * chunk_size;
            let end = start.saturating_add(chunk_size).min(count);
            if start >= end {
                break;
            }

            state.register_task();
            let completion = SharedScopedTaskCompletion {
                state: Arc::clone(&state),
            };
            let scoped_job = move |_| {
                let completion = completion;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    for index in start..end {
                        task(index);
                    }
                }));

                if result.is_err() {
                    completion.mark_failed();
                }
            };

            if let Err(error) =
                self.schedule_scoped_job::<C, _>(priority, locality_hint, scoped_job)
            {
                state.complete_task();
                schedule_result = Err(error);
                break;
            }
        }

        let caller_result = if schedule_result.is_ok() {
            catch_unwind(AssertUnwindSafe(|| {
                for index in 0..caller_end {
                    task(index);
                }
            }))
            .map_err(|_| ExecutorError::SpawnFailed(moirai_core::error::TaskError::Panicked))
        } else {
            Ok(())
        };

        state.wait();

        if state.failed_tasks.load(Ordering::Acquire) || caller_result.is_err() {
            Err(ExecutorError::SpawnFailed(
                moirai_core::error::TaskError::Panicked,
            ))
        } else {
            schedule_result
        }
    }

    /// Run an indexed map/reduce with one result slot per physical chunk.
    ///
    /// `identity` must be the neutral element for `reduce`. The scheduler
    /// computes local chunk reductions before combining them on the caller's
    /// thread, avoiding per-item atomic aggregation.
    pub fn map_reduce_indexed<C, T, Map, Reduce>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        count: usize,
        identity: T,
        map: Map,
        reduce: Reduce,
    ) -> ExecutorResult<T>
    where
        C: WorkClass,
        T: Send + Clone,
        Map: Fn(usize) -> T + Send + Sync,
        Reduce: Fn(T, T) -> T + Send + Sync,
    {
        if self.inner.shutdown.load(Ordering::Acquire) {
            return Err(ExecutorError::ShuttingDown);
        }

        if count == 0 {
            return Ok(identity);
        }

        let worker_count = self.worker_count().max(1);
        if count <= inline_reduction_limit::<T>(worker_count) {
            return inline_map_reduce(count, identity, map, reduce);
        }

        let chunk_count = indexed_reduce_chunk_count::<T>(count, worker_count);
        let chunk_size = count.div_ceil(chunk_count);
        let caller_end = chunk_size.min(count);
        if chunk_count == 1 {
            return inline_map_reduce(count, identity, map, reduce);
        }

        let state = Arc::new(SchedulerScopeState::new());
        let slots = Arc::new(super::super::reduce::ReduceSlots::new(chunk_count - 1));
        let map = &map;
        let reduce = &reduce;
        let mut schedule_result = Ok(());

        for chunk_index in 1..chunk_count {
            let start = chunk_index * chunk_size;
            let end = start.saturating_add(chunk_size).min(count);
            if start >= end {
                break;
            }

            state.register_task();
            let completion = SharedScopedTaskCompletion {
                state: Arc::clone(&state),
            };
            let slots = Arc::clone(&slots);
            let identity = identity.clone();
            let scoped_job = move |_| {
                let completion = completion;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    let accumulator = map_reduce_range(start, end, identity, map, reduce);
                    slots.write(chunk_index - 1, accumulator);
                }));

                if result.is_err() {
                    completion.mark_failed();
                }
            };

            if let Err(error) =
                self.schedule_scoped_job::<C, _>(priority, locality_hint, scoped_job)
            {
                state.complete_task();
                schedule_result = Err(error);
                break;
            }
        }

        let caller_result = if schedule_result.is_ok() {
            catch_unwind(AssertUnwindSafe(|| {
                map_reduce_range(0, caller_end, identity.clone(), map, reduce)
            }))
            .map_err(|_| ExecutorError::SpawnFailed(moirai_core::error::TaskError::Panicked))
        } else {
            Ok(identity.clone())
        };

        state.wait();

        if state.failed_tasks.load(Ordering::Acquire) {
            Err(ExecutorError::SpawnFailed(
                moirai_core::error::TaskError::Panicked,
            ))
        } else {
            schedule_result?;
            Ok(slots.reduce(caller_result?, reduce))
        }
    }

    pub(super) fn schedule_job<C>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        job: ScheduledJob,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
    {
        if self.inner.shutdown.load(Ordering::Acquire) {
            return Err(ExecutorError::ShuttingDown);
        }

        let pending_before_submit = self.inner.pending_tasks.load(Ordering::Acquire);
        let active_before_submit = self.inner.active_workers.load(Ordering::Acquire);
        let worker_index = self.select_worker_for_state::<C>(
            priority,
            locality_hint,
            pending_before_submit,
            active_before_submit,
        );
        let previous_pending = self.inner.pending_tasks.fetch_add(1, Ordering::Release);

        let is_local = get_current_worker_id() == Some(worker_index);
        if is_local {
            if let Some(old_job) = self.inner.workers[worker_index].lifo_slot.push(job) {
                self.inner.workers[worker_index]
                    .queues
                    .push(priority, old_job);
            }
        } else {
            self.inner.workers[worker_index].queues.push(priority, job);
        }

        // Try to wake up an idle worker via the lock-free wake lottery
        let mut woken = false;
        let mut idle = self.inner.idle_workers.load(Ordering::SeqCst);
        while idle != 0 {
            let worker_to_wake = idle.trailing_zeros() as usize;
            if worker_to_wake >= self.inner.workers.len() {
                break;
            }
            let mask = 1 << worker_to_wake;
            match self.inner.idle_workers.compare_exchange_weak(
                idle,
                idle & !mask,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => {
                    wake_worker(&self.inner.workers[worker_to_wake]);
                    woken = true;
                    break;
                }
                Err(actual) => {
                    idle = actual;
                }
            }
        }

        if !woken {
            if previous_pending == 0 {
                wake_worker(&self.inner.workers[worker_index]);
            } else {
                let worker_count = self.inner.workers.len();
                if previous_pending < worker_count {
                    let _ = wake_contended_workers::<BoundedContendedWake>(
                        &*self.inner,
                        worker_index,
                        previous_pending,
                    );
                }
            }
        }
        Ok(())
    }

    pub(super) fn schedule_scoped_job<'scope, C, F>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        scoped_job: F,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(usize) + Send + 'scope,
    {
        // Safety: callers wait for their scope state counter to reach zero
        // before returning. Every scoped scheduler job owns a completion token
        // whose Drop decrements that counter on normal return, panic, or queued
        // drop. Scheduler shutdown drains queued work before workers exit.
        let job = unsafe { ScheduledJob::new_scoped(scoped_job) };
        self.schedule_job::<C>(priority, locality_hint, job)
    }

    /// Approximate number of queued jobs.
    pub fn pending_tasks(&self) -> usize {
        self.inner.pending_tasks.load(Ordering::Acquire)
    }

    /// Number of workers currently executing jobs.
    pub fn active_workers(&self) -> usize {
        self.inner.active_workers.load(Ordering::Acquire)
    }

    /// Returns true when queued or active work exists.
    pub fn has_work(&self) -> bool {
        !is_quiescent(&self.inner)
    }

    /// Wait until all queued and active work has completed without stopping workers.
    pub fn join(&self) -> ExecutorResult<()> {
        for _ in 0..JOIN_FAST_SPIN_ATTEMPTS {
            if is_quiescent(&self.inner) {
                return Ok(());
            }
            core::hint::spin_loop();
        }

        let mut guard = lock_mutex(&self.inner.wait_lock);
        self.inner.join_waiters.fetch_add(1, Ordering::AcqRel);
        while !is_quiescent(&self.inner) {
            guard = self
                .inner
                .wait_signal
                .wait(guard)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        self.inner.join_waiters.fetch_sub(1, Ordering::AcqRel);
        Ok(())
    }

    /// Number of worker threads.
    pub fn worker_count(&self) -> usize {
        self.inner.workers.len()
    }

    /// Capture scheduler metrics.
    pub fn metrics(&self) -> super::types::ScheduleMetrics {
        super::types::ScheduleMetrics {
            worker_count: self.worker_count(),
            pending_tasks: self.pending_tasks(),
            active_workers: self.active_workers(),
            completed_tasks: self.inner.completed_tasks.load(Ordering::Acquire),
            failed_tasks: self.inner.failed_tasks.load(Ordering::Acquire),
        }
    }

    /// Stop workers after queued work drains.
    pub fn shutdown(&self) {
        if !self.inner.shutdown.swap(true, Ordering::AcqRel) {
            wake_all_workers(&self.inner);
        }

        let mut handles = lock_mutex(&self.inner.handles);
        while let Some(handle) = handles.pop() {
            let _ = handle.join();
        }
    }

    #[cfg(test)]
    pub(super) fn select_worker<C>(&self, priority: Priority, locality_hint: Option<usize>) -> usize
    where
        C: WorkClass,
    {
        self.select_worker_for_state::<C>(
            priority,
            locality_hint,
            self.inner.pending_tasks.load(Ordering::Acquire),
            self.inner.active_workers.load(Ordering::Acquire),
        )
    }

    pub(super) fn select_worker_for_state<C>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        pending_tasks: usize,
        active_workers: usize,
    ) -> usize
    where
        C: WorkClass,
    {
        let worker_count = self.inner.workers.len();
        if let Some(hint) = locality_hint {
            return hint % worker_count;
        }

        if pending_tasks == 0 && active_workers <= 1 {
            return C::SERIAL_AFFINITY_OFFSET.wrapping_add(priority_weight(priority))
                % worker_count;
        }

        let ticket = self.inner.next_worker.fetch_add(1, Ordering::Relaxed);
        ticket
            .wrapping_add(C::AFFINITY_OFFSET)
            .wrapping_add(priority_weight(priority))
            % worker_count
    }
}

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
            let completion = completion;
            let result = catch_unwind(AssertUnwindSafe(|| task(worker_id)));
            if result.is_err() {
                completion.mark_failed();
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
        self.scheduler
            .schedule_job::<C>(self.priority, self.locality_hint, job)?;
        Ok(())
    }

    fn schedule_chunk(&self, jobs: Vec<ScheduledJob>) -> ExecutorResult<()> {
        let scoped_job = move |worker_id| {
            for job in jobs {
                let _ = job.execute(worker_id);
            }
        };

        self.scheduler.schedule_scoped_job::<C, _>(
            self.priority,
            self.locality_hint,
            scoped_job,
        )?;
        Ok(())
    }

    fn state(&self) -> &SchedulerScopeState {
        // Safety: `ThreadScheduler::scope` creates this pointer from a local
        // state value and waits for every scheduled scoped job before returning.
        unsafe { self.state.as_ref() }
    }
}

impl<const QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize> Drop
    for ThreadScheduler<QUEUE_CAPACITY, SPIN_LIMIT>
{
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) == 1 {
            self.shutdown();
        }
    }
}
