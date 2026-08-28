//! Indexed data-parallel fan-out for `ThreadScheduler`.
//!
//! Implements the [`DataParallel`](crate::schedule::DataParallel) seam role:
//! `for_each_indexed` and `map_reduce_indexed` schedule at most one erased job
//! per worker (not one per item), keeping the item closure statically typed and
//! shared by reference across chunks. Separated from core scheduling
//! (submission, scope, lifecycle) per the single-responsibility principle — this
//! module owns the indexed data-parallel concern.
//!
//! ## Queue-full resilience
//!
//! When the per-worker admission queue is full (`ResourceExhausted`), a rejected
//! job is dropped by the scheduler before returning the error. Dropping the job
//! fires its borrowing `ScopedTaskCompletion` token, correctly decrementing
//! the stack-owned scope's pending-task counter. The *work* inside the job was
//! never executed, however. Both `for_each_indexed` and `map_reduce_indexed`
//! detect this case and execute the rejected chunk inline on the caller thread
//! before continuing the scheduling loop. Inline execution uses the same panic
//! boundary as worker execution, and [`ThreadScheduler::admission_caller_runs`]
//! exposes each backpressure event. The result is identical to parallel
//! execution (every item is visited exactly once) without per-call completion
//! state allocation.

use std::{
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
    sync::atomic::Ordering,
};

use moirai_core::{
    error::{ExecutorError, ExecutorResult},
    Priority,
};

use super::super::super::{class::WorkClass, reduce::ReduceSlots};
use super::super::types::{
    get_current_worker_id, is_in_indexed_region, IndexedRegionGuard, SchedulerScopeState,
    ScopedTaskCompletion, ThreadScheduler,
};
use super::super::worker::{
    indexed_chunk_bounds, indexed_chunk_count, inline_map_reduce, map_reduce_range,
};

fn execute_catching_panic<T>(operation: impl FnOnce() -> T) -> ExecutorResult<T> {
    catch_unwind(AssertUnwindSafe(operation))
        .map_err(|_| ExecutorError::SpawnFailed(moirai_core::error::TaskError::Panicked))
}

impl<const BLOCKING_QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>
    ThreadScheduler<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>
{
    /// Run an indexed scoped fan-out with worker-sized scheduler chunks.
    ///
    /// This path is for data-parallel work where the caller needs completion,
    /// not one task handle per logical item. It schedules at most one erased
    /// scheduler job per worker, while the item closure remains statically
    /// typed and shared by reference across chunks. A scheduler worker calling
    /// this method participates in queued work while waiting, so nested indexed
    /// fan-out remains work-conserving under pool saturation.
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

        // A worker already contributes one lane of outer parallelism. Flatten
        // nested indexed regions on that lane: recursively stealing unrelated
        // outer jobs grows the worker stack with every nested tensor reduction.
        if get_current_worker_id().is_some() || is_in_indexed_region() {
            return execute_catching_panic(|| {
                for index in 0..count {
                    task(index);
                }
            });
        }

        let chunk_count = indexed_chunk_count(count, self.worker_count());
        let (_, caller_end) = indexed_chunk_bounds(count, chunk_count, 0);
        if chunk_count == 1 {
            return execute_catching_panic(|| {
                let _region = IndexedRegionGuard::enter();
                for index in 0..caller_end {
                    task(index);
                }
            });
        }

        let state = SchedulerScopeState::new();
        let task = &task;
        let mut schedule_result = Ok(());
        let mut inline_result = Ok(());

        for chunk_index in 1..chunk_count {
            let (start, end) = indexed_chunk_bounds(count, chunk_count, chunk_index);

            state.register_task();
            let completion = ScopedTaskCompletion::new(&state);
            let chunk_task = move |_| {
                for index in start..end {
                    task(index);
                }
            };
            let complete = move |succeeded: bool| {
                if !succeeded {
                    completion.mark_failed();
                }
            };

            if let Err(error) = self.schedule_indexed_job::<C, _, _>(
                &state,
                priority,
                locality_hint,
                chunk_task,
                complete,
            ) {
                match error {
                    // Admission queue was full. The scheduler dropped the job,
                    // which fired the completion token (scope counter correct).
                    // Run the work inline so every item is visited exactly once.
                    ExecutorError::ResourceExhausted(_) => {
                        self.record_admission_caller_run();
                        inline_result = execute_catching_panic(|| {
                            for index in start..end {
                                task(index);
                            }
                        });
                        if inline_result.is_err() {
                            break;
                        }
                    }
                    other => {
                        schedule_result = Err(other);
                        break;
                    }
                }
            }
        }

        let caller_result = if schedule_result.is_ok() && inline_result.is_ok() {
            execute_catching_panic(|| {
                let _region = IndexedRegionGuard::enter();
                for index in 0..caller_end {
                    task(index);
                }
            })
        } else {
            Ok(())
        };

        self.drain_scope(&state);

        if state.failed_tasks.load(Ordering::Acquire)
            || inline_result.is_err()
            || caller_result.is_err()
        {
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
    /// thread, avoiding per-item atomic aggregation. A scheduler worker calling
    /// this method participates in queued work while waiting, so nested indexed
    /// reductions remain work-conserving under pool saturation.
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

        if get_current_worker_id().is_some() || is_in_indexed_region() {
            return inline_map_reduce(count, identity, map, reduce);
        }

        let chunk_count = indexed_chunk_count(count, self.worker_count());
        let (_, caller_end) = indexed_chunk_bounds(count, chunk_count, 0);
        if chunk_count == 1 {
            let _region = IndexedRegionGuard::enter();
            return inline_map_reduce(count, identity, map, reduce);
        }

        let state = SchedulerScopeState::new();
        let slots = ReduceSlots::new(chunk_count - 1);
        let map = &map;
        let reduce = &reduce;
        let mut schedule_result = Ok(());
        let mut inline_result = Ok(());

        for chunk_index in 1..chunk_count {
            let (start, end) = indexed_chunk_bounds(count, chunk_count, chunk_index);

            let identity_chunk = match execute_catching_panic(|| identity.clone()) {
                Ok(identity) => identity,
                Err(error) => {
                    inline_result = Err(error);
                    break;
                }
            };
            state.register_task();
            let completion = ScopedTaskCompletion::new(&state);
            // Use distinct names so the outer `slots`/`identity` remain accessible
            // in the ResourceExhausted inline fallback below.
            let slots_chunk = &slots;
            let chunk_task = move |_| {
                let accumulator = map_reduce_range(start, end, identity_chunk, map, reduce);
                slots_chunk.write(chunk_index - 1, accumulator);
            };
            let complete = move |succeeded: bool| {
                if !succeeded {
                    completion.mark_failed();
                }
            };

            if let Err(error) = self.schedule_indexed_job::<C, _, _>(
                &state,
                priority,
                locality_hint,
                chunk_task,
                complete,
            ) {
                match error {
                    // Admission queue was full. The scheduler dropped the job,
                    // which fired the completion token (scope counter correct).
                    // Run the reduction inline and write the result slot so the
                    // final combine step sees every chunk's contribution.
                    ExecutorError::ResourceExhausted(_) => {
                        self.record_admission_caller_run();
                        inline_result = execute_catching_panic(|| {
                            let accumulator =
                                map_reduce_range(start, end, identity.clone(), map, reduce);
                            slots.write(chunk_index - 1, accumulator);
                        });
                        if inline_result.is_err() {
                            break;
                        }
                    }
                    other => {
                        schedule_result = Err(other);
                        break;
                    }
                }
            }
        }

        let caller_result = (schedule_result.is_ok() && inline_result.is_ok()).then(|| {
            execute_catching_panic(|| {
                let _region = IndexedRegionGuard::enter();
                map_reduce_range(0, caller_end, identity.clone(), map, reduce)
            })
        });

        self.drain_scope(&state);

        if state.failed_tasks.load(Ordering::Acquire) || inline_result.is_err() {
            Err(ExecutorError::SpawnFailed(
                moirai_core::error::TaskError::Panicked,
            ))
        } else {
            schedule_result?;
            let caller_result = caller_result
                .expect("invariant: caller reduction runs after successful scheduling")?;
            Ok(slots.reduce(caller_result, reduce))
        }
    }

    fn schedule_indexed_job<'scope, C, F, Complete>(
        &self,
        state: &'scope SchedulerScopeState,
        priority: Priority,
        locality_hint: Option<usize>,
        scoped_job: F,
        complete: Complete,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(usize) + Send + 'scope,
        Complete: FnOnce(bool) + Send + 'scope,
    {
        // Scoped job storage erases `'scope`; a scheduling unwind after an
        // earlier admission must not release the borrowed stack state first.
        match catch_unwind(AssertUnwindSafe(|| {
            self.schedule_scoped_job::<C, _, _>(priority, locality_hint, scoped_job, complete)
        })) {
            Ok(result) => result,
            Err(payload) => {
                self.drain_scope(state);
                resume_unwind(payload);
            }
        }
    }
}
