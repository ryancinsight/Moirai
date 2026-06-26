//! Indexed data-parallel fan-out for `ThreadScheduler`.
//!
//! Implements the [`DataParallel`](crate::schedule::DataParallel) seam role:
//! `for_each_indexed` and `map_reduce_indexed` schedule at most one erased job
//! per worker (not one per item), keeping the item closure statically typed and
//! shared by reference across chunks. Separated from core scheduling
//! (submission, scope, lifecycle) per the single-responsibility principle — this
//! module owns the indexed data-parallel concern.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{atomic::Ordering, Arc},
};

use moirai_core::{
    error::{ExecutorError, ExecutorResult},
    Priority,
};

use super::super::super::{
    class::WorkClass,
    reduce::{inline_reduction_limit, ReduceSlots},
};
use super::super::types::{SchedulerScopeState, SharedScopedTaskCompletion, ThreadScheduler};
use super::super::worker::{
    indexed_chunk_count, indexed_reduce_chunk_count, inline_map_reduce, map_reduce_range,
};

impl<const QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>
    ThreadScheduler<QUEUE_CAPACITY, SPIN_LIMIT>
{
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
        let slots = Arc::new(ReduceSlots::new(chunk_count - 1));
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
}
