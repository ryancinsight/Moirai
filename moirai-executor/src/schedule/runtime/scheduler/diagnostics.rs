//! Scheduler diagnostic probes used by provider conformance tests.
//!
//! These helpers intentionally expose small, value-returning probes rather than
//! scheduler internals. They let the benchmark/diagnostic contract exercise
//! queue publication, worker wakeup, job representation, and join behavior
//! without adding a second scheduler implementation.

use std::sync::atomic::{AtomicUsize, Ordering};

use moirai_core::Priority;

use crate::schedule::job::ScheduledJob;
use crate::schedule::queue::WorkerQueues;
use crate::schedule::runtime::types::DiagnosticWakeDecision;
use crate::schedule::runtime::worker::{
    diagnostic_publish_work_available, execute_job, is_quiescent, next_shared_job, wake_worker,
    JOIN_FAST_SPIN_ATTEMPTS,
};
use crate::schedule::{ThreadScheduler, WorkClass};

impl<const QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>
    ThreadScheduler<QUEUE_CAPACITY, SPIN_LIMIT>
{
    /// Probe worker selection for a diagnostic scheduler state.
    pub fn diagnostic_select_worker_for_state<C>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        pending_tasks: usize,
        active_workers: usize,
    ) -> usize
    where
        C: WorkClass,
    {
        self.select_worker_for_state::<C>(priority, locality_hint, pending_tasks, active_workers)
    }

    /// Increment and decrement the pending counter, returning its prior value.
    pub fn diagnostic_pending_counter_pair(&self) -> usize {
        let previous = self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        self.inner.pending_tasks.fetch_sub(1, Ordering::Release);
        previous
    }

    /// Wake the normalized worker index and return that index.
    pub fn diagnostic_worker_unpark(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        wake_worker(&self.inner.workers[index]);
        index
    }

    /// Push and execute one diagnostic priority-queue job.
    pub fn diagnostic_priority_queue_push_pop(priority: Priority) -> usize {
        let (mut owner, queues) = WorkerQueues::<QUEUE_CAPACITY>::new();
        let () = queues
            .try_push_external(priority, ScheduledJob::new(|_| {}))
            .map_or((), |_| panic!("diagnostic queue has capacity"));
        owner
            .pop_local()
            .map(|job| usize::from(job.execute(0)))
            .unwrap_or(0)
    }

    /// Probe submission publication and return the combined diagnostic result.
    pub fn diagnostic_submission_queue_publication<C>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
    ) -> usize
    where
        C: WorkClass,
    {
        let pending_tasks = AtomicUsize::new(0);
        let active_workers = AtomicUsize::new(0);
        let pending_before_submit = pending_tasks.load(Ordering::Acquire);
        let active_before_submit = active_workers.load(Ordering::Acquire);
        let worker_index = self.select_worker_for_state::<C>(
            priority,
            locality_hint,
            pending_before_submit,
            active_before_submit,
        );
        let previous_pending = pending_tasks.fetch_add(1, Ordering::Release);
        let (mut owner, queues) = WorkerQueues::<QUEUE_CAPACITY>::new();
        let () = queues
            .try_push_external(priority, ScheduledJob::new(|_| {}))
            .map_or((), |_| panic!("diagnostic queue has capacity"));
        let completed = owner
            .pop_local()
            .map(|job| usize::from(job.execute(worker_index)))
            .unwrap_or(0);
        pending_tasks.fetch_sub(1, Ordering::Release);

        worker_index + previous_pending + completed
    }

    /// Execute one ready job on a normalized worker index.
    pub fn diagnostic_worker_execute_ready_job(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        execute_job(&self.inner, index, ScheduledJob::new(|_| {}));
        index
    }

    /// Queue and execute one job through a worker's shared-dequeue path.
    pub fn diagnostic_worker_local_dequeue_execute(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        let () = self.inner.workers[index]
            .queues
            .try_push_external(Priority::Normal, ScheduledJob::new(|_| {}))
            .map_or((), |_| panic!("diagnostic queue has capacity"));

        next_shared_job(&self.inner, index)
            .map(|job| {
                execute_job(&self.inner, index, job);
                index + 1
            })
            .unwrap_or(0)
    }

    /// Construct and drop a maximum inline-sized job, returning its word count.
    pub fn diagnostic_max_inline_job_construct_drop() -> usize {
        let words = [1usize; 14];
        let job = ScheduledJob::new(move |_| {
            std::hint::black_box(words.iter().copied().sum::<usize>());
        });
        drop(std::hint::black_box(job));
        words.len()
    }

    /// Construct and execute a maximum inline-sized job.
    pub fn diagnostic_max_inline_job_construct_execute() -> usize {
        let words = [1usize; 14];
        let job = ScheduledJob::new(move |_| {
            std::hint::black_box(words.iter().copied().sum::<usize>());
        });
        usize::from(std::hint::black_box(job).execute(0))
    }

    /// Construct and drop an oversized job, returning its word count.
    pub fn diagnostic_oversized_job_construct_drop() -> usize {
        let words = [1usize; 32];
        let job = ScheduledJob::new(move |_| {
            std::hint::black_box(words.iter().copied().sum::<usize>());
        });
        drop(std::hint::black_box(job));
        words.len()
    }

    /// Construct and execute an oversized job.
    pub fn diagnostic_oversized_job_construct_execute() -> usize {
        let words = [1usize; 32];
        let job = ScheduledJob::new(move |_| {
            std::hint::black_box(words.iter().copied().sum::<usize>());
        });
        usize::from(std::hint::black_box(job).execute(0))
    }

    /// Push and execute a maximum inline-sized queue job.
    pub fn diagnostic_max_inline_queue_push_pop_execute() -> usize {
        let words = [1usize; 14];
        let (mut owner, queues) = WorkerQueues::<QUEUE_CAPACITY>::new();
        let () = queues
            .try_push_external(
                Priority::Normal,
                ScheduledJob::new(move |_| {
                    std::hint::black_box(words.iter().copied().sum::<usize>());
                }),
            )
            .map_or((), |_| panic!("diagnostic queue has capacity"));

        owner
            .pop_local()
            .map(|job| usize::from(job.execute(0)))
            .unwrap_or(0)
    }

    /// Push and execute an oversized queue job.
    pub fn diagnostic_oversized_queue_push_pop_execute() -> usize {
        let words = [1usize; 32];
        let (mut owner, queues) = WorkerQueues::<QUEUE_CAPACITY>::new();
        let () = queues
            .try_push_external(
                Priority::Normal,
                ScheduledJob::new(move |_| {
                    std::hint::black_box(words.iter().copied().sum::<usize>());
                }),
            )
            .map_or((), |_| panic!("diagnostic queue has capacity"));

        owner
            .pop_local()
            .map(|job| usize::from(job.execute(0)))
            .unwrap_or(0)
    }

    /// Queue and execute a maximum inline-sized job on a local worker dequeue.
    pub fn diagnostic_worker_local_max_inline_dequeue_execute(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        let words = [1usize; 14];
        self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        let () = self.inner.workers[index]
            .queues
            .try_push_external(
                Priority::Normal,
                ScheduledJob::new(move |_| {
                    std::hint::black_box(words.iter().copied().sum::<usize>());
                }),
            )
            .map_or((), |_| panic!("diagnostic queue has capacity"));

        next_shared_job(&self.inner, index)
            .map(|job| {
                execute_job(&self.inner, index, job);
                index + 1
            })
            .unwrap_or(0)
    }

    /// Queue and execute an oversized job on a local worker dequeue.
    pub fn diagnostic_worker_local_oversized_dequeue_execute(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        let words = [1usize; 32];
        self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        let () = self.inner.workers[index]
            .queues
            .try_push_external(
                Priority::Normal,
                ScheduledJob::new(move |_| {
                    std::hint::black_box(words.iter().copied().sum::<usize>());
                }),
            )
            .map_or((), |_| panic!("diagnostic queue has capacity"));

        next_shared_job(&self.inner, index)
            .map(|job| {
                execute_job(&self.inner, index, job);
                index + 1
            })
            .unwrap_or(0)
    }

    /// Probe the fast join spin on a quiescent scheduler.
    pub fn diagnostic_join_fast_spin_quiescent(&self) -> usize {
        for attempt in 0..JOIN_FAST_SPIN_ATTEMPTS {
            if is_quiescent(&self.inner) {
                return attempt + 1;
            }
            core::hint::spin_loop();
        }
        0
    }

    /// Probe the fast join spin while one task remains pending.
    pub fn diagnostic_join_fast_spin_pending(&self) -> usize {
        self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        let mut misses = 0usize;
        for _ in 0..JOIN_FAST_SPIN_ATTEMPTS {
            if !is_quiescent(&self.inner) {
                misses = misses.wrapping_add(1);
            }
            core::hint::spin_loop();
        }
        self.inner.pending_tasks.fetch_sub(1, Ordering::Release);
        misses
    }

    /// Probe a wake decision policy for a normalized worker index.
    pub fn diagnostic_wake_decision<P>(&self, worker_index: usize) -> usize
    where
        P: DiagnosticWakeDecision,
    {
        let worker_count = self.inner.workers.len();
        let index = worker_index % worker_count;
        diagnostic_publish_work_available(
            self.inner.as_ref(),
            index,
            P::previous_pending(worker_count),
        )
    }
}
