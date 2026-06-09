use std::sync::atomic::{AtomicUsize, Ordering};

use moirai_core::Priority;

use crate::schedule::{ThreadScheduler, WorkClass};
use crate::schedule::job::ScheduledJob;
use crate::schedule::queue::WorkerQueues;
use crate::schedule::runtime::worker::{
    execute_job, next_job, diagnostic_publish_work_available, wake_worker,
    JOIN_FAST_SPIN_ATTEMPTS, is_quiescent,
};
use crate::schedule::runtime::types::DiagnosticWakeDecision;

impl<const QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>
    ThreadScheduler<QUEUE_CAPACITY, SPIN_LIMIT>
{
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

    pub fn diagnostic_pending_counter_pair(&self) -> usize {
        let previous = self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        self.inner.pending_tasks.fetch_sub(1, Ordering::Release);
        previous
    }

    pub fn diagnostic_worker_unpark(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        wake_worker(&self.inner.workers[index]);
        index
    }

    pub fn diagnostic_priority_queue_push_pop(priority: Priority) -> usize {
        let queues = WorkerQueues::<QUEUE_CAPACITY>::new();
        queues.push(priority, ScheduledJob::new(|_| {}));
        queues
            .pop_local()
            .map(|job| usize::from(job.execute(0)))
            .unwrap_or(0)
    }

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
        let queues = WorkerQueues::<QUEUE_CAPACITY>::new();
        queues.push(priority, ScheduledJob::new(|_| {}));
        let completed = queues
            .pop_local()
            .map(|job| usize::from(job.execute(worker_index)))
            .unwrap_or(0);
        pending_tasks.fetch_sub(1, Ordering::Release);

        worker_index + previous_pending + completed
    }

    pub fn diagnostic_worker_execute_ready_job(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        execute_job(&self.inner, index, ScheduledJob::new(|_| {}));
        index
    }

    pub fn diagnostic_worker_local_dequeue_execute(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        self.inner.workers[index]
            .queues
            .push(Priority::Normal, ScheduledJob::new(|_| {}));

        next_job(&self.inner, index)
            .map(|job| {
                execute_job(&self.inner, index, job);
                index + 1
            })
            .unwrap_or(0)
    }

    pub fn diagnostic_max_inline_job_construct_drop() -> usize {
        let words = [1usize; 14];
        let job = ScheduledJob::new(move |_| {
            std::hint::black_box(words.iter().copied().sum::<usize>());
        });
        drop(std::hint::black_box(job));
        words.len()
    }

    pub fn diagnostic_max_inline_job_construct_execute() -> usize {
        let words = [1usize; 14];
        let job = ScheduledJob::new(move |_| {
            std::hint::black_box(words.iter().copied().sum::<usize>());
        });
        usize::from(std::hint::black_box(job).execute(0))
    }

    pub fn diagnostic_oversized_job_construct_drop() -> usize {
        let words = [1usize; 32];
        let job = ScheduledJob::new(move |_| {
            std::hint::black_box(words.iter().copied().sum::<usize>());
        });
        drop(std::hint::black_box(job));
        words.len()
    }

    pub fn diagnostic_oversized_job_construct_execute() -> usize {
        let words = [1usize; 32];
        let job = ScheduledJob::new(move |_| {
            std::hint::black_box(words.iter().copied().sum::<usize>());
        });
        usize::from(std::hint::black_box(job).execute(0))
    }

    pub fn diagnostic_max_inline_queue_push_pop_execute() -> usize {
        let words = [1usize; 14];
        let queues = WorkerQueues::<QUEUE_CAPACITY>::new();
        queues.push(
            Priority::Normal,
            ScheduledJob::new(move |_| {
                std::hint::black_box(words.iter().copied().sum::<usize>());
            }),
        );

        queues
            .pop_local()
            .map(|job| usize::from(job.execute(0)))
            .unwrap_or(0)
    }

    pub fn diagnostic_oversized_queue_push_pop_execute() -> usize {
        let words = [1usize; 32];
        let queues = WorkerQueues::<QUEUE_CAPACITY>::new();
        queues.push(
            Priority::Normal,
            ScheduledJob::new(move |_| {
                std::hint::black_box(words.iter().copied().sum::<usize>());
            }),
        );

        queues
            .pop_local()
            .map(|job| usize::from(job.execute(0)))
            .unwrap_or(0)
    }

    pub fn diagnostic_worker_local_max_inline_dequeue_execute(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        let words = [1usize; 14];
        self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        self.inner.workers[index].queues.push(
            Priority::Normal,
            ScheduledJob::new(move |_| {
                std::hint::black_box(words.iter().copied().sum::<usize>());
            }),
        );

        next_job(&self.inner, index)
            .map(|job| {
                execute_job(&self.inner, index, job);
                index + 1
            })
            .unwrap_or(0)
    }

    pub fn diagnostic_worker_local_oversized_dequeue_execute(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        let words = [1usize; 32];
        self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        self.inner.workers[index].queues.push(
            Priority::Normal,
            ScheduledJob::new(move |_| {
                std::hint::black_box(words.iter().copied().sum::<usize>());
            }),
        );

        next_job(&self.inner, index)
            .map(|job| {
                execute_job(&self.inner, index, job);
                index + 1
            })
            .unwrap_or(0)
    }

    pub fn diagnostic_join_fast_spin_quiescent(&self) -> usize {
        for attempt in 0..JOIN_FAST_SPIN_ATTEMPTS {
            if is_quiescent(&self.inner) {
                return attempt + 1;
            }
            core::hint::spin_loop();
        }
        0
    }

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
