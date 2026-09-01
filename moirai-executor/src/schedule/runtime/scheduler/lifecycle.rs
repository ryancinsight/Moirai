//! Scheduler shutdown and external-handle lifecycle.

use std::{sync::atomic::Ordering, thread};

use super::super::{
    types::{SchedulerInner, ThreadScheduler},
    worker::{join_other_threads, lock_mutex, wake_all_workers},
};

const JOIN_OPEN: u8 = 0;
const JOIN_IN_PROGRESS: u8 = 1;
const JOIN_COMPLETE: u8 = 2;

impl<const BLOCKING_QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>
    ThreadScheduler<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>
{
    /// Stop workers after queued work drains.
    ///
    /// Exactly one concurrent caller owns worker joining. Other external
    /// callers wait for completion. A scheduler worker that loses the election
    /// returns so the elected caller can join it; that worker exits after its
    /// current task returns.
    pub fn shutdown(&self) {
        if !self.inner.shutdown.swap(true, Ordering::SeqCst) {
            wake_all_workers(&self.inner);
        }

        #[cfg(test)]
        if let Some(barrier) = self.inner.shutdown_started_barrier.get() {
            barrier.wait();
        }

        if self
            .inner
            .shutdown_join_state
            .compare_exchange(
                JOIN_OPEN,
                JOIN_IN_PROGRESS,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.join_worker_sets();
            {
                // The completion predicate and waiter transition share this
                // mutex. Publishing while holding it prevents a waiter from
                // observing `JOIN_IN_PROGRESS` and sleeping after the sole
                // notification has already passed.
                let _completion_guard = lock_mutex(&self.inner.wait_lock);
                self.inner
                    .shutdown_join_state
                    .store(JOIN_COMPLETE, Ordering::Release);
            }
            self.inner.wait_signal.notify_all();
        } else if !current_thread_belongs_to(&self.inner) {
            let mut guard = lock_mutex(&self.inner.wait_lock);
            while self.inner.shutdown_join_state.load(Ordering::Acquire) != JOIN_COMPLETE {
                guard = self
                    .inner
                    .wait_signal
                    .wait(guard)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        }
    }

    fn join_worker_sets(&self) {
        let blocking_lane = {
            let _lane_init = lock_mutex(&self.inner.blocking_lane_init);
            self.inner.blocking_lane.get()
        };
        if let Some(lane) = blocking_lane {
            lane.shutdown();
        }

        let mut handles = std::mem::take(&mut *lock_mutex(&self.inner.handles));
        join_other_threads(&mut handles);
    }
}

impl<const BLOCKING_QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize> Drop
    for ThreadScheduler<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>
{
    fn drop(&mut self) {
        // Acquire/release orders prior external-handle activity before the
        // final owner initiates synchronous shutdown. Queue publication and
        // worker wakeup retain their own stronger synchronization boundaries.
        let previous = self.inner.external_handles.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "scheduler handle count must not underflow");
        if previous == 1 {
            self.shutdown();
        }
    }
}

fn current_thread_belongs_to<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
) -> bool {
    let current = thread::current().id();
    inner.workers.iter().any(|worker| {
        worker
            .thread
            .get()
            .is_some_and(|registered| registered.id() == current)
    }) || inner
        .blocking_lane
        .get()
        .is_some_and(super::super::blocking::BlockingLane::is_current_worker)
}
