//! Scheduler shutdown and external-handle lifecycle.

use std::{sync::atomic::Ordering, thread};

use super::super::{
    types::{SchedulerInner, ThreadScheduler},
    worker::{join_other_threads, lock_mutex, wake_all_workers},
};

const JOIN_OPEN: u8 = 0;
const JOIN_IN_PROGRESS: u8 = 1;
const JOIN_COMPLETE: u8 = 2;

#[cfg(test)]
struct ShutdownWaitRegistration<'a>(&'a std::sync::atomic::AtomicUsize);

#[cfg(test)]
impl Drop for ShutdownWaitRegistration<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}

impl<const BLOCKING_QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>
    ThreadScheduler<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>
{
    /// Stop workers after queued work drains.
    ///
    /// Exactly one concurrent caller owns worker joining. Other external
    /// callers wait for completion. Scheduler workers close the blocking lane
    /// and return before the election so only a non-worker caller can join
    /// peers. When the final external handle is dropped by a worker, every
    /// worker drains and releases its scheduler ownership after its current
    /// task returns.
    pub fn shutdown(&self) {
        close_admission(&self.inner);

        if current_thread_belongs_to(&self.inner) {
            return;
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
            self.inner
                .shutdown_joiner
                .set(thread::current().id())
                .expect("invariant: shutdown join owner is elected once");
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
        } else {
            if self
                .inner
                .shutdown_joiner
                .get()
                .is_some_and(|joiner| *joiner == thread::current().id())
            {
                return;
            }

            let mut guard = lock_mutex(&self.inner.wait_lock);
            #[cfg(test)]
            let _registration =
                if self.inner.shutdown_join_state.load(Ordering::Acquire) != JOIN_COMPLETE {
                    self.inner
                        .shutdown_wait_registration
                        .get()
                        .map(|(signal, count)| {
                            count.fetch_add(1, Ordering::Release);
                            signal.send(()).expect("test observer remains connected");
                            ShutdownWaitRegistration(count)
                        })
                } else {
                    None
                };
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
            lane.join();
        }

        let mut handles = std::mem::take(&mut *lock_mutex(&self.inner.handles));
        join_other_threads(&mut handles);
    }

    #[cfg(test)]
    pub(crate) fn current_thread_is_worker(&self) -> bool {
        current_thread_belongs_to(&self.inner)
    }
}

/// Close every scheduler admission path without joining workers.
///
/// This is the shared first half of explicit shutdown and worker-failure
/// drainage. It wakes parked compute workers and closes blocking queues so all
/// work accepted before the publication drains before their owners exit.
pub(in crate::schedule::runtime) fn close_admission<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
) {
    // SeqCst: the producer half of the parking handshake (wait.rs pairs this
    // store with its `pending_tasks` recheck), so admission closure cannot hide
    // behind a worker that parks between the two observations.
    let published = !inner.shutdown.swap(true, Ordering::SeqCst);
    if published {
        wake_all_workers(inner);
    }
    {
        let _lane_init = lock_mutex(&inner.blocking_lane_init);
        if let Some(lane) = inner.blocking_lane.get() {
            lane.close();
        }
    }
    #[cfg(test)]
    if published && let Some(barrier) = inner.shutdown_started_barrier.get() {
        barrier.wait();
    }
}

impl<const BLOCKING_QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize> Drop
    for ThreadScheduler<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>
{
    fn drop(&mut self) {
        // Acquire/release orders prior external-handle activity before the
        // final owner initiates shutdown. An external owner joins synchronously;
        // a worker owner returns before the join election so peers cannot form
        // a dependency cycle. Queue publication and worker wakeup retain their
        // own stronger synchronization boundaries.
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
