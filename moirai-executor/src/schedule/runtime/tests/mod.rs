//! Unit tests for the thread scheduler runtime.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{
    Arc, Barrier, Condvar, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc,
};
use std::time::Duration;

use super::types::{ThreadScheduler, get_current_worker_id};
use crate::schedule::{AsyncTask, BlockingTask, SyncTask};
use moirai_core::{
    Priority,
    error::{ExecutorError, TaskError},
    executor::{ExecutorConfig, config::DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY},
};

const TEST_ADMISSION_CAPACITY: usize = 8;
const TEST_EVENT_DEADLINE: Duration = Duration::from_secs(5);

struct DropProbe {
    drops: Arc<AtomicUsize>,
}

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

struct DropSignal {
    state: Arc<(Mutex<bool>, Condvar)>,
}

impl Drop for DropSignal {
    fn drop(&mut self) {
        let (lock, signal) = &*self.state;
        let mut dropped = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        *dropped = true;
        signal.notify_all();
    }
}

fn scheduler_with_queue_config<const BLOCKING_QUEUE_CAPACITY: usize>(
    worker_count: usize,
    name: &str,
    max_global_queue_size: usize,
    local_queue_initial_capacity: usize,
) -> Result<ThreadScheduler<BLOCKING_QUEUE_CAPACITY>, ExecutorError> {
    ThreadScheduler::<BLOCKING_QUEUE_CAPACITY>::from_executor_config(&ExecutorConfig {
        worker_threads: worker_count,
        max_global_queue_size,
        local_queue_initial_capacity,
        thread_name_prefix: name.into(),
        ..ExecutorConfig::default()
    })
}

fn scheduler_with_bounded_admission(name: &str) -> ThreadScheduler<256> {
    scheduler_with_queue_config::<256>(
        1,
        name,
        TEST_ADMISSION_CAPACITY,
        DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY,
    )
    .unwrap()
}

fn occupy_compute_worker(
    scheduler: &ThreadScheduler,
    locality_hint: usize,
) -> (usize, mpsc::SyncSender<()>) {
    let (started_sender, started_receiver) = mpsc::sync_channel(0);
    let (release_sender, release_receiver) = mpsc::sync_channel(0);
    scheduler
        .schedule::<SyncTask, _>(Priority::Critical, Some(locality_hint), move |worker_id| {
            started_sender
                .send(worker_id)
                .expect("test observer remains connected");
            release_receiver
                .recv()
                .expect("test controller releases the occupied worker");
        })
        .expect("gate job must be admitted");
    let worker_id = started_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("gate job must start before the test deadline");
    (worker_id, release_sender)
}

mod admission;
mod capacity;
mod indexed;
mod lifecycle;
mod placement;
mod scope;
mod shutdown;
