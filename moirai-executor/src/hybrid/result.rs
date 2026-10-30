//! Task-result publication and deferred lifecycle notification.

use crate::{metrics::ExecutorMetrics, registry::TaskCompletion};
use moirai_core::{error::TaskError, task::TaskResultSender};
use std::sync::atomic::Ordering;
use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
};

// Runtime panic payloads have compile-time-unknown types at this cold boundary.
pub(super) type PanicPayload = Box<dyn Any + Send>;

pub(super) enum TaskEnd<T> {
    Completed(std::thread::Result<T>),
    Cancelled,
}

/// Dispose an unrun closure before a notification can start unwinding.
pub(super) fn cancel_task_result<F, T>(
    func: F,
    sender: Option<TaskResultSender<T>>,
    metrics: &ExecutorMetrics,
    completion: TaskCompletion,
) {
    let end = match catch_unwind(AssertUnwindSafe(|| drop(func))) {
        Ok(()) => TaskEnd::Cancelled,
        Err(payload) => TaskEnd::Completed(Err(payload)),
    };
    send_task_result(end, sender, metrics, completion);
}

pub(super) fn send_task_result<T>(
    end: TaskEnd<T>,
    sender: Option<TaskResultSender<T>>,
    metrics: &ExecutorMetrics,
    completion: TaskCompletion,
) {
    let (result, panic_payload) = match end {
        TaskEnd::Completed(Ok(value)) => {
            metrics.record_task_completed(completion.duration());
            (Ok(value), None)
        }
        TaskEnd::Completed(Err(payload)) => {
            metrics.record_task_failed();
            (Err(TaskError::Panicked), Some(payload))
        }
        TaskEnd::Cancelled => {
            metrics.record_task_cancelled();
            (Err(TaskError::Cancelled), None)
        }
    };
    let callback = publish_result(sender, result, Some(completion), metrics);
    if let Some(payload) = panic_payload {
        drop_after_settlement(payload);
    }
    resume_callback(callback);
}

pub(super) fn publish_result<T>(
    sender: Option<TaskResultSender<T>>,
    result: Result<T, TaskError>,
    completion: Option<TaskCompletion>,
    metrics: &ExecutorMetrics,
) -> Option<PanicPayload> {
    let publication = catch_unwind(AssertUnwindSafe(|| {
        if let Some(sender) = sender {
            sender.send(result);
        } else {
            drop(result);
        }
    }));
    let notification = catch_unwind(AssertUnwindSafe(|| {
        if let Some(completion) = completion {
            completion.notify();
        }
    }));
    // This counter reports failures at these two publication boundaries,
    // including destruction of an uncollected result. It carries no ordering.
    for failed in [publication.is_err(), notification.is_err()] {
        if failed {
            metrics.publication_panics.fetch_add(1, Ordering::Relaxed);
        }
    }
    match (publication, notification) {
        (Ok(()), Ok(())) => None,
        (Err(payload), Ok(())) | (Ok(()), Err(payload)) => Some(payload),
        (Err(first), Err(second)) => {
            drop_after_settlement(second);
            Some(first)
        }
    }
}

/// Destroy retained completion values outside an existing unwind.
pub(super) fn drop_after_settlement<T>(value: T) {
    // The primary failure is already published. The scheduler boundary
    // contains this panic and aborts if its own panic payload also panics.
    crate::schedule::job::run_caught(|| drop(value));
}

pub(super) fn resume_callback(panic: Option<PanicPayload>) {
    if let Some(payload) = panic {
        resume_unwind(payload);
    }
}
