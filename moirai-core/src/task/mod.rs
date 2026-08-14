//! # Task Abstraction Layer
//!
//! This module provides the core task abstractions for the Moirai concurrency library.
//! All task types are designed to be zero-cost abstractions that compile away to optimal code.
//!
//! ## Safety Guarantees
//!
//! - **Memory Safety**: All task operations are memory-safe by construction
//! - **Data Race Freedom**: Rust's ownership system prevents data races
//! - **Resource Cleanup**: Automatic resource cleanup on task completion or panic
//! - **Type Safety**: Generic type system ensures compile-time correctness
//!
//! ## Performance Characteristics
//!
//! - **Task Creation**: O(1) constant time with zero allocations for simple closures
//! - **Task Execution**: Zero-cost abstractions compile to direct function calls
//! - **Memory Overhead**: < 64 bytes per task for metadata and context
//! - **Cache Efficiency**: Task data structures are cache-line aligned
//!
//! ## Examples
//!
//! ### Basic Task Creation
//!
//! ```rust
//! use moirai_core::{Task, TaskBuilder, Priority};
//!
//! // Simple closure task
//! let task = TaskBuilder::new()
//!     .priority(Priority::Normal)
//!     .name("computation")
//!     .build(|| {
//!         (1..=100).sum::<i32>()
//!     });
//!
//! assert_eq!(task.execute(), 5050);
//! ```
//!
//! ### Task Chaining and Composition
//!
//! ```rust,ignore
//! use moirai_core::{TaskBuilder, TaskExt, Task};
//!
//! let base_task = TaskBuilder::new().build(|| 21);
//!
//! // Chain operations
//! let doubled = base_task.then(|x| x * 2);
//! let result = doubled.execute();
//! assert_eq!(result, 42);
//!
//! // Map transformations
//! let mapped = TaskBuilder::new().build(|| "hello")
//!     .map(|s| s.to_uppercase());
//! assert_eq!(mapped.execute(), "HELLO");
//! ```
//!
//! ### Error Handling
//!
//! `Task::execute` has no error channel: fallible work returns a `Result` as
//! its `Output` and callers branch on the value. Panic recovery is the
//! executor's responsibility (the hybrid executor catches unwinds at the job
//! boundary), not a task-combinator concern.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

/// Maximum generic spin attempts before falling back to blocking
pub(crate) const MAX_SPIN_ATTEMPTS: usize = 64;

/// Task builder types: `TaskBuilder`, `BaseTask`, `Closure`, `Chained`, `Mapped`, `Parameterized`, `Group`, `Spawner`.
pub(crate) mod builder;
/// Task extension trait and combinator types: `TaskExt`, `ContextualTask`.
pub(crate) mod ext;
/// `TaskFuture<T>`: fused one-shot future that runs a task synchronously on first poll.
pub(crate) mod future;
/// Task handle and result-slot types: `TaskHandle`, `TaskResultSender`, `BlockingResultWait`, `ResultWaitPolicy`.
pub(crate) mod handle;
/// Core identity and context types: `TaskId`, `Priority`, `TaskContext`.
pub(crate) mod id_and_context;
/// Core `Task` trait definition and `Box<T: Task>` delegation impl.
pub(crate) mod traits;

// ── Public API re-exports ─────────────────────────────────────────────────────

pub use builder::{BaseTask, Chained, Closure, Group, Mapped, Parameterized, Spawner, TaskBuilder};
pub use ext::{ContextualTask, TaskExt};
pub use future::TaskFuture;
pub use handle::TaskHandle;
pub use id_and_context::{Priority, TaskContext, TaskId};
pub use traits::Task;

#[cfg(feature = "std")]
pub use handle::{BlockingResultWait, ResultWaitPolicy, TaskResultSender};

#[cfg(all(feature = "std", feature = "result-diagnostics"))]
pub use handle::{
    diagnostic_result_slot_complete_waiting, diagnostic_result_slot_ready_take,
    diagnostic_result_slot_register_waiter, diagnostic_result_slot_spin_miss,
};

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TaskBuilder;

    #[test]
    fn test_task_future() {
        let id = TaskId::new(1);
        let task = TaskBuilder::new().with_id(id).build(|| 42);
        let future = TaskFuture::new(task, TaskContext::new(id));

        assert_eq!(future.context().id, id);
    }

    fn poll_once<F: core::future::Future + Unpin>(future: &mut F) -> core::task::Poll<F::Output> {
        // No-op waker: TaskFuture never returns Pending, so the waker is unused.
        let waker = core::task::Waker::noop();
        let mut cx = core::task::Context::from_waker(waker);
        core::pin::Pin::new(future).poll(&mut cx)
    }

    #[test]
    fn task_future_resolves_task_output_on_first_poll() {
        let id = TaskId::new(2);
        let task = TaskBuilder::new().with_id(id).build(|| 21 * 2);
        let mut future = TaskFuture::new(task, TaskContext::new(id));

        assert_eq!(poll_once(&mut future), core::task::Poll::Ready(42));
    }

    #[test]
    #[should_panic(expected = "TaskFuture polled after completion")]
    fn task_future_panics_when_polled_after_completion() {
        let id = TaskId::new(3);
        let task = TaskBuilder::new().with_id(id).build(|| 1);
        let mut future = TaskFuture::new(task, TaskContext::new(id));

        assert_eq!(poll_once(&mut future), core::task::Poll::Ready(1));
        let _ = poll_once(&mut future); // fused: must panic, not hang as Pending
    }

    #[test]
    fn test_task_composition() {
        let id = TaskId::new(1);
        let task = TaskBuilder::new().with_id(id).build(|| 10);

        // Test map combinator
        let mapped = task.map(|x| x * 2);
        assert_eq!(mapped.execute(), 20);
    }

    #[test]
    fn test_task_group() {
        let mut group = Group::new(TaskId::new(1));

        let task1 = TaskBuilder::new().with_id(TaskId::new(2)).build(|| 42);

        let task2 = TaskBuilder::new().with_id(TaskId::new(3)).build(|| 24);

        // Wrap tasks in closures for the group
        group.add_task(|| {
            let _ = task1.execute();
        });
        group.add_task(|| {
            let _ = task2.execute();
        });

        assert_eq!(group.len(), 2);
        assert!(!group.is_empty());

        // Execute the group
        group.execute();
    }

    #[test]
    fn test_parameterized_task() {
        let id = TaskId::new(1);
        let task = Parameterized::new(|x: i32| x * 3, 7, TaskContext::new(id));

        assert_eq!(task.execute(), 21);
    }

    #[test]
    fn test_spawner_task() {
        let id = TaskId::new(1);
        let spawner = Spawner::new(
            || {
                // This would spawn other tasks in a real implementation
            },
            TaskContext::new(id),
        );

        spawner.execute(); // Should not panic
    }

    #[test]
    fn task_handle_returns_sent_result() {
        let (handle, sender) = TaskHandle::new_pending(TaskId::new(10));

        assert!(!handle.is_finished());
        sender.send(Ok(42usize));
        assert!(handle.is_finished());
        assert_eq!(handle.join(), Some(Ok(42)));
    }

    #[test]
    fn task_handle_ready_returns_stored_result() {
        let handle = TaskHandle::ready(TaskId::new(12), Ok(84usize));

        assert!(handle.is_finished());
        assert_eq!(handle.id(), TaskId::new(12));
        assert_eq!(handle.join(), Some(Ok(84)));
    }

    #[test]
    fn task_handle_reports_cancelled_when_sender_drops() {
        let (handle, sender) = TaskHandle::<usize>::new_pending(TaskId::new(11));

        drop(sender);

        assert!(handle.is_finished());
        assert_eq!(handle.join(), Some(Err(crate::error::TaskError::Cancelled)));
    }

    #[test]
    fn task_handle_waits_for_cross_thread_completion() {
        let (handle, sender) = TaskHandle::new_pending(TaskId::new(13));

        let worker = std::thread::spawn(move || {
            sender.send(Ok(168usize));
        });

        assert_eq!(handle.join(), Some(Ok(168)));
        worker.join().unwrap();
    }

    #[test]
    fn task_handle_parks_until_delayed_completion() {
        let (handle, sender) = TaskHandle::new_pending(TaskId::new(14));

        let worker = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            sender.send(Ok(336usize));
        });

        assert_eq!(handle.join(), Some(Ok(336)));
        worker.join().unwrap();
    }

    #[test]
    fn result_wait_policy_is_zero_sized_and_const_bounded() {
        assert_eq!(core::mem::size_of::<BlockingResultWait>(), 0);
        assert_eq!(BlockingResultWait::SPIN_ATTEMPTS, MAX_SPIN_ATTEMPTS);
    }
}
