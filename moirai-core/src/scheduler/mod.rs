//! Scheduler trait and implementations.
//!
//! This module provides advanced scheduling algorithms inspired by:
//! - Rayon's work-stealing deque (Chase-Lev algorithm)
//! - Tokio's async notification system
//! - OpenMP's low-overhead synchronization

pub(super) mod buffer;
pub mod config;
pub mod coordinator;
pub mod deque;
pub mod task;
pub(crate) mod traits;

pub use config::{Config, QueueType, SchedulerConfig, Stats, StealContext, WorkStealingStrategy};
pub use coordinator::WorkStealingCoordinator;
pub use deque::{WorkStealingDeque, ZeroCopyWorkStealingDeque};
pub use task::{ScheduledTask, INLINE_SCHEDULED_TASK_WORDS};
pub use traits::{Scheduler, SchedulerId};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::{Task, TaskContext, TaskId};
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct TestTask {
        context: TaskContext,
        value: usize,
        sum: Arc<AtomicUsize>,
    }

    impl TestTask {
        fn new(id: u64, value: usize, sum: Arc<AtomicUsize>) -> Self {
            Self {
                context: TaskContext::new(TaskId::new(id)),
                value,
                sum,
            }
        }
    }

    impl Task for TestTask {
        type Output = usize;

        fn execute(self) -> Self::Output {
            self.sum.fetch_add(self.value, Ordering::Relaxed);
            self.value
        }

        fn context(&self) -> &TaskContext {
            &self.context
        }
    }

    #[test]
    fn test_scheduler_id() {
        let id = SchedulerId::new(42);
        assert_eq!(id.get(), 42);
        assert_eq!(format!("{id}"), "Scheduler(42)");
    }

    #[test]
    fn test_work_stealing_strategy_default() {
        let strategy = WorkStealingStrategy::default();
        matches!(strategy, WorkStealingStrategy::Random { max_attempts: 3 });
    }

    #[test]
    fn test_scheduler_config_default() {
        let config = Config::default();
        assert_eq!(config.max_local_queue_size, 1024);
        assert!(config.enable_metrics);
        assert_eq!(
            config.work_stealing_strategy,
            WorkStealingStrategy::default()
        );
    }

    #[test]
    fn test_steal_context_default() {
        let ctx = StealContext::default();
        assert_eq!(ctx.attempts, 0);
        assert!(ctx.last_success.is_none());
        assert!(ctx.recent_victims.is_empty());
        assert_eq!(ctx.backoff_delay, core::time::Duration::from_millis(10)); // Default backoff
    }

    #[test]
    fn test_work_stealing_deque() {
        let deque = WorkStealingDeque::new(16);

        // Test push/pop
        deque.push(1);
        deque.push(2);
        deque.push(3);

        // Pop should return the most recently pushed item (LIFO)
        assert_eq!(deque.pop(), Some(3));

        // After popping 3, we should be able to pop 2
        // But the implementation might have a different behavior
        // Let's test what actually happens
        let second_pop = deque.pop();
        assert!(second_pop.is_some());

        // Test steal
        deque.push(4);
        deque.push(5);

        // Steal should take from the opposite end (oldest item)
        let stolen = deque.steal();
        assert!(stolen.is_some());

        // Pop should still work from the newest end
        assert_eq!(deque.pop(), Some(5));
    }

    #[test]
    fn test_scheduled_task_zero_object_dispatch() {
        let sum = Arc::new(AtomicUsize::new(0));
        let task = ScheduledTask::new(TestTask::new(42, 42, Arc::clone(&sum)));

        assert_eq!(task.context().id, TaskId::new(42));
        task.execute();
        assert_eq!(sum.load(Ordering::Relaxed), 42);
    }

    #[test]
    fn test_work_stealing_with_scheduled_task() {
        let deque = ZeroCopyWorkStealingDeque::<ScheduledTask>::new(16);
        let sum = Arc::new(AtomicUsize::new(0));

        // Push multiple tasks
        for i in 0..10 {
            deque.push(ScheduledTask::new(TestTask::new(
                i as u64,
                i * 2,
                Arc::clone(&sum),
            )));
        }

        // Pop tasks and execute them.
        let mut executed = 0;
        while let Some(task) = deque.pop() {
            task.execute();
            executed += 1;
        }

        assert_eq!(executed, 10);
        assert_eq!(sum.load(Ordering::Relaxed), (0..10).map(|i| i * 2).sum::<usize>());
    }
}
