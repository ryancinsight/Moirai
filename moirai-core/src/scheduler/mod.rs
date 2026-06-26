//! Scheduler trait and implementations.
//!
//! This module provides advanced scheduling algorithms inspired by:
//! - Rayon's work-stealing deque (Chase-Lev algorithm)
//! - Tokio's async notification system
//! - OpenMP's low-overhead synchronization

pub mod config;
pub mod task;
pub(crate) mod traits;

pub use config::{Config, QueueType, SchedulerConfig, Stats, StealContext, WorkStealingStrategy};
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
    fn test_scheduled_task_zero_object_dispatch() {
        let sum = Arc::new(AtomicUsize::new(0));
        let task = ScheduledTask::new(TestTask::new(42, 42, Arc::clone(&sum)));

        assert_eq!(task.context().id, TaskId::new(42));
        task.execute();
        assert_eq!(sum.load(Ordering::Relaxed), 42);
    }
}
