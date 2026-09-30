use moirai_core::{
    Priority,
    executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner, TaskStatus},
    task::TaskBuilder,
};

use super::super::HybridExecutor;

#[test]
fn priority_spawn_preserves_task_result() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let task = TaskBuilder::new()
        .priority(Priority::Critical)
        .build(|| 11usize);
    let handle = executor
        .spawn_with_priority(task, Priority::Critical, Some(0))
        .unwrap();

    assert_eq!(handle.join().unwrap().unwrap(), 11);
    executor.shutdown();
}

#[test]
fn task_stats_reports_recorded_spawn_priority() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let task = TaskBuilder::new()
        .priority(Priority::Critical)
        .build(|| 2usize);
    let critical = executor
        .spawn_with_priority(task, Priority::Critical, None)
        .unwrap();
    let normal = executor.spawn_blocking(|| 4usize).unwrap();
    let critical_id = critical.id();
    let normal_id = normal.id();

    assert_eq!(critical.join().unwrap().unwrap(), 2);
    assert_eq!(normal.join().unwrap().unwrap(), 4);

    let critical_stats = executor.task_stats(critical_id).unwrap();
    assert_eq!(critical_stats.priority, Priority::Critical);
    assert_eq!(critical_stats.status, TaskStatus::Completed);

    let normal_stats = executor.task_stats(normal_id).unwrap();
    assert_eq!(normal_stats.priority, Priority::Normal);
    executor.shutdown();
}

#[test]
fn spawn_honors_task_context_priority() {
    // `spawn` must record the task's own context priority (previously only
    // `spawn_with_priority` did).
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let task = TaskBuilder::new().priority(Priority::High).build(|| 8usize);
    let handle = executor.spawn(task).unwrap();
    let id = handle.id();
    assert_eq!(handle.join().unwrap().unwrap(), 8);
    assert_eq!(executor.task_stats(id).unwrap().priority, Priority::High);
    executor.shutdown();
}
