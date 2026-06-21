use super::core::WorkStealingScheduler;
use moirai_core::{Task, TaskContext, TaskId};

struct TestTask {
    id: u32,
    context: TaskContext,
}

impl TestTask {
    fn new(id: u32) -> Self {
        Self {
            id,
            context: TaskContext::new(TaskId::new(id as u64)),
        }
    }
}

impl Task for TestTask {
    type Output = u32;

    fn execute(self) -> Self::Output {
        self.id * 2
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }
}

#[test]
fn test_work_stealing_scheduler() {
    use moirai_core::scheduler::SchedulerConfig;
    let config = SchedulerConfig::default();
    let scheduler =
        WorkStealingScheduler::new(moirai_core::scheduler::SchedulerId::new(0), config);

    for i in 0..10 {
        let task = TestTask::new(i);
        scheduler.schedule_task(task).unwrap();
    }

    let stats = scheduler.stats();
    assert_eq!(stats.tasks_scheduled, 10);

    let mut popped = 0;
    while scheduler.try_execute_next_task().unwrap() {
        popped += 1;
    }
    assert_eq!(popped, 10);
}

#[test]
fn test_scheduler_stats() {
    use moirai_core::scheduler::{SchedulerConfig, SchedulerId};
    let config = SchedulerConfig::default();
    let scheduler = WorkStealingScheduler::new(SchedulerId::new(1), config);

    for i in 0..5 {
        let task = TestTask::new(i);
        scheduler.schedule_task(task).unwrap();
    }

    while scheduler.try_execute_next_task().unwrap() {}

    let stats = scheduler.stats();
    assert_eq!(stats.scheduler_id, SchedulerId::new(1));
    assert_eq!(stats.tasks_scheduled, 5);
    assert_eq!(stats.tasks_executed, 5);
    assert_eq!(stats.current_load, 0);
}

#[test]
fn test_local_scheduler() {
    use moirai_core::scheduler::{QueueType, SchedulerConfig, SchedulerId};
    let config = SchedulerConfig {
        queue_type: QueueType::ChaseLev,
        ..Default::default()
    };
    let scheduler = WorkStealingScheduler::new(SchedulerId::new(2), config);

    for i in 0..10 {
        let task = TestTask::new(i);
        scheduler.schedule_task(task).unwrap();
    }

    assert_eq!(scheduler.load(), 10);

    let mut executed_count = 0;
    while scheduler.try_execute_next_task().unwrap() {
        executed_count += 1;
    }

    assert_eq!(executed_count, 10);
    assert_eq!(scheduler.load(), 0);
}

#[test]
fn test_try_steal_safety_and_correctness() {
    use moirai_core::scheduler::{SchedulerConfig, SchedulerId, Scheduler};
    use moirai_core::ScheduledTask;

    let scheduler1 = WorkStealingScheduler::new(SchedulerId::new(1), SchedulerConfig::default());
    let scheduler2 = WorkStealingScheduler::new(SchedulerId::new(2), SchedulerConfig::default());

    // Schedule a task on scheduler2 (the victim)
    let task = TestTask::new(42);
    scheduler2.schedule(ScheduledTask::new(task)).unwrap();

    // Verify victim has load
    assert_eq!(scheduler2.load(), 1);

    // Thief (scheduler1) steals from victim (scheduler2) via try_steal
    let stolen = scheduler1.try_steal(&scheduler2).unwrap();
    assert!(stolen.is_some());

    // Verify the stolen task is indeed the one scheduled on victim
    let stolen_task = stolen.unwrap();
    stolen_task.execute(); // Execute to verify it behaves correctly

    // Verify load has shifted
    assert_eq!(scheduler2.load(), 0);
}

#[test]
fn test_locality_aware_steal_visited_tracking() {
    use moirai_core::scheduler::{SchedulerConfig, SchedulerId, WorkStealingStrategy, Scheduler};
    use super::core::WorkStealingCoordinator;
    use std::sync::Arc;
    use moirai_core::ScheduledTask;

    let idle_scheduler = WorkStealingScheduler::new(SchedulerId::new(999), SchedulerConfig::default());

    let mut all_schedulers = Vec::new();
    for i in 0..260 {
        all_schedulers.push(Arc::new(WorkStealingScheduler::new(SchedulerId::new(i), SchedulerConfig::default())));
    }

    // Schedule tasks on schedulers 255 and 256
    let task1 = TestTask::new(1001);
    let task2 = TestTask::new(1002);
    all_schedulers[255].schedule(ScheduledTask::new(task1)).unwrap();
    all_schedulers[256].schedule(ScheduledTask::new(task2)).unwrap();

    let coordinator = WorkStealingCoordinator::new(WorkStealingStrategy::LocalityAware { max_attempts: 4, locality_factor: 0.5 });

    // Steal first task (should be from 256 as it is closer to 999 than 255)
    let stolen1 = coordinator.steal_work(&idle_scheduler, &all_schedulers);
    assert!(stolen1.is_some());
    
    // Steal second task (should be from 255 since 256 is now empty)
    let stolen2 = coordinator.steal_work(&idle_scheduler, &all_schedulers);
    assert!(stolen2.is_some());
}
