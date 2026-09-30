use std::{
    sync::{Arc, mpsc},
    time::Duration,
};

use moirai_core::{Priority, executor::ExecutorConfig};

use super::super::HybridExecutor;
use crate::{BlockingTask, SyncTask, WorkClass};

fn executor_last_owner_drops_inside_job<C: WorkClass>() {
    let executor = Arc::new(
        HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .expect("executor"),
    );
    let registry_owner = Arc::downgrade(&executor.task_registry);
    let metrics_owner = Arc::downgrade(&executor.metrics);
    let executor_in_job = Arc::clone(&executor);
    let (release_sender, release_receiver) = mpsc::sync_channel(0);
    let (completed_sender, completed_receiver) = mpsc::sync_channel(1);
    let handle = executor
        .spawn_result::<C, _>(Priority::Normal, None, move || {
            release_receiver.recv().expect("release observer alive");
            drop(executor_in_job);
            assert!(
                registry_owner.upgrade().is_some(),
                "scheduler must retain registry storage through the job wrapper"
            );
            assert!(
                metrics_owner.upgrade().is_some(),
                "scheduler must retain metrics storage through the job wrapper"
            );
            completed_sender
                .send(37usize)
                .expect("completion observer alive");
            37usize
        })
        .expect("job admits");

    drop(executor);
    release_sender
        .send(())
        .expect("scheduled job retains the executor");
    assert_eq!(
        completed_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("re-entrant executor drop must not self-join"),
        37
    );
    assert_eq!(handle.join(), Some(Ok(37)));
}

#[test]
fn scheduler_owner_outlives_reentrant_executor_drop() {
    executor_last_owner_drops_inside_job::<SyncTask>();
    executor_last_owner_drops_inside_job::<BlockingTask>();
}

#[test]
fn configured_global_queue_capacity_reaches_worker_injectors() {
    let mut executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 3,
        max_global_queue_size: 1000,
        ..ExecutorConfig::default()
    })
    .unwrap();

    assert_eq!(executor.scheduler.injector_capacity_per_worker(), 256);
    HybridExecutor::shutdown(&mut executor).unwrap();
}

#[test]
fn impossible_global_queue_capacity_is_rejected_before_startup() {
    let result = HybridExecutor::new(ExecutorConfig {
        worker_threads: 4,
        max_global_queue_size: 3,
        ..ExecutorConfig::default()
    });

    assert!(matches!(
        result,
        Err(moirai_core::ExecutorError::InvalidConfiguration)
    ));
}

#[test]
fn local_queue_capacity_configuration_is_retained() {
    let mut executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 2,
        local_queue_initial_capacity: 17,
        ..ExecutorConfig::default()
    })
    .unwrap();

    assert_eq!(executor.config().local_queue_initial_capacity, 17);
    HybridExecutor::shutdown(&mut executor).unwrap();
}

#[test]
fn invalid_local_queue_capacity_is_rejected_before_startup() {
    let result = HybridExecutor::new(ExecutorConfig {
        worker_threads: 2,
        local_queue_initial_capacity: usize::MAX,
        ..ExecutorConfig::default()
    });

    assert!(matches!(
        result,
        Err(
            moirai_core::ExecutorError::InvalidLocalQueueInitialCapacity {
                requested: usize::MAX
            }
        )
    ));
}
