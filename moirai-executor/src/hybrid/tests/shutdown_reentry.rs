use std::{
    sync::{Arc, mpsc},
    time::Duration,
};

use moirai_core::{
    Priority,
    executor::{ExecutorConfig, ExecutorControl},
};

use super::{
    super::{HybridExecutor, shutdown::Context},
    fixtures::gate_single_worker,
};
use crate::{BlockingTask, SyncTask};

const EVENT_DEADLINE: Duration = Duration::from_secs(5);

fn executor(thread_name: &str) -> HybridExecutor {
    HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        thread_name_prefix: thread_name.into(),
        ..ExecutorConfig::default()
    })
    .expect("shutdown reentry executor constructs")
}

#[test]
fn final_drop_returns_while_compute_and_blocking_jobs_are_gated() {
    let executor = executor("drop-request-only");
    let (release_compute, compute) = gate_single_worker::<SyncTask>(&executor);
    let (release_blocking, blocking) = gate_single_worker::<BlockingTask>(&executor);
    let (returned_sender, returned) = mpsc::sync_channel(1);

    let dropping = std::thread::spawn(move || {
        drop(executor);
        returned_sender
            .send(())
            .expect("drop-return observer remains connected");
    });
    returned
        .recv_timeout(EVENT_DEADLINE)
        .expect("executor drop returns while both accepted jobs remain gated");
    dropping.join().expect("drop caller does not panic");

    release_compute
        .send(())
        .expect("compute gate remains connected");
    release_blocking
        .send(())
        .expect("blocking gate remains connected");
    assert_eq!(compute.join(), Some(Ok(())));
    assert_eq!(blocking.join(), Some(Ok(())));
}

#[test]
fn target_compute_worker_requests_shutdown_without_self_waiting() {
    let executor = Arc::new(executor("native-self-shutdown"));
    let worker_executor = Arc::clone(&executor);
    let (returned_sender, returned) = mpsc::sync_channel(1);
    let handle = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, move || {
            ExecutorControl::shutdown(&*worker_executor);
            returned_sender
                .send(())
                .expect("worker-return observer remains connected");
            31usize
        })
        .expect("self-shutdown job admits");

    returned
        .recv_timeout(EVENT_DEADLINE)
        .expect("target worker shutdown returns before its own job settles");
    assert_eq!(handle.join(), Some(Ok(31)));
    ExecutorControl::shutdown(&*executor);
}

#[test]
fn target_context_requests_shutdown_without_waiting_on_owned_work() {
    let executor = Arc::new(executor("context-self-shutdown"));
    let (release, gate) = gate_single_worker::<SyncTask>(&executor);
    let caller_executor = Arc::clone(&executor);
    let (returned_sender, returned) = mpsc::sync_channel(1);
    let caller = std::thread::spawn(move || {
        Context::with(&caller_executor.admission, || {
            ExecutorControl::shutdown(&*caller_executor);
        });
        returned_sender
            .send(())
            .expect("context-return observer remains connected");
    });

    returned
        .recv_timeout(EVENT_DEADLINE)
        .expect("target context returns while its accepted job remains gated");
    release.send(()).expect("worker gate remains connected");
    assert_eq!(gate.join(), Some(Ok(())));
    caller.join().expect("context caller does not panic");
    ExecutorControl::shutdown(&*executor);
}
