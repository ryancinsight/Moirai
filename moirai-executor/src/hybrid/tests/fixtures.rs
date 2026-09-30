use moirai_core::Priority;

use super::super::HybridExecutor;
use crate::WorkClass;

/// Occupy the single worker with a job gated on a channel; returns the
/// release sender and blocks until the gate job has started.
pub(super) fn gate_single_worker<C: WorkClass>(
    executor: &HybridExecutor,
) -> (
    std::sync::mpsc::Sender<()>,
    moirai_core::task::TaskHandle<()>,
) {
    let (release_sender, release_receiver) = std::sync::mpsc::channel::<()>();
    let (started_sender, started_receiver) = std::sync::mpsc::channel::<()>();
    let handle = executor
        .spawn_result::<C, _>(Priority::Normal, None, move || {
            started_sender.send(()).unwrap();
            release_receiver
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("gate must be released before the test deadline");
        })
        .unwrap();
    started_receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("gate task must start");
    (release_sender, handle)
}
