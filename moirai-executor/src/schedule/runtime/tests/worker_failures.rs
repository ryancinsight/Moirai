//! Worker failure cleanup during scheduler shutdown.

use std::{
    any::Any,
    panic::panic_any,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread::{self, ThreadId},
};

use super::*;
use crate::schedule::register_idle_hook;

const JOIN_IN_PROGRESS_STATE: u8 = 1;
const JOIN_COMPLETE_STATE: u8 = 2;

static HOOK_PAYLOAD: Mutex<Option<HookPayload>> = Mutex::new(None);
static HOOK_REGISTERED: OnceLock<()> = OnceLock::new();
static TEST_SERIAL: Mutex<()> = Mutex::new(());

struct HookPayload {
    target: ThreadId,
    payload: Box<dyn Any + Send>,
    fired: mpsc::SyncSender<()>,
}

enum CleanupAction {
    Reenter {
        scheduler: ThreadScheduler,
        observed_worker: Arc<AtomicBool>,
    },
    PanicAgain {
        second_drops: Arc<AtomicUsize>,
    },
}

struct WorkerPanicPayload {
    action: CleanupAction,
    entered: mpsc::SyncSender<()>,
    release: mpsc::Receiver<()>,
    drops: Arc<AtomicUsize>,
}

struct SecondPanicPayload {
    drops: Arc<AtomicUsize>,
}

#[derive(Clone, Copy)]
enum CleanupScenario {
    Reenter,
    PanicAgain,
}

#[derive(Debug, PartialEq, Eq)]
enum WaiterObservation {
    Reentry,
    SecondDrops(usize),
}

impl Drop for WorkerPanicPayload {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Release);
        match &self.action {
            CleanupAction::Reenter {
                scheduler,
                observed_worker,
            } => {
                observed_worker.store(scheduler.is_current_worker(), Ordering::Release);
                scheduler.shutdown();
                self.entered
                    .send(())
                    .expect("test controller observes recursive shutdown return");
                self.release
                    .recv()
                    .expect("test controller releases panic-payload cleanup");
            }
            CleanupAction::PanicAgain { second_drops } => {
                self.entered
                    .send(())
                    .expect("test controller observes panic-payload cleanup");
                self.release
                    .recv()
                    .expect("test controller releases panic-payload cleanup");
                panic_any(SecondPanicPayload {
                    drops: Arc::clone(second_drops),
                });
            }
        }
    }
}

impl Drop for SecondPanicPayload {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Release);
    }
}

fn worker_panic_hook() {
    let payload = {
        let mut slot = HOOK_PAYLOAD
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if slot
            .as_ref()
            .is_some_and(|payload| payload.target == thread::current().id())
        {
            slot.take()
        } else {
            None
        }
    };
    if let Some(payload) = payload {
        let HookPayload { payload, fired, .. } = payload;
        fired
            .send(())
            .expect("test controller observes the idle hook");
        std::panic::resume_unwind(payload);
    }
}

fn install_worker_panic(
    target: ThreadId,
    payload: WorkerPanicPayload,
    fired: mpsc::SyncSender<()>,
) {
    HOOK_REGISTERED.get_or_init(|| {
        register_idle_hook(worker_panic_hook)
            .expect("worker-failure test reserves one idle-hook slot");
    });
    let replaced = HOOK_PAYLOAD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .replace(HookPayload {
            target,
            payload: Box::new(payload),
            fired,
        });
    assert!(replaced.is_none(), "worker panic payload must be unique");
}

fn occupy_worker(scheduler: &ThreadScheduler) -> (ThreadId, mpsc::SyncSender<()>) {
    let (started_sender, started_receiver) = mpsc::sync_channel(0);
    let (release_sender, release_receiver) = mpsc::sync_channel(0);
    scheduler
        .schedule::<SyncTask, _>(Priority::Critical, Some(0), move |_| {
            started_sender
                .send(thread::current().id())
                .expect("test observer remains connected");
            release_receiver
                .recv()
                .expect("test controller releases the occupied worker");
        })
        .expect("gate job must be admitted");
    let worker = started_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("gate job must start before the test deadline");
    (worker, release_sender)
}

fn install_wait_registration(scheduler: &ThreadScheduler) -> mpsc::Receiver<()> {
    let (sender, receiver) = mpsc::channel();
    scheduler
        .inner
        .shutdown_wait_registration
        .set((sender, AtomicUsize::new(0)))
        .expect("test installs one external-wait observer");
    receiver
}

fn observe_wait_registration(scheduler: &ThreadScheduler, registered: &mpsc::Receiver<()>) {
    registered
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("external caller registers before the test deadline");
    let guard = scheduler
        .inner
        .wait_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (_, count) = scheduler.inner.shutdown_wait_registration.get().unwrap();
    assert_eq!(count.load(Ordering::Acquire), 1);
    drop(guard);
}

fn run_cleanup_scenario(scenario: CleanupScenario) {
    let _serial = TEST_SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scheduler_name = match scenario {
        CleanupScenario::Reenter => "panic-payload-reentry",
        CleanupScenario::PanicAgain => "second-panic-cleanup",
    };
    let scheduler = ThreadScheduler::new(1, scheduler_name).unwrap();
    let (worker, release_worker) = occupy_worker(&scheduler);
    let wait_registered = install_wait_registration(&scheduler);
    let observed_worker = Arc::new(AtomicBool::new(true));
    let first_drops = Arc::new(AtomicUsize::new(0));
    let second_drops = Arc::new(AtomicUsize::new(0));
    let (cleanup_entered_sender, cleanup_entered_receiver) = mpsc::sync_channel(0);
    let (cleanup_release_sender, cleanup_release_receiver) = mpsc::sync_channel(0);
    let (hook_fired_sender, hook_fired_receiver) = mpsc::sync_channel(0);
    let action = match scenario {
        CleanupScenario::Reenter => CleanupAction::Reenter {
            scheduler: scheduler.clone(),
            observed_worker: Arc::clone(&observed_worker),
        },
        CleanupScenario::PanicAgain => CleanupAction::PanicAgain {
            second_drops: Arc::clone(&second_drops),
        },
    };
    install_worker_panic(
        worker,
        WorkerPanicPayload {
            action,
            entered: cleanup_entered_sender,
            release: cleanup_release_receiver,
            drops: Arc::clone(&first_drops),
        },
        hook_fired_sender,
    );

    release_worker
        .send(())
        .expect("occupied worker remains connected");
    hook_fired_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("idle hook fires before the test deadline");
    let owner_scheduler = scheduler.clone();
    let (owner_done_sender, owner_done_receiver) = mpsc::sync_channel(0);
    let owner = thread::spawn(move || {
        owner_scheduler.shutdown();
        owner_done_sender
            .send(())
            .expect("test observer remains connected");
    });
    cleanup_entered_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect(match scenario {
            CleanupScenario::Reenter => "recursive shutdown returns before the test deadline",
            CleanupScenario::PanicAgain => "panic-payload cleanup starts before the test deadline",
        });

    let waiter_scheduler = scheduler.clone();
    let waiter_second_drops = Arc::clone(&second_drops);
    let (waiter_done_sender, waiter_done_receiver) = mpsc::sync_channel(0);
    let waiter = thread::spawn(move || {
        waiter_scheduler.shutdown();
        waiter_done_sender
            .send(match scenario {
                CleanupScenario::Reenter => WaiterObservation::Reentry,
                CleanupScenario::PanicAgain => {
                    WaiterObservation::SecondDrops(waiter_second_drops.load(Ordering::Acquire))
                }
            })
            .expect("test observer remains connected");
    });
    observe_wait_registration(&scheduler, &wait_registered);
    assert_eq!(
        scheduler.inner.shutdown_join_state.load(Ordering::Acquire),
        JOIN_IN_PROGRESS_STATE
    );
    match scenario {
        CleanupScenario::Reenter => {
            assert!(!observed_worker.load(Ordering::Acquire));
            assert_eq!(first_drops.load(Ordering::Acquire), 1);
            assert_eq!(
                owner_done_receiver.try_recv(),
                Err(mpsc::TryRecvError::Empty)
            );
            assert_eq!(
                waiter_done_receiver.try_recv(),
                Err(mpsc::TryRecvError::Empty)
            );
        }
        CleanupScenario::PanicAgain => {
            assert_eq!(second_drops.load(Ordering::Acquire), 0);
        }
    }

    cleanup_release_sender
        .send(())
        .expect("panic-payload cleanup remains connected");
    owner_done_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("shutdown owner finishes before the test deadline");
    let waiter_observation = waiter_done_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("external waiter finishes before the test deadline");
    match scenario {
        CleanupScenario::Reenter => {
            assert_eq!(waiter_observation, WaiterObservation::Reentry);
        }
        CleanupScenario::PanicAgain => {
            assert_eq!(waiter_observation, WaiterObservation::SecondDrops(1));
        }
    }
    owner.join().expect("shutdown owner must not panic");
    waiter.join().expect("external waiter must not panic");
    match scenario {
        CleanupScenario::Reenter => {
            assert_eq!(first_drops.load(Ordering::Acquire), 1);
        }
        CleanupScenario::PanicAgain => {
            assert_eq!(first_drops.load(Ordering::Acquire), 1);
            assert_eq!(second_drops.load(Ordering::Acquire), 1);
        }
    }
    assert_eq!(
        scheduler.inner.shutdown_join_state.load(Ordering::Acquire),
        JOIN_COMPLETE_STATE
    );
}

#[test]
fn panic_payload_reentry_returns_while_external_caller_waits() {
    run_cleanup_scenario(CleanupScenario::Reenter);
}

#[test]
fn second_panic_payload_is_cleaned_before_completion() {
    run_cleanup_scenario(CleanupScenario::PanicAgain);
}
