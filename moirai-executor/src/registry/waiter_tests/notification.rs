use std::{
    sync::atomic::Ordering,
    sync::{Arc, Barrier, Mutex},
    task::Waker,
    thread,
};

use crate::counting_wake::counting_waker;

use super::super::registry::TaskRegistry;
use super::callback_behavior::{CLONE_NONE, WAKE_NONE, callback_target, callback_waker};
use super::owned_registration::pending_task;

struct GatedWake {
    entered: std::sync::mpsc::SyncSender<()>,
    release: Mutex<std::sync::mpsc::Receiver<()>>,
}

impl std::task::Wake for GatedWake {
    fn wake(self: Arc<Self>) {
        self.entered.send(()).unwrap();
        self.release.lock().unwrap().recv().unwrap();
    }
}

#[test]
fn dropping_an_unclaimed_closed_waiter_prevents_its_notification() {
    let registry = Arc::new(TaskRegistry::new());
    let (id, lifecycle) = pending_task(&registry);
    let (entered_sender, entered_receiver) = std::sync::mpsc::sync_channel(1);
    let (release_sender, release_receiver) = std::sync::mpsc::sync_channel(1);
    let gated_waker = Waker::from(Arc::new(GatedWake {
        entered: entered_sender,
        release: Mutex::new(release_receiver),
    }));
    let _gated_waiter = registry.register_waker(id, &gated_waker).unwrap();
    let pending = callback_target(
        Arc::clone(&registry),
        id,
        Waker::noop().clone(),
        CLONE_NONE,
        false,
        false,
        WAKE_NONE,
    );
    let pending_waker = callback_waker(Arc::clone(&pending));
    let pending_waiter = registry.register_waker(id, &pending_waker).unwrap();

    let completion = thread::spawn(move || lifecycle.start(0).complete().notify());
    entered_receiver.recv().unwrap();
    drop(pending_waiter);
    assert_eq!(
        pending.waker_drops.load(Ordering::Acquire),
        1,
        "dropping the owner releases its unclaimed closed registration"
    );
    assert_eq!(pending.wakes.load(Ordering::Acquire), 0);

    release_sender.send(()).unwrap();
    completion.join().unwrap();
    assert_eq!(pending.wakes.load(Ordering::Acquire), 0);
}

#[test]
fn completion_and_unregister_can_race_block_retirement() {
    use std::time::Duration;

    let registry = Arc::new(TaskRegistry::new());
    let (id, lifecycle) = pending_task(&registry);
    let (target, waker) = counting_waker();
    let waiter = registry.register_waker(id, &waker).unwrap();
    for _ in 1..super::state::TASK_STATE_BLOCK_SIZE {
        let (_, other) = pending_task(&registry);
        other.start(0).complete().notify();
    }
    let start = Arc::new(Barrier::new(3));
    let completion_start = Arc::clone(&start);
    let completion = thread::spawn(move || {
        completion_start.wait();
        lifecycle.start(0).complete().notify();
    });
    let removal_start = Arc::clone(&start);
    let removal = thread::spawn(move || {
        removal_start.wait();
        drop(waiter);
    });

    start.wait();
    completion.join().unwrap();
    removal.join().unwrap();
    assert!(target.wakes() <= 1);
    assert_eq!(registry.cleanup_completed(Duration::ZERO), 1);
    assert!(registry.is_completed(id));
}

/// Registrations from different threads all land before completion, so every
/// waiter's post-registration recheck is `false` and only the completion wake
/// can reach it.
#[test]
fn waiters_on_separate_threads_are_all_woken_by_completion() {
    const WAITERS: usize = 6;
    let registry = Arc::new(TaskRegistry::new());
    let (id, lifecycle) = pending_task(&registry);
    let registered = Arc::new(Barrier::new(WAITERS + 1));
    let waiters: Vec<_> = (0..WAITERS)
        .map(|_| {
            let (target, waker) = counting_waker();
            let registry = Arc::clone(&registry);
            let registered = Arc::clone(&registered);
            let thread = thread::spawn(move || {
                let waiter = registry.register_waker(id, &waker).unwrap();
                let completed_at_recheck = registry.is_completed(id);
                registered.wait();
                (completed_at_recheck, waiter)
            });
            (target, thread)
        })
        .collect();

    registered.wait();
    lifecycle.start(0).complete().notify();

    for (index, (target, thread)) in waiters.into_iter().enumerate() {
        let (completed_at_recheck, waiter) = thread.join().unwrap();
        assert!(!completed_at_recheck, "waiter {index} saw completion early");
        assert_eq!(target.wakes(), 1, "waiter {index}");
        drop(waiter);
    }
}

/// Completion races the registrations. Whatever the interleaving, every waiter
/// ends up woken exactly once: by the completion, or by the registration or
/// another waiter's recheck that finds the task complete.
#[test]
fn completion_racing_registrations_never_strands_a_waiter() {
    const WAITERS: usize = 4;
    const ROUNDS: usize = 200;
    for round in 0..ROUNDS {
        let registry = Arc::new(TaskRegistry::new());
        let (id, lifecycle) = pending_task(&registry);
        let start = Arc::new(Barrier::new(WAITERS + 1));
        let (targets, threads): (Vec<_>, Vec<_>) = (0..WAITERS)
            .map(|_| {
                let (target, waker) = counting_waker();
                let registry = Arc::clone(&registry);
                let start = Arc::clone(&start);
                let thread = thread::spawn(move || {
                    start.wait();
                    let waiter = registry.register_waker(id, &waker);
                    let completed = registry.is_completed(id);
                    (waiter, completed)
                });
                (target, thread)
            })
            .unzip();

        start.wait();
        lifecycle.start(0).complete().notify();

        // Join every thread before asserting: a recheck on one thread may take
        // another waiter's waker and wake it before that waiter's thread
        // returns.
        for (index, (target, thread)) in targets.iter().zip(threads).enumerate() {
            let (waiter, completed) = thread.join().unwrap();
            if waiter.is_some() {
                assert_eq!(target.wakes(), 1, "round {round}: waiter {index}");
            } else {
                assert!(completed, "a rejected registration observes completion");
                assert_eq!(target.wakes(), 0, "round {round}: waiter {index}");
            }
        }
    }
}
