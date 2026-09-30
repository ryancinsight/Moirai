#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

//! Several waiters on one task: completion wakes every distinct waker, a
//! repeated registration of one waker counts once, and nothing registered is
//! retained past completion or past the task state.

use std::{
    sync::{Arc, Barrier},
    thread,
};

use crate::counting_wake::counting_waker;

use super::registry::TaskRegistry;
use super::state::TaskState;
use super::token::TaskLifecycleToken;

/// Register a task and keep its lifecycle token, so the task stays pending
/// until the token completes it (a dropped unstarted token completes the task).
fn pending_task(registry: &TaskRegistry) -> (u64, TaskLifecycleToken) {
    let (id, lifecycle) = registry.register_next_task();
    assert!(
        !registry.is_completed(id),
        "a held token leaves the task pending"
    );
    (id, lifecycle)
}

#[test]
fn completion_wakes_two_waiters_on_one_task() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (first, first_waker) = counting_waker();
    let (second, second_waker) = counting_waker();

    assert!(registry.register_waker(id, &first_waker));
    assert!(registry.register_waker(id, &second_waker));
    assert_eq!(first.wakes() + second.wakes(), 0, "nothing completed yet");

    lifecycle.start(0).complete();

    assert_eq!(first.wakes(), 1, "the first waiter was overwritten");
    assert_eq!(second.wakes(), 1);
}

#[test]
fn completion_wakes_every_distinct_waiter_once() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let waiters: Vec<_> = (0..5).map(|_| counting_waker()).collect();

    // A second round with the same wakers models waiters that are polled
    // again: each must still be woken once, and none may accumulate.
    for _round in 0..2 {
        for (_, waker) in &waiters {
            assert!(registry.register_waker(id, waker));
        }
    }
    lifecycle.start(0).complete();

    for (index, (target, _)) in waiters.iter().enumerate() {
        assert_eq!(target.wakes(), 1, "waiter {index}");
    }
}

#[test]
fn one_waker_registered_repeatedly_is_held_and_woken_once() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (target, waker) = counting_waker();

    for _ in 0..3 {
        assert!(registry.register_waker(id, &waker));
    }
    assert_eq!(
        Arc::strong_count(&target),
        3,
        "the test's handle, the waker, and exactly one registry entry"
    );
    lifecycle.start(0).complete();

    assert_eq!(target.wakes(), 1);
    assert_eq!(
        Arc::strong_count(&target),
        2,
        "completion releases the registry entry"
    );
}

#[test]
fn waiters_registered_after_completion_are_woken_and_not_retained() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (early, early_waker) = counting_waker();
    let (early_second, early_second_waker) = counting_waker();
    registry.register_waker(id, &early_waker);
    registry.register_waker(id, &early_second_waker);
    lifecycle.start(0).complete();

    let (late, late_waker) = counting_waker();
    let (late_second, late_second_waker) = counting_waker();
    assert!(registry.register_waker(id, &late_waker));
    assert!(registry.register_waker(id, &late_second_waker));
    drop((
        early_waker,
        early_second_waker,
        late_waker,
        late_second_waker,
    ));

    for target in [&early, &early_second, &late, &late_second] {
        assert_eq!(target.wakes(), 1);
        assert_eq!(
            Arc::strong_count(target),
            1,
            "a waker registered for a completed task is held by the registry"
        );
    }
}

#[test]
fn dropping_an_unfinished_state_releases_every_registered_waker() {
    let state = TaskState::new();
    let waiters: Vec<_> = (0..3).map(|_| counting_waker()).collect();
    for (_, waker) in &waiters {
        state.register_waker(waker);
    }
    let targets: Vec<_> = waiters
        .into_iter()
        .map(|(target, waker)| {
            drop(waker);
            target
        })
        .collect();

    drop(state);

    for target in &targets {
        assert_eq!(target.wakes(), 0, "an unfinished task wakes nobody");
        assert_eq!(Arc::strong_count(target), 1, "a waker outlived its state");
    }
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
                assert!(registry.register_waker(id, &waker));
                let completed_at_recheck = registry.is_completed(id);
                registered.wait();
                completed_at_recheck
            });
            (target, thread)
        })
        .collect();

    registered.wait();
    lifecycle.start(0).complete();

    for (index, (target, thread)) in waiters.into_iter().enumerate() {
        assert!(
            !thread.join().unwrap(),
            "waiter {index} saw completion early"
        );
        assert_eq!(target.wakes(), 1, "waiter {index}");
    }
}

/// Completion races the registrations. A waiter is served if its recheck saw
/// the completion or it was woken; neither is a lost wakeup.
#[test]
fn completion_racing_registrations_never_strands_a_waiter() {
    const WAITERS: usize = 4;
    const ROUNDS: usize = 200;
    for round in 0..ROUNDS {
        let registry = Arc::new(TaskRegistry::new());
        let (id, lifecycle) = pending_task(&registry);
        let start = Arc::new(Barrier::new(WAITERS + 1));
        let waiters: Vec<_> = (0..WAITERS)
            .map(|_| {
                let (target, waker) = counting_waker();
                let registry = Arc::clone(&registry);
                let start = Arc::clone(&start);
                let thread = thread::spawn(move || {
                    start.wait();
                    assert!(registry.register_waker(id, &waker));
                    registry.is_completed(id)
                });
                (target, thread)
            })
            .collect();

        start.wait();
        lifecycle.start(0).complete();

        for (index, (target, thread)) in waiters.into_iter().enumerate() {
            let saw_completion = thread.join().unwrap();
            assert!(
                saw_completion || target.wakes() >= 1,
                "round {round}: waiter {index} neither saw the completion nor was woken"
            );
        }
    }
}
