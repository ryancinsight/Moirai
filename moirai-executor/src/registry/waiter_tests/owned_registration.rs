use std::sync::Arc;

use crate::counting_wake::counting_waker;

use super::super::registry::TaskRegistry;
use super::super::state::TaskState;
use super::super::token::TaskLifecycleToken;

/// Register a task and keep its lifecycle token, so the task stays pending
/// until the token completes it (a dropped unstarted token completes the task).
pub(super) fn pending_task(registry: &TaskRegistry) -> (u64, TaskLifecycleToken) {
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

    let _first_waiter = registry.register_waker(id, &first_waker).unwrap();
    let _second_waiter = registry.register_waker(id, &second_waker).unwrap();
    assert_eq!(first.wakes() + second.wakes(), 0, "nothing completed yet");

    lifecycle.start(0).complete().notify();

    assert_eq!(first.wakes(), 1, "the first waiter was overwritten");
    assert_eq!(second.wakes(), 1);
}

#[test]
fn completion_wakes_every_distinct_waiter_once() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let waiters: Vec<_> = (0..5).map(|_| counting_waker()).collect();

    let mut registrations: Vec<_> = waiters
        .iter()
        .map(|(_, waker)| registry.register_waker(id, waker).unwrap())
        .collect();
    // Re-polling updates each owned registration instead of adding history.
    for (registration, (_, waker)) in registrations.iter_mut().zip(&waiters) {
        assert!(registration.update(waker));
    }
    lifecycle.start(0).complete().notify();

    for (index, (target, _)) in waiters.iter().enumerate() {
        assert_eq!(target.wakes(), 1, "waiter {index}");
    }
}

#[test]
fn independent_waiters_with_one_caller_waker_are_each_preserved() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (target, waker) = counting_waker();

    let registrations: Vec<_> = (0..3)
        .map(|_| registry.register_waker(id, &waker).unwrap())
        .collect();
    assert_eq!(
        Arc::strong_count(&target),
        5,
        "the handle, caller waker, and three independent registrations"
    );
    lifecycle.start(0).complete().notify();

    assert_eq!(target.wakes(), 3);
    drop(registrations);
    assert_eq!(
        Arc::strong_count(&target),
        2,
        "completion releases the registry entry"
    );
}

#[test]
fn completed_tasks_reject_new_waiters_without_retaining_callbacks() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (early, early_waker) = counting_waker();
    let (early_second, early_second_waker) = counting_waker();
    let _early_waiter = registry.register_waker(id, &early_waker).unwrap();
    let _early_second_waiter = registry.register_waker(id, &early_second_waker).unwrap();
    lifecycle.start(0).complete().notify();

    let (late, late_waker) = counting_waker();
    let (late_second, late_second_waker) = counting_waker();
    assert!(registry.register_waker(id, &late_waker).is_none());
    assert!(registry.register_waker(id, &late_second_waker).is_none());
    drop((
        early_waker,
        early_second_waker,
        late_waker,
        late_second_waker,
    ));

    for target in [&early, &early_second] {
        assert_eq!(target.wakes(), 1);
        assert_eq!(Arc::strong_count(target), 1, "completion retains no waker");
    }
    for target in [&late, &late_second] {
        assert_eq!(target.wakes(), 0);
        assert_eq!(Arc::strong_count(target), 1);
    }
}

#[test]
fn waiter_ownership_outlives_state_storage_without_a_state_pointer() {
    let state = TaskState::new();
    let waiters: Vec<_> = (0..3).map(|_| counting_waker()).collect();
    let registrations: Vec<_> = waiters
        .iter()
        .map(|(_, waker)| state.register_waker(waker).unwrap())
        .collect();
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
        assert_eq!(Arc::strong_count(target), 2, "the owned waiter remains");
    }
    drop(registrations);

    for target in &targets {
        assert_eq!(target.wakes(), 0, "an unfinished task wakes nobody");
        assert_eq!(Arc::strong_count(target), 1, "a waker outlived its state");
    }
}

#[test]
fn repeated_abandonment_reuses_one_spill_slot_and_releases_each_payload() {
    const ABANDONED: usize = 1_000;
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (survivor, survivor_waker) = counting_waker();
    let survivor_waiter = registry.register_waker(id, &survivor_waker).unwrap();

    for index in 0..ABANDONED {
        let (target, waker) = counting_waker();
        let waiter = registry.register_waker(id, &waker).unwrap();
        assert_eq!(waiter.spill_slots(), 1, "abandonment {index}");
        drop(waiter);
        drop(waker);
        assert_eq!(
            Arc::strong_count(&target),
            1,
            "abandonment {index} retained its waker payload"
        );
    }

    lifecycle.start(0).complete().notify();
    assert_eq!(survivor.wakes(), 1);
    drop(survivor_waiter);
}

#[test]
fn an_empty_set_releases_its_spill_storage() {
    let registry = TaskRegistry::new();
    let (id, _lifecycle) = pending_task(&registry);
    let waiters: Vec<_> = (0..8).map(|_| counting_waker()).collect();
    let registrations: Vec<_> = waiters
        .iter()
        .map(|(_, waker)| registry.register_waker(id, waker).unwrap())
        .collect();
    assert_eq!(registrations[0].spill_slots(), 7);
    drop(registrations);

    let (_, waker) = counting_waker();
    let waiter = registry.register_waker(id, &waker).unwrap();
    assert_eq!(waiter.spill_slots(), 0);
}

#[test]
fn repoll_replaces_and_releases_the_previous_waker() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (old_target, old_waker) = counting_waker();
    let (current_target, current_waker) = counting_waker();
    let mut waiter = registry.register_waker(id, &old_waker).unwrap();

    assert!(waiter.update(&current_waker));
    assert_eq!(Arc::strong_count(&old_target), 2);
    assert_eq!(Arc::strong_count(&current_target), 3);

    lifecycle.start(0).complete().notify();
    assert_eq!(old_target.wakes(), 0);
    assert_eq!(current_target.wakes(), 1);
}
