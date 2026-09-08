//! Queue behaviour: priority order, steal retry policy, and plane capacities.

#![allow(clippy::unwrap_used, reason = "test scope")]

use core::mem::size_of;
use std::{
    cell::Cell,
    sync::{Arc, Mutex},
};

use super::steal::{STEAL_SPINS_BEFORE_YIELD, steal_after_contention_with};
use super::worker::WorkerQueues;
use crate::schedule::job::ScheduledJob;
use moirai_core::Priority;
use moirai_scheduler::{DequeCapacity, StealResult};

const TEST_INJECTOR_CAPACITY: usize = 8;

#[test]
fn steal_contention_spins_yields_and_preserves_victim_priority() {
    let attempts = Cell::new(0usize);
    let spins = Cell::new(0usize);
    let yields = Cell::new(0usize);
    let retries = 2 * (STEAL_SPINS_BEFORE_YIELD + 1);

    let result = steal_after_contention_with(
        || {
            let attempt = attempts.get();
            attempts.set(attempt + 1);
            if attempt < retries {
                StealResult::Retry
            } else {
                StealResult::Success(7usize)
            }
        },
        || spins.set(spins.get() + 1),
        || yields.set(yields.get() + 1),
    );

    assert_eq!(result, Some(7));
    assert_eq!(attempts.get(), retries + 1);
    assert_eq!(spins.get(), 2 * STEAL_SPINS_BEFORE_YIELD);
    assert_eq!(yields.get(), 2);
}

fn local_capacity(requested: usize) -> DequeCapacity<ScheduledJob> {
    DequeCapacity::try_from(requested).expect("test capacity must be representable")
}

#[test]
fn injector_payload_uses_seventeen_machine_words() {
    type InjectorPayload = (Priority, ScheduledJob);
    let expected_payload_bytes = 17 * size_of::<usize>();

    assert_eq!(size_of::<InjectorPayload>(), expected_payload_bytes);
    assert_eq!(size_of::<Option<InjectorPayload>>(), expected_payload_bytes);
}

#[test]
fn worker_queue_pops_highest_priority_first() {
    let observed = Arc::new(Mutex::new(Vec::new()));
    let (mut owner, queues) = WorkerQueues::new(TEST_INJECTOR_CAPACITY, local_capacity(256));

    for (priority, value) in [(Priority::Low, 1), (Priority::Critical, 2)] {
        let observed = Arc::clone(&observed);
        let () = queues
            .try_push_external(
                priority,
                ScheduledJob::new(move |_| {
                    observed.lock().unwrap().push(value);
                }),
            )
            .map_or((), |_| panic!("test queue has capacity"));
    }

    owner.pop_local().unwrap().execute(0);
    owner.pop_local().unwrap().execute(0);

    assert_eq!(*observed.lock().unwrap(), vec![2, 1]);
    assert_eq!(queues.len(), 0);
}

#[test]
fn injector_uses_configured_capacity() {
    let (_owner, queues) = WorkerQueues::new(TEST_INJECTOR_CAPACITY, local_capacity(256));
    assert_eq!(queues.injector_capacity(), TEST_INJECTOR_CAPACITY);
}

#[test]
fn local_queues_use_the_normalized_initial_capacity() {
    let (_owner, queues) = WorkerQueues::new(TEST_INJECTOR_CAPACITY, local_capacity(17));

    assert_eq!(
        queues.local_queue_capacities()[Priority::default().index()],
        32
    );
}

#[test]
fn only_the_default_priority_plane_carries_the_configured_capacity() {
    // Retained local storage is `priority levels x capacity`, but a
    // consumer that never sets a priority uses one plane. The other three
    // start at the minimum and grow on push, so an unused plane costs
    // 2,048 bytes rather than 16,384 at the 128-slot default.
    let (_owner, queues) = WorkerQueues::new(TEST_INJECTOR_CAPACITY, local_capacity(128));
    let capacities = queues.local_queue_capacities();
    let minimum = DequeCapacity::<ScheduledJob>::minimum().get();

    for (plane, capacity) in capacities.iter().copied().enumerate() {
        if plane == Priority::default().index() {
            assert_eq!(capacity, 128, "default plane keeps the configured capacity");
        } else {
            assert_eq!(capacity, minimum, "plane {plane} starts at the minimum");
        }
    }
    assert!(
        minimum < 128,
        "the minimum must actually be smaller, or this policy saves nothing"
    );
}

#[test]
fn a_non_default_plane_grows_past_its_minimum_initial_capacity() {
    // The saving is only sound because the deque grows on the owner's
    // push. Drive a minimum-capacity plane well past its initial slots
    // through the injector and require every job back.
    let minimum = DequeCapacity::<ScheduledJob>::minimum().get();
    let count = minimum * 4;
    let observed = Arc::new(Mutex::new(Vec::new()));
    let (mut owner, queues) = WorkerQueues::new(count * 2, local_capacity(128));

    for value in 0..count {
        let sink = Arc::clone(&observed);
        let () = queues
            .try_push_external(
                Priority::Critical,
                ScheduledJob::new(move |_| sink.lock().unwrap().push(value)),
            )
            .map_or((), |_| panic!("injector sized for the whole burst"));
    }

    let mut drained = 0;
    while let Some(job) = owner.pop_local() {
        job.execute(0);
        drained += 1;
    }

    assert_eq!(
        drained, count,
        "every job queued past the minimum initial capacity ran"
    );
    let mut values = observed.lock().unwrap().clone();
    values.sort_unstable();
    assert_eq!(values, (0..count).collect::<Vec<_>>());
}

#[test]
fn injector_round_trips_through_external_push() {
    // The reduced-capacity injector still enqueues and drains: an external
    // push lands in the injector and pops out via pop_local's drain path.
    let observed = Arc::new(Mutex::new(Vec::new()));
    let (mut owner, queues) = WorkerQueues::new(TEST_INJECTOR_CAPACITY, local_capacity(256));

    for (priority, value) in [(Priority::Normal, 7), (Priority::Critical, 9)] {
        let observed = Arc::clone(&observed);
        let () = queues
            .try_push_external(
                priority,
                ScheduledJob::new(move |_| {
                    observed.lock().unwrap().push(value);
                }),
            )
            .map_or((), |_| panic!("test queue has capacity"));
    }

    // Critical drains ahead of Normal once moved into the local queues.
    owner.pop_local().unwrap().execute(0);
    owner.pop_local().unwrap().execute(0);

    assert_eq!(*observed.lock().unwrap(), vec![9, 7]);
    assert_eq!(queues.len(), 0);
}

#[test]
fn full_injector_returns_and_drops_rejected_job_once() {
    let (_owner, queues) = WorkerQueues::new(TEST_INJECTOR_CAPACITY, local_capacity(256));
    for _ in 0..TEST_INJECTOR_CAPACITY {
        let () = queues
            .try_push_external(Priority::Normal, ScheduledJob::new(|_| {}))
            .map_or((), |_| panic!("capacity-sized admission must succeed"));
    }

    let capture = Arc::new(());
    let rejected_capture = Arc::clone(&capture);
    let rejected = queues
        .try_push_external(
            Priority::Normal,
            ScheduledJob::new(move |_| drop(rejected_capture)),
        )
        .expect("one job beyond capacity must be rejected");

    assert_eq!(queues.len(), TEST_INJECTOR_CAPACITY);
    assert_eq!(Arc::strong_count(&capture), 2);
    drop(rejected);
    assert_eq!(Arc::strong_count(&capture), 1);
}
