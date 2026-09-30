use super::chase_lev::{ChaseLevDeque, DequeCapacity, StealResult};
use super::reclaim::SharedEpochReclaim;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn capacity<T>(requested: usize) -> DequeCapacity<T> {
    DequeCapacity::try_from(requested).expect("test capacity must be representable")
}

#[test]
fn a_drained_grown_deque_returns_to_its_configured_capacity() {
    let mut deque: ChaseLevDeque<usize> = ChaseLevDeque::new(capacity(16));
    for value in 0..500 {
        deque.push(value);
    }
    assert_eq!(deque.capacity(), 512);
    assert!(deque.retired_array_count() > 0);

    let mut popped = 0;
    while deque.pop().is_some() {
        popped += 1;
    }
    assert_eq!(popped, 500);

    assert!(deque.shrink_to(capacity(16)));
    assert_eq!(deque.capacity(), 16);
    assert_eq!(deque.retired_array_count(), 0);
}

#[test]
fn shrink_is_a_no_op_at_or_below_the_target() {
    let mut deque: ChaseLevDeque<usize> = ChaseLevDeque::new(capacity(16));
    assert!(!deque.shrink_to(capacity(16)));
    assert!(!deque.shrink_to(capacity(64)));
    assert_eq!(deque.capacity(), 16);
}

#[test]
fn shrink_refuses_when_live_items_leave_no_headroom() {
    let mut deque: ChaseLevDeque<usize> = ChaseLevDeque::new(capacity(16));
    for value in 0..100 {
        deque.push(value);
    }
    let grown = deque.capacity();
    let retired = deque.retired_array_count();

    // 16 slots hold at most 14 items and still leave a push without a regrow.
    for _ in 0..85 {
        deque.pop().expect("items remain");
    }
    assert_eq!(deque.len(), 15);
    assert!(!deque.shrink_to(capacity(16)));
    assert_eq!(deque.capacity(), grown);
    assert_eq!(deque.retired_array_count(), retired);

    deque.pop().expect("items remain");
    assert!(deque.shrink_to(capacity(16)));
    assert_eq!(deque.capacity(), 16);
}

#[test]
fn shrink_keeps_live_items_in_order_for_owner_and_thieves() {
    let mut deque: ChaseLevDeque<usize> = ChaseLevDeque::new(capacity(16));
    let stealer = deque.stealer();
    for value in 0..200 {
        deque.push(value);
    }
    // Thieves take the oldest items; the owner pops the newest.
    for expected in 0..190 {
        assert_eq!(stealer.steal(), StealResult::Success(expected));
    }
    assert_eq!(deque.len(), 10);

    assert!(deque.shrink_to(capacity(16)));
    assert_eq!(deque.capacity(), 16);
    assert_eq!(deque.len(), 10);

    assert_eq!(stealer.steal(), StealResult::Success(190));
    assert_eq!(deque.pop(), Some(199));
    for value in 200..210 {
        deque.push(value);
    }
    let mut drained = Vec::new();
    while let Some(value) = deque.pop() {
        drained.push(value);
    }
    let mut expected: Vec<usize> = (191..199).chain(200..210).collect();
    expected.reverse();
    assert_eq!(drained, expected);
}

#[test]
fn shrink_after_indices_wrap_the_buffer_preserves_items() {
    let mut deque: ChaseLevDeque<usize> = ChaseLevDeque::new(capacity(16));
    let stealer = deque.stealer();
    // Advance both indices far past the capacity so live slots sit mid-buffer.
    for round in 0..40 {
        for value in 0..10 {
            deque.push(round * 10 + value);
        }
        for _ in 0..10 {
            assert!(matches!(stealer.steal(), StealResult::Success(_)));
        }
    }
    for value in 0..100 {
        deque.push(value);
    }
    for _ in 0..92 {
        deque.pop().expect("items remain");
    }
    assert!(deque.shrink_to(capacity(16)));
    let mut rest = Vec::new();
    while let Some(value) = deque.pop() {
        rest.push(value);
    }
    assert_eq!(rest, (0..8).rev().collect::<Vec<_>>());
}

#[test]
fn shrink_frees_retired_arrays_under_the_shared_epoch_policy_too() {
    let mut deque: ChaseLevDeque<usize, SharedEpochReclaim> = ChaseLevDeque::new(capacity(16));
    for value in 0..300 {
        deque.push(value);
    }
    while deque.pop().is_some() {}
    assert!(deque.retired_array_count() > 0);
    assert!(deque.shrink_to(capacity(16)));
    assert_eq!(deque.retired_array_count(), 0);
}

#[test]
fn shrink_and_growth_cycles_lose_no_item_while_thieves_steal() {
    // Miri interprets every step; a few rounds still cross grow and shrink.
    const ROUNDS: usize = if cfg!(miri) { 3 } else { 40 };
    const BURST: usize = if cfg!(miri) { 60 } else { 300 };
    let mut deque: ChaseLevDeque<usize> = ChaseLevDeque::new(capacity(16));
    let done = Arc::new(AtomicBool::new(false));

    let thieves: Vec<_> = (0..2)
        .map(|_| {
            let stealer = deque.stealer();
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                let mut taken = Vec::new();
                loop {
                    match stealer.steal() {
                        StealResult::Success(value) => taken.push(value),
                        StealResult::Retry => {}
                        StealResult::Empty => {
                            if done.load(Ordering::Acquire) {
                                break;
                            }
                            std::thread::yield_now();
                        }
                    }
                }
                taken
            })
        })
        .collect();

    let mut owned = Vec::new();
    for round in 0..ROUNDS {
        for value in 0..BURST {
            deque.push(round * BURST + value);
        }
        while let Some(value) = deque.pop() {
            owned.push(value);
        }
        deque.shrink_to(capacity(16));
    }
    done.store(true, Ordering::Release);

    let mut seen = owned;
    for thief in thieves {
        seen.extend(thief.join().expect("thief must not panic"));
    }
    seen.sort_unstable();
    assert_eq!(seen, (0..ROUNDS * BURST).collect::<Vec<_>>());
    assert_eq!(deque.capacity(), 16);
}
