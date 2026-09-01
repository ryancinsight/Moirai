//! Property coverage for the lock-free work-stealing deque.
//!
//! Generated operation sequences and bounded two-thread partitions,
//! complementing `deque_concurrency.rs` (fixed contention shapes) and
//! `loom_chase_lev.rs` (bounded interleavings). The concurrent case
//! terminates deterministically: the thief exits on the owner-published
//! consumed count, never on timing.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use moirai_scheduler::{ChaseLevDeque, DequeCapacity, StealResult};

use proptest::prelude::*;

fn capacity<T>(requested: usize) -> DequeCapacity<T> {
    DequeCapacity::try_from(requested).expect("generated capacity must be representable")
}

#[derive(Clone, Debug, PartialEq)]
enum DequeOp {
    Push(u64),
    Pop,
}

fn deque_ops() -> impl Strategy<Value = Vec<DequeOp>> {
    (proptest::collection::vec(any::<u64>(), 0..48), 0usize..=120).prop_map(|(values, flips)| {
        let mut ops = Vec::with_capacity(values.len() + flips);
        let mut next = values.iter().copied();
        for push_turn in (0..flips).map(|i| i % 2 == 0) {
            if push_turn {
                if let Some(v) = next.next() {
                    ops.push(DequeOp::Push(v));
                    continue;
                }
            }
            ops.push(DequeOp::Pop);
        }
        ops.extend(next.map(DequeOp::Push));
        ops
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn owner_lifo_matches_stack_model(
        init_cap_exp in 1u32..8,
        ops in deque_ops(),
    ) {
        let mut deque: ChaseLevDeque<u64> =
            ChaseLevDeque::new(capacity(1usize << init_cap_exp));
        let stealer = deque.stealer();
        let mut model: Vec<u64> = Vec::new();
        for op in ops {
            match op {
                DequeOp::Push(v) => {
                    deque.push(v);
                    model.push(v);
                }
                DequeOp::Pop => {
                    // Owner pop is LIFO; no thief runs in this scenario.
                    prop_assert_eq!(deque.pop(), model.last().copied());
                    model.pop();
                    match stealer.steal() {
                        StealResult::Empty => {}
                        other => panic!("expected Empty from a drained deque, got {other:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn concurrent_owner_and_thief_consume_exactly_once(total in 1usize..512) {
        let owned: ChaseLevDeque<usize> = ChaseLevDeque::new(capacity(4));
        let stealer = owned.stealer();
        let deque = Arc::new(Mutex::new(owned));

        let remaining = Arc::new(AtomicUsize::new(total));
        let stolen = Arc::new(Mutex::new(Vec::<usize>::new()));

        let thief = {
            let remaining = Arc::clone(&remaining);
            let stolen = Arc::clone(&stolen);
            std::thread::spawn(move || loop {
                let left = remaining.load(Ordering::Acquire);
                if left == 0 {
                    break;
                }
                match stealer.steal() {
                    StealResult::Success(v) => {
                        stolen.lock().expect("stolen mutex").push(v);
                        remaining.fetch_sub(1, Ordering::AcqRel);
                    }
                    // Lost a race or transient state; retry until the count
                    // publishes zero. No wall-clock waiting anywhere.
                    StealResult::Retry | StealResult::Empty => {
                        std::hint::spin_loop();
                    }
                }
            })
        };

        let mut popped = Vec::with_capacity(total);
        {
            let mut guard = deque.lock().expect("owner mutex");
            for v in 0..total {
                guard.push(v);
                // Interleave owner pops so both consumption paths run.
                if v % 3 == 0 && guard.pop().is_some() {
                    remaining.fetch_sub(1, Ordering::AcqRel);
                    popped.push(v);
                }
            }
            while let Some(x) = guard.pop() {
                popped.push(x);
                remaining.fetch_sub(1, Ordering::AcqRel);
            }
        }

        thief.join().expect("thief");
        prop_assert_eq!(remaining.load(Ordering::Acquire), 0);

        let mut stolen = match Arc::try_unwrap(stolen) {
            Ok(mutex) => mutex.into_inner().expect("stolen mutex"),
            Err(arc) => arc.lock().expect("stolen mutex").clone(),
        };
        stolen.extend(popped);
        stolen.sort_unstable();

        let expected: Vec<usize> = (0..total).collect();
        prop_assert_eq!(stolen, expected);
    }

    #[test]
    fn capacity_growth_preserves_lifo(init_cap_exp in 1u32..5, factor in 2usize..9) {
        let total = (1usize << init_cap_exp) * factor;
        let mut deque: ChaseLevDeque<usize> =
            ChaseLevDeque::new(capacity(1usize << init_cap_exp));
        for v in 0..total {
            deque.push(v);
        }
        for v in (0..total).rev() {
            prop_assert_eq!(deque.pop(), Some(v));
        }
        prop_assert_eq!(deque.pop(), None);
    }
}
