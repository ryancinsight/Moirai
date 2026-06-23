use super::block_based::BlockBasedDeque;
use super::chase_lev::{ChaseLevDeque, StealResult};
use super::reclaim::{
    DequeReclaimState, QuiescentAccessGuard, QuiescentReclaim, QuiescentState, SharedEpochReclaim,
    SharedEpochState,
};
use super::split::SplitDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[test]
fn test_chase_lev_deque_basic_operations() {
    let deque: ChaseLevDeque<i32> = ChaseLevDeque::new(16);

    deque.push(1);
    deque.push(2);
    deque.push(3);

    assert_eq!(deque.len(), 3);
    assert!(!deque.is_empty());

    assert_eq!(deque.pop(), Some(3));
    assert_eq!(deque.pop(), Some(2));
    assert_eq!(deque.pop(), Some(1));
    assert_eq!(deque.pop(), None);

    assert!(deque.is_empty());
}

#[test]
fn test_chase_lev_deque_steal() {
    let deque: ChaseLevDeque<i32> = ChaseLevDeque::new(16);

    for i in 1..=5 {
        deque.push(i);
    }

    assert_eq!(deque.steal(), StealResult::Success(1));
    assert_eq!(deque.steal(), StealResult::Success(2));

    assert_eq!(deque.pop(), Some(5));
    assert_eq!(deque.pop(), Some(4));

    assert_eq!(deque.steal(), StealResult::Success(3));

    assert_eq!(deque.steal(), StealResult::Empty);
    assert_eq!(deque.pop(), None);
}

#[test]
fn chase_lev_deque_resizes_without_per_item_heap_nodes() {
    let deque: ChaseLevDeque<usize> = ChaseLevDeque::new(2);

    for value in 0..40 {
        deque.push(value);
    }

    assert_eq!(deque.steal(), StealResult::Success(0));
    assert_eq!(deque.steal(), StealResult::Success(1));
    assert_eq!(deque.pop(), Some(39));
    assert_eq!(deque.pop(), Some(38));

    let mut remaining = Vec::new();
    while let Some(value) = deque.pop() {
        remaining.push(value);
    }

    assert_eq!(remaining.len(), 36);
    assert_eq!(remaining.iter().sum::<usize>(), (2..=37).sum::<usize>());
}

#[test]
fn chase_lev_deque_reclaims_retired_arrays_after_quiescence() {
    let mut deque: ChaseLevDeque<usize> = ChaseLevDeque::new(2);

    for value in 0..40 {
        deque.push(value);
    }

    assert!(
        !deque.retired_arrays.lock().unwrap().is_empty(),
        "resize must retire at least one backing array"
    );

    deque.reclaim_memory(QuiescentReclaim);

    assert_eq!(deque.retired_arrays.lock().unwrap().len(), 0);

    let mut observed = Vec::new();
    while let Some(value) = deque.pop() {
        observed.push(value);
    }

    assert_eq!(observed.len(), 40);
    assert_eq!(observed.iter().sum::<usize>(), (0..40).sum::<usize>());
}

#[test]
fn chase_lev_deque_reclamation_policies_are_static() {
    assert_eq!(std::mem::size_of::<QuiescentReclaim>(), 0);
    assert_eq!(std::mem::size_of::<QuiescentState>(), 0);
    assert_eq!(std::mem::size_of::<QuiescentAccessGuard>(), 0);
    assert_eq!(std::mem::size_of::<SharedEpochReclaim>(), 0);
    assert_eq!(
        std::mem::size_of::<SharedEpochState>(),
        std::mem::size_of::<AtomicUsize>()
    );
}

#[test]
fn chase_lev_deque_shared_epoch_reclaim_waits_for_active_access() {
    let deque: ChaseLevDeque<usize, SharedEpochReclaim> = ChaseLevDeque::new(2);

    for value in 0..40 {
        deque.push(value);
    }

    assert!(
        !deque.retired_arrays.lock().unwrap().is_empty(),
        "resize must retire at least one backing array"
    );

    let guard = deque.reclaim.enter();
    assert!(!deque.try_reclaim_shared(SharedEpochReclaim));
    drop(guard);

    assert!(deque.try_reclaim_shared(SharedEpochReclaim));
    assert_eq!(deque.retired_arrays.lock().unwrap().len(), 0);

    let mut observed = Vec::new();
    while let Some(value) = deque.pop() {
        observed.push(value);
    }

    assert_eq!(observed.len(), 40);
    assert_eq!(observed.iter().sum::<usize>(), (0..40).sum::<usize>());
}

#[test]
fn chase_lev_deque_drops_each_inline_item_once() {
    struct DropProbe(Arc<AtomicUsize>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));

    {
        let deque: ChaseLevDeque<DropProbe> = ChaseLevDeque::new(2);
        for _ in 0..40 {
            deque.push(DropProbe(Arc::clone(&drops)));
        }

        for _ in 0..10 {
            match deque.steal() {
                StealResult::Success(item) => drop(item),
                StealResult::Empty | StealResult::Retry => {
                    panic!("expected successful steal")
                }
            }
        }

        assert_eq!(drops.load(Ordering::Relaxed), 10);
    }

    assert_eq!(drops.load(Ordering::Relaxed), 40);
}

#[test]
fn test_block_based_deque_basic_operations() {
    let deque: BlockBasedDeque<i32> = BlockBasedDeque::new();

    deque.push(1);
    deque.push(2);
    deque.push(3);

    assert_eq!(deque.len(), 3);
    assert!(!deque.is_empty());

    assert_eq!(deque.pop(), Some(3));
    assert_eq!(deque.pop(), Some(2));
    assert_eq!(deque.pop(), Some(1));
    assert_eq!(deque.pop(), None);

    assert!(deque.is_empty());
}

#[test]
fn test_block_based_deque_steal() {
    let deque: BlockBasedDeque<i32> = BlockBasedDeque::new();

    for i in 1..=5 {
        deque.push(i);
    }

    assert_eq!(deque.steal(), StealResult::Success(1));
    assert_eq!(deque.steal(), StealResult::Success(2));

    assert_eq!(deque.pop(), Some(5));
    assert_eq!(deque.pop(), Some(4));

    assert_eq!(deque.steal(), StealResult::Success(3));

    assert_eq!(deque.steal(), StealResult::Empty);
    assert_eq!(deque.pop(), None);
}

#[test]
fn test_block_based_deque_bulk_steal() {
    let deque: BlockBasedDeque<i32> = BlockBasedDeque::new();

    for i in 1..=10 {
        deque.push(i);
    }

    let mut stolen = Vec::new();
    let first = deque.steal_batch_with(|item| stolen.push(item));

    assert_eq!(first, StealResult::Success(1));
    assert_eq!(stolen, vec![2, 3, 4, 5]);

    assert_eq!(deque.pop(), Some(10));
    assert_eq!(deque.pop(), Some(9));
}

#[test]
fn test_block_based_deque_drops_each_item_once() {
    struct DropProbe(Arc<AtomicUsize>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));

    {
        let deque: BlockBasedDeque<DropProbe> = BlockBasedDeque::new();
        for _ in 0..100 {
            deque.push(DropProbe(Arc::clone(&drops)));
        }

        for _ in 0..20 {
            match deque.steal() {
                StealResult::Success(item) => drop(item),
                StealResult::Empty | StealResult::Retry => {
                    panic!("expected successful steal")
                }
            }
        }

        assert_eq!(drops.load(Ordering::Relaxed), 20);
    }

    assert_eq!(drops.load(Ordering::Relaxed), 100);
}

#[test]
fn test_block_based_deque_multithreaded() {
    use std::thread;

    let deque = Arc::new(BlockBasedDeque::new());
    let num_items = 1000;

    for i in 0..num_items {
        deque.push(i);
    }

    let deque_clone1 = deque.clone();
    let handle1 = thread::spawn(move || {
        let mut stolen = Vec::new();
        for _ in 0..(num_items / 2) {
            if let StealResult::Success(item) = deque_clone1.steal() {
                stolen.push(item);
            }
        }
        stolen
    });

    let deque_clone2 = deque.clone();
    let handle2 = thread::spawn(move || {
        let mut stolen = Vec::new();
        for _ in 0..(num_items / 2) {
            if let StealResult::Success(item) = deque_clone2.steal() {
                stolen.push(item);
            }
        }
        stolen
    });

    let mut popped = Vec::new();
    while let Some(item) = deque.pop() {
        popped.push(item);
    }

    let stolen1 = handle1.join().unwrap();
    let stolen2 = handle2.join().unwrap();

    let total_processed = stolen1.len() + stolen2.len() + popped.len();
    assert_eq!(total_processed, num_items);

    let mut all_items = Vec::new();
    all_items.extend(stolen1);
    all_items.extend(stolen2);
    all_items.extend(popped);
    all_items.sort();
    for (i, &item) in all_items.iter().enumerate() {
        assert_eq!(item, i);
    }
}

#[test]
fn test_split_deque_basic_operations() {
    let deque: SplitDeque<i32, 8> = SplitDeque::new();

    deque.push(1);
    deque.push(2);
    deque.push(3);

    assert_eq!(deque.len(), 3);
    assert!(!deque.is_empty());

    assert_eq!(deque.pop(), Some(3));
    assert_eq!(deque.pop(), Some(2));
    assert_eq!(deque.pop(), Some(1));
    assert_eq!(deque.pop(), None);

    assert!(deque.is_empty());
}

#[test]
fn test_split_deque_threshold_offloading() {
    let deque: SplitDeque<i32, 4> = SplitDeque::new();

    deque.push(1);
    deque.push(2);
    deque.push(3);
    deque.push(4);

    assert_eq!(deque.steal(), StealResult::Empty);

    deque.push(5);

    assert_eq!(deque.steal(), StealResult::Success(1));
    assert_eq!(deque.steal(), StealResult::Success(2));
    assert_eq!(deque.steal(), StealResult::Empty);

    assert_eq!(deque.pop(), Some(5));
    assert_eq!(deque.pop(), Some(4));
    assert_eq!(deque.pop(), Some(3));
    assert_eq!(deque.pop(), None);
}

#[test]
fn test_chase_lev_deque_index_wrapping() {
    let deque: ChaseLevDeque<usize> = ChaseLevDeque::new(4);

    // Artificially initialize bottom and top to near overflow (isize::MAX)
    deque.top.store(isize::MAX - 2, Ordering::Relaxed);
    deque.bottom.store(isize::MAX - 2, Ordering::Relaxed);

    // Push 3 items. bottom will transition:
    // isize::MAX - 2 -> isize::MAX - 1 -> isize::MAX -> isize::MIN
    deque.push(10);
    deque.push(20);
    deque.push(30);

    assert_eq!(deque.len(), 3);

    // Pop the last element (should be 30)
    assert_eq!(deque.pop(), Some(30));
    assert_eq!(deque.len(), 2);

    // Now, push more elements to force a resize.
    // The capacity is 4. Currently we have 2 elements.
    // Push 4 more elements.
    deque.push(40);
    deque.push(50);
    deque.push(60);
    deque.push(70);

    // The deque should resize successfully and retain all 6 elements.
    assert_eq!(deque.len(), 6);

    // Steal elements from the top (should be 10, then 20)
    assert_eq!(deque.steal(), StealResult::Success(10));
    assert_eq!(deque.steal(), StealResult::Success(20));

    // Pop elements from the bottom (should be 70, then 60, then 50, then 40)
    assert_eq!(deque.pop(), Some(70));
    assert_eq!(deque.pop(), Some(60));
    assert_eq!(deque.pop(), Some(50));
    assert_eq!(deque.pop(), Some(40));

    assert_eq!(deque.pop(), None);
    assert_eq!(deque.steal(), StealResult::Empty);
    assert!(deque.is_empty());
}
