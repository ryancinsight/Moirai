use super::chase_lev::{ChaseLevDeque, ChaseLevStealer, StealResult};
use super::reclaim::{
    DeferredAccessGuard, DeferredReclaim, DeferredState, DequeReclaimState, SharedEpochReclaim,
    SharedEpochState,
};
use super::split::SplitDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

static_assertions::assert_impl_all!(ChaseLevDeque<usize>: Send);
static_assertions::assert_not_impl_any!(ChaseLevDeque<usize>: Clone, Sync);
static_assertions::assert_impl_all!(ChaseLevStealer<usize>: Clone, Send, Sync);

#[test]
fn test_chase_lev_deque_basic_operations() {
    let mut deque: ChaseLevDeque<i32> = ChaseLevDeque::new(16);

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
    let mut deque: ChaseLevDeque<i32> = ChaseLevDeque::new(16);
    let stealer = deque.stealer();

    for i in 1..=5 {
        deque.push(i);
    }

    assert_eq!(stealer.steal(), StealResult::Success(1));
    assert_eq!(stealer.steal(), StealResult::Success(2));

    assert_eq!(deque.pop(), Some(5));
    assert_eq!(deque.pop(), Some(4));

    assert_eq!(stealer.steal(), StealResult::Success(3));

    assert_eq!(stealer.steal(), StealResult::Empty);
    assert_eq!(deque.pop(), None);
}

#[test]
fn chase_lev_deque_resizes_without_per_item_heap_nodes() {
    let mut deque: ChaseLevDeque<usize> = ChaseLevDeque::new(2);
    let stealer = deque.stealer();

    for value in 0..40 {
        deque.push(value);
    }

    assert_eq!(stealer.steal(), StealResult::Success(0));
    assert_eq!(stealer.steal(), StealResult::Success(1));
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
fn chase_lev_deque_defers_retired_arrays_until_final_endpoint_drop() {
    let mut deque: ChaseLevDeque<usize> = ChaseLevDeque::new(2);

    for value in 0..40 {
        deque.push(value);
    }

    assert_eq!(deque.retired_array_count(), 2);

    let mut observed = Vec::new();
    while let Some(value) = deque.pop() {
        observed.push(value);
    }

    assert_eq!(observed.len(), 40);
    assert_eq!(observed.iter().sum::<usize>(), (0..40).sum::<usize>());
}

#[test]
fn chase_lev_deque_reclamation_policies_are_static() {
    assert_eq!(std::mem::size_of::<DeferredReclaim>(), 0);
    assert_eq!(std::mem::size_of::<DeferredState>(), 0);
    assert_eq!(std::mem::size_of::<DeferredAccessGuard>(), 0);
    assert_eq!(std::mem::size_of::<SharedEpochReclaim>(), 0);
    assert_eq!(
        std::mem::size_of::<SharedEpochState>(),
        std::mem::size_of::<AtomicUsize>()
    );
}

#[test]
fn chase_lev_deque_shared_epoch_reclaim_waits_for_active_access() {
    let mut deque: ChaseLevDeque<usize, SharedEpochReclaim> = ChaseLevDeque::new(2);

    for value in 0..40 {
        deque.push(value);
    }

    assert_eq!(deque.retired_array_count(), 2);

    let guard = deque.inner.reclaim.enter();
    assert!(!deque.try_reclaim_shared(SharedEpochReclaim));
    drop(guard);

    assert!(deque.try_reclaim_shared(SharedEpochReclaim));
    assert_eq!(deque.retired_array_count(), 0);

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
        let mut deque: ChaseLevDeque<DropProbe> = ChaseLevDeque::new(2);
        let stealer = deque.stealer();
        for _ in 0..40 {
            deque.push(DropProbe(Arc::clone(&drops)));
        }

        let mut stolen = 0;
        while stolen < 10 {
            match stealer.steal() {
                StealResult::Success(item) => {
                    drop(item);
                    stolen += 1;
                }
                StealResult::Retry => std::hint::spin_loop(),
                StealResult::Empty => panic!("ten queued items must remain stealable"),
            }
        }

        assert_eq!(drops.load(Ordering::Relaxed), 10);
    }

    assert_eq!(drops.load(Ordering::Relaxed), 40);
}

#[test]
fn chase_lev_stealer_keeps_pending_storage_alive_after_owner_drop() {
    let mut owner: ChaseLevDeque<usize> = ChaseLevDeque::new(2);
    let stealer = owner.stealer();
    for value in 0..40 {
        owner.push(value);
    }
    drop(owner);

    let mut observed = Vec::new();
    loop {
        match stealer.steal() {
            StealResult::Success(value) => observed.push(value),
            StealResult::Retry => continue,
            StealResult::Empty => break,
        }
    }
    assert_eq!(observed, (0..40).collect::<Vec<_>>());
}

#[test]
fn chase_lev_batch_drops_unconsumed_tail_exactly_once() {
    struct DropProbe(Arc<AtomicUsize>);
    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));
    let mut owner: ChaseLevDeque<DropProbe> = ChaseLevDeque::new(2);
    let stealer = owner.stealer();
    for _ in 0..10 {
        owner.push(DropProbe(Arc::clone(&drops)));
    }
    let mut batch = loop {
        match stealer.steal_batch() {
            StealResult::Success(batch) => break batch,
            StealResult::Retry => std::hint::spin_loop(),
            StealResult::Empty => panic!("ten queued items must remain stealable"),
        }
    };
    drop(batch.next().expect("successful batch is non-empty"));
    drop(batch);
    drop(stealer);
    drop(owner);
    assert_eq!(drops.load(Ordering::Relaxed), 10);
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
    let mut deque: ChaseLevDeque<usize> = ChaseLevDeque::new(4);
    let stealer = deque.stealer();

    // Artificially initialize bottom and top to near overflow (isize::MAX)
    deque.inner.top.store(isize::MAX - 2, Ordering::Relaxed);
    deque.inner.bottom.store(isize::MAX - 2, Ordering::Relaxed);

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
    assert_eq!(stealer.steal(), StealResult::Success(10));
    assert_eq!(stealer.steal(), StealResult::Success(20));

    // Pop elements from the bottom (should be 70, then 60, then 50, then 40)
    assert_eq!(deque.pop(), Some(70));
    assert_eq!(deque.pop(), Some(60));
    assert_eq!(deque.pop(), Some(50));
    assert_eq!(deque.pop(), Some(40));

    assert_eq!(deque.pop(), None);
    assert_eq!(stealer.steal(), StealResult::Empty);
    assert!(deque.is_empty());
}
