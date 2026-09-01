use core::future::Future;
use core::marker::PhantomPinned;
use core::pin::Pin;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::task::{Context, Poll};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use futures::task::{waker, ArcWake};
use futures::{Stream, StreamExt};

use super::wake::WakeBlock;
use super::{retained_buffered, retained_unordered, RetainedSlots, SlotKey};

struct PendingOnce<T> {
    value: Option<T>,
    pending: bool,
}

impl<T> Unpin for PendingOnce<T> {}

impl<T> Future for PendingOnce<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.pending {
            this.pending = false;
            context.waker().wake_by_ref();
            Poll::Pending
        } else {
            Poll::Ready(
                this.value
                    .take()
                    .expect("pending-once future polled after completion"),
            )
        }
    }
}

fn pending_once<T>(value: T) -> PendingOnce<T> {
    PendingOnce {
        value: Some(value),
        pending: true,
    }
}

struct AddressCheckingFuture {
    expected_address: *const Self,
    value: usize,
    pending: bool,
    _pin: PhantomPinned,
}

impl AddressCheckingFuture {
    const fn new(value: usize) -> Self {
        Self {
            expected_address: core::ptr::null(),
            value,
            pending: true,
            _pin: PhantomPinned,
        }
    }
}

impl Future for AddressCheckingFuture {
    type Output = usize;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let address = self.as_ref().get_ref() as *const Self;
        // SAFETY: the future is pinned for this poll. The implementation
        // mutates scalar state only and never moves the `PhantomPinned` value
        // or exposes an unpinned reference.
        let this = unsafe { self.get_unchecked_mut() };
        if this.expected_address.is_null() {
            this.expected_address = address;
            this.pending = false;
            context.waker().wake_by_ref();
            Poll::Pending
        } else {
            assert_eq!(this.expected_address, address);
            Poll::Ready(this.value)
        }
    }
}

struct CrossThreadWake {
    value: Option<usize>,
    started: bool,
}

impl Future for CrossThreadWake {
    type Output = usize;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.started {
            return Poll::Ready(
                self.value
                    .take()
                    .expect("cross-thread future polled after completion"),
            );
        }
        self.started = true;
        let waker = context.waker().clone();
        std::thread::spawn(move || waker.wake());
        Poll::Pending
    }
}

struct DropFuture {
    drops: Arc<AtomicUsize>,
    panic_on_poll: bool,
}

impl Future for DropFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
        assert!(!self.panic_on_poll, "poll failure sentinel");
        Poll::Pending
    }
}

impl Drop for DropFuture {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct ParentWake {
    wakes: AtomicUsize,
}

impl ArcWake for ParentWake {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.wakes.fetch_add(1, Ordering::SeqCst);
    }
}

struct PanicWake;

impl ArcWake for PanicWake {
    fn wake_by_ref(_arc_self: &Arc<Self>) {
        panic!("parent wake failure sentinel");
    }
}

#[test]
fn ordered_slots_preserve_values_across_pending_refills() {
    let stream = futures::stream::iter((0..37).map(pending_once));
    let values = futures::executor::block_on(retained_buffered(stream, 5).collect::<Vec<_>>());
    assert_eq!(values, (0..37).collect::<Vec<_>>());
}

#[test]
fn exact_empty_stream_does_not_reserve_a_large_limit() {
    let stream = futures::stream::empty::<PendingOnce<usize>>();
    let values =
        futures::executor::block_on(retained_buffered(stream, usize::MAX).collect::<Vec<_>>());
    assert!(values.is_empty());
}

#[test]
fn exact_single_item_clamps_a_large_limit() {
    let stream = futures::stream::iter([pending_once(17)]);
    let values =
        futures::executor::block_on(retained_buffered(stream, usize::MAX).collect::<Vec<_>>());
    assert_eq!(values, [17]);
}

#[test]
fn unknown_single_item_grows_only_after_admission() {
    let mut future = Some(pending_once(23));
    let stream = futures::stream::poll_fn(move |_| Poll::Ready(future.take()));
    let values =
        futures::executor::block_on(retained_buffered(stream, usize::MAX).collect::<Vec<_>>());
    assert_eq!(values, [23]);
}

#[test]
fn unknown_stream_preserves_values_across_geometric_blocks() {
    let mut next = 0_u64;
    let stream = futures::stream::poll_fn(move |_| {
        if next == 9 {
            Poll::Ready(None)
        } else {
            let value = next;
            next += 1;
            Poll::Ready(Some(pending_once(value)))
        }
    });

    let values =
        futures::executor::block_on(retained_buffered(stream, usize::MAX).collect::<Vec<_>>());

    assert_eq!(values, (0..9).collect::<Vec<_>>());
}

#[test]
fn repeated_tail_slot_refill_uses_one_word_and_head_probe() {
    const CAPACITY: usize = 64;
    const REPLACEMENTS: usize = 128;

    let mut slots = RetainedSlots::new(CAPACITY, CAPACITY, true);
    for index in 0..CAPACITY {
        slots.insert(core::future::ready(index), index);
    }
    let tail = SlotKey {
        block: 0,
        slot: CAPACITY - 1,
    };
    let baseline_probes = slots.vacancy_probe_counts();
    let mut expected = CAPACITY - 1;

    for replacement in 0..REPLACEMENTS {
        assert_eq!(slots.poll(tail), Poll::Ready((expected, expected)));
        expected = CAPACITY + replacement;
        slots.insert(core::future::ready(expected), expected);
    }

    let final_probes = slots.vacancy_probe_counts();
    assert_eq!(
        final_probes.0 - baseline_probes.0,
        REPLACEMENTS,
        "each refill must inspect only the first vacancy bitmap word"
    );
    assert_eq!(
        final_probes.1 - baseline_probes.1,
        REPLACEMENTS,
        "each refill must inspect only the intrusive vacancy head"
    );
}

#[test]
fn unordered_slots_complete_every_value_once() {
    let stream = futures::stream::iter((0..37).map(pending_once));
    let mut values = futures::executor::block_on(retained_unordered(stream, 5).collect::<Vec<_>>());
    values.sort_unstable();
    assert_eq!(values, (0..37).collect::<Vec<_>>());
}

#[test]
fn retained_slots_do_not_move_non_unpin_futures() {
    let stream = futures::stream::iter((0..37).map(AddressCheckingFuture::new));
    let values = futures::executor::block_on(retained_buffered(stream, 5).collect::<Vec<_>>());
    assert_eq!(values, (0..37).collect::<Vec<_>>());
}

#[test]
fn retained_slots_route_cross_thread_wakes() {
    let stream = futures::stream::iter((0..37).map(|value| CrossThreadWake {
        value: Some(value),
        started: false,
    }));
    let values = futures::executor::block_on(retained_buffered(stream, 5).collect::<Vec<_>>());
    assert_eq!(values, (0..37).collect::<Vec<_>>());
}

#[test]
fn cloned_slot_wakers_keep_the_shared_block_alive() {
    let parent = Arc::new(ParentWake {
        wakes: AtomicUsize::new(0),
    });
    let parent_waker = waker(Arc::clone(&parent));
    let block = WakeBlock::new_root(1);
    block.register(&parent_waker);
    let owner = Arc::downgrade(&block);
    let first = WakeBlock::waker(&block, 0);
    let last = first.clone();

    drop(block);
    assert!(owner.upgrade().is_some());
    first.wake_by_ref();
    assert_eq!(parent.wakes.load(Ordering::SeqCst), 1);
    drop(first);
    let retained = owner
        .upgrade()
        .expect("cloned waker must retain the shared wake block");
    retained.register(&parent_waker);
    drop(retained);

    last.wake();
    assert_eq!(parent.wakes.load(Ordering::SeqCst), 2);
    assert!(owner.upgrade().is_none());
}

#[test]
fn recreated_slot_wakers_preserve_identity() {
    let block = WakeBlock::new_root(2);
    let first = WakeBlock::waker(&block, 0);
    let recreated = WakeBlock::waker(&block, 0);
    let other = WakeBlock::waker(&block, 1);

    assert!(first.will_wake(&recreated));
    assert!(!first.will_wake(&other));
}

#[test]
fn consuming_wake_releases_ownership_when_parent_panics() {
    let parent_waker = waker(Arc::new(PanicWake));
    let block = WakeBlock::new_root(1);
    block.register(&parent_waker);
    let owner = Arc::downgrade(&block);
    let slot_waker = WakeBlock::waker(&block, 0);
    drop(block);

    let result = catch_unwind(AssertUnwindSafe(|| slot_waker.wake()));

    assert!(result.is_err());
    assert!(owner.upgrade().is_none());
}

#[test]
fn dropping_ordered_slots_drops_each_constructed_future_once() {
    let drops = Arc::new(AtomicUsize::new(0));
    let stream = futures::stream::iter((0..11).map({
        let drops = Arc::clone(&drops);
        move |_| DropFuture {
            drops: Arc::clone(&drops),
            panic_on_poll: false,
        }
    }));
    let mut buffered = retained_buffered(stream, 4);
    assert!(futures::executor::block_on(futures::future::poll_fn(
        |context| {
            assert!(Pin::new(&mut buffered).poll_next(context).is_pending());
            Poll::Ready(true)
        }
    )));
    drop(buffered);
    assert_eq!(drops.load(Ordering::SeqCst), 4);
}

#[test]
fn poll_panic_drops_all_initialized_slots_once() {
    let drops = Arc::new(AtomicUsize::new(0));
    let stream = futures::stream::iter((0..4).map({
        let drops = Arc::clone(&drops);
        move |index| DropFuture {
            drops: Arc::clone(&drops),
            panic_on_poll: index == 0,
        }
    }));
    let result = catch_unwind(AssertUnwindSafe(|| {
        futures::executor::block_on(retained_unordered(stream, 4).collect::<Vec<_>>())
    }));
    assert!(result.is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 4);
}
