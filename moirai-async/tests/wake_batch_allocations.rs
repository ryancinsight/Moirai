//! Allocation contract for batch wake delivery.
//!
//! This binary installs a counting global allocator, so it remains isolated
//! from the ordinary async synchronization test harness. Waiter registration
//! occurs before the measured window; the window covers only the public
//! `Notify::notify_waiters` batch-grant operation.

use core::{
    future::Future,
    pin::Pin,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll, Waker},
};
use moirai_async::Notify;
use std::alloc::{GlobalAlloc, Layout, System};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every operation delegates unchanged pointers and layouts to the
// system allocator; the counter observes calls without altering allocation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is forwarded unchanged to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` and `layout` came from this delegated allocator.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the arguments are forwarded unchanged to the system
        // allocator that created `pointer`.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const WAITER_COUNT: usize = 64;
const BATCH_ALLOCATIONS: usize = 1;

#[test]
fn notify_waiters_uses_one_owned_waker_allocation() {
    let notify = Notify::new();
    let mut waiters = (0..WAITER_COUNT)
        .map(|_| notify.notified())
        .collect::<Vec<_>>();
    let mut context = Context::from_waker(Waker::noop());

    for waiter in &mut waiters {
        assert_eq!(Pin::new(waiter).poll(&mut context), Poll::Pending);
    }

    ALLOCATIONS.store(0, Ordering::Relaxed);
    notify.notify_waiters();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);

    assert_eq!(allocations, BATCH_ALLOCATIONS);
    for waiter in &mut waiters {
        assert_eq!(Pin::new(waiter).poll(&mut context), Poll::Ready(()));
    }
}
