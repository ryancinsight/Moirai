//! Allocation contract for polling a spawned async task.
//!
//! This binary installs a counting global allocator, so it stays isolated from
//! the ordinary async test harness. A task that yields `YIELDS` times is driven
//! to completion twice with different yield counts; every poll beyond the first
//! is one extra executor poll, so the allocation difference between the runs,
//! divided by the yield difference, is the per-poll allocation cost of the
//! executor's poll path (waker mint, run-queue re-entry, reactor iteration).

use moirai_async::AsyncExecutor;
use std::alloc::{GlobalAlloc, Layout, System};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

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

/// Re-wakes itself and returns `Pending` for `remaining` polls, then completes.
struct YieldThenReady {
    remaining: usize,
}

impl Future for YieldThenReady {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        if self.remaining == 0 {
            return Poll::Ready(());
        }
        self.remaining -= 1;
        context.waker().wake_by_ref();
        Poll::Pending
    }
}

fn allocations_to_complete(yields: usize) -> usize {
    let executor = AsyncExecutor::new().expect("a fresh AsyncExecutor must build");
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    executor.block_on(YieldThenReady { remaining: yields });
    ALLOCATIONS.load(Ordering::Relaxed) - before
}

#[test]
fn polling_a_yielding_task_allocates_nothing_per_poll() {
    const SHORT: usize = 64;
    const LONG: usize = 1_024;

    // Warm lazily initialized process state (reactor, thread-locals) outside
    // the measured windows.
    allocations_to_complete(1);

    let short = allocations_to_complete(SHORT);
    let long = allocations_to_complete(LONG);

    assert_eq!(
        long.saturating_sub(short),
        0,
        "{} extra polls cost {} extra allocations ({} at {SHORT} yields, {} at {LONG})",
        LONG - SHORT,
        long.saturating_sub(short),
        short,
        long,
    );
}
