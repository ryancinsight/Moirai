//! Allocation contract for polling a spawned async task.
//!
//! This binary installs the mnemosyne per-thread counting allocator,
//! so it stays isolated from the ordinary async test harness. A task
//! that yields `YIELDS` times is driven to completion twice with
//! different yield counts; every poll beyond the first is one extra
//! executor poll, so the allocation difference between the runs,
//! divided by the yield difference, is the per-poll allocation cost
//! of the executor's poll path (waker mint, run-queue re-entry,
//! reactor iteration).
//!
//! The counter is per-thread. `block_on` polls on the calling thread
//! (`moirai-async/src/executor/core.rs`), so the calling thread's
//! count is exactly the poll path's cost and excludes every other
//! thread. A process-wide counter would also book the libtest
//! harness thread, which allocates inside the window at an
//! unpredictable rate, so the longer `LONG` window would read more
//! than the shorter `SHORT` one and the assertion would be flaky
//! rather than wrong.

use moirai_async::AsyncExecutor;
use mnemosyne::counting::{CountingAllocator, measure};
use std::alloc::System;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

#[global_allocator]
static ALLOCATOR: CountingAllocator<System> = CountingAllocator::new(System);

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
    let ((), delta) = measure(|| {
        executor.block_on(YieldThenReady { remaining: yields });
    });
    // The budget counts allocation and reallocation calls -- the calls
    // the process-global counter this test replaced counted. Deallocations
    // were never part of it.
    delta.allocations + delta.reallocations
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
