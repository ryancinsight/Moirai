//! Producer-side wake gate shared by the send paths and both endpoint drops.

use std::sync::atomic::{fence, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::task::Waker;

/// Wake every parked thread and registered async waker after a publication.
///
/// The caller has just published state consumers must observe — a produced
/// element or the `closed` flag — and gates the wake on counters that are read
/// *outside* the registration locks. Counter registration and the publication
/// live in different locations, so the two sides form a store-buffer (Dekker)
/// pair:
///
/// | | consumer (registration) | producer (this gate) |
/// |---|---|---|
/// | first | `parked_count` / `waker_count` increment | publish (produce / close) |
/// | then  | ring / `closed` re-check | counter gate load below |
///
/// Without a `StoreLoad` barrier on both sides, both "then" loads may observe
/// stale values at once: this gate reads a zero counter and skips the wake
/// while the consumer's re-check misses the publication and it parks with the
/// message already delivered — the last-message hang. `moirai-sync`'s
/// `FutexMutex` (`lock_slow`/`unlock` in `src/sync/futex_mutex.rs`) documents
/// and fences this exact pattern on its `waiters`/`locked` pair; the
/// `fence(SeqCst)` here follows it and pairs with the fence each registration
/// site executes between its counter increment and its re-check.
///
/// Wakers are drained under the registry lock but woken after it is released:
/// `Waker::wake` may execute the woken task inline on this thread (the
/// executor polls inline under injector saturation), and that poll may
/// re-enter this channel's registration lock — waking under the lock would
/// self-deadlock. Unparking follows the same shape so both critical sections
/// stay minimal.
pub(super) fn notify_consumers(
    parker: &Mutex<Vec<std::thread::Thread>>,
    parked_count: &AtomicUsize,
    async_wakers: &Mutex<Vec<(u64, Waker)>>,
    waker_count: &AtomicUsize,
) {
    fence(Ordering::SeqCst);

    if parked_count.load(Ordering::Relaxed) > 0 {
        let parked = {
            let mut parked = parker
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let drained = std::mem::take(&mut *parked);
            // Inside the lock, so a concurrently-registering consumer cannot
            // have its increment erased by this reset.
            parked_count.store(0, Ordering::Release);
            drained
        };
        for thread in parked {
            thread.unpark();
        }
    }

    if waker_count.load(Ordering::Relaxed) > 0 {
        let wakers = {
            let mut wakers = async_wakers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let drained = std::mem::take(&mut *wakers);
            // Inside the lock, matching the parked-count reset above.
            waker_count.store(0, Ordering::Release);
            drained
        };
        for (_, waker) in wakers {
            waker.wake();
        }
    }
}
