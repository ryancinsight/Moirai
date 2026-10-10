//! Resize-owner and thief-access admission for Chase-Lev storage.

#[cfg(loom)]
use loom::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicUsize, Ordering};

use super::contention::ContentionWait;

const RESIZE_CLAIMED_BIT: usize = 1;
const STEAL_ACCESS_UNIT: usize = 2;

pub(super) struct ResizeGate {
    state: AtomicUsize,
}

pub(super) struct StealAccessGuard<'a> {
    state: &'a AtomicUsize,
}

impl Drop for StealAccessGuard<'_> {
    fn drop(&mut self) {
        // Release: every storage read and slot claim of this access happens
        // before the drain in `ResizeGate::claim` observes the count return, so
        // `resize` never rewrites a buffer a thief still reads. Argued from C11,
        // not model-checked: loom does not explore a prior load moving past this
        // RMW, so the loom models in tests/loom_chase_lev_resize_gate.rs pass
        // with Relaxed here too.
        self.state.fetch_sub(STEAL_ACCESS_UNIT, Ordering::Release);
    }
}

pub(super) struct ResizeGateClaim<'a> {
    state: &'a AtomicUsize,
}

impl Drop for ResizeGateClaim<'_> {
    fn drop(&mut self) {
        // Release: the replacement buffer and every resize write precede the
        // cleared bit. An admission's `fetch_add` that reads it synchronizes
        // with this release, so its later `array` load is current.
        self.state.fetch_and(!RESIZE_CLAIMED_BIT, Ordering::Release);
    }
}

impl ResizeGate {
    pub(super) fn new() -> Self {
        Self {
            state: AtomicUsize::new(0),
        }
    }

    pub(super) fn enter(
        &self,
        mut before_attempt: impl FnMut(),
        mut on_backoff: impl FnMut(),
    ) -> StealAccessGuard<'_> {
        loop {
            before_attempt();
            // Every live contribution represents a real thread holding or
            // attempting one guard, so process resource limits make wrapping
            // this `usize` counter unreachable. The returned prior value is
            // the admission decision: an increment ordered before the owner
            // claim is visible to its drain; one ordered after sees the claim
            // bit and backs out before loading storage.
            // SeqCst, although the admitted thief needs only Acquire: it takes the
            // edge from `ResizeGateClaim::drop`, which publishes the replacement
            // buffer, so its later `array` load is current. Weaker orderings on
            // this retry path exhaust the loom model's branch budget (tests/
            // loom_chase_lev_resize_gate.rs), so the claim-release edge is checked
            // only with this ordering. Every ordering lowers to a lock-prefixed
            // instruction on x86-64.
            let previous = self.state.fetch_add(STEAL_ACCESS_UNIT, Ordering::SeqCst);
            if previous & RESIZE_CLAIMED_BIT == 0 {
                return StealAccessGuard { state: &self.state };
            }
            // SeqCst for the same loom reason as the admission above. The rejected
            // attempt touched no storage, so only its place in the total order matters.
            self.state.fetch_sub(STEAL_ACCESS_UNIT, Ordering::SeqCst);
            on_backoff();
            yield_now();
        }
    }

    pub(super) fn claim(&self, after_claim: impl FnOnce()) -> ResizeGateClaim<'_> {
        // Relaxed: the claim publishes no data that an admission reads. An
        // admission ordered after it reads the bit and backs out before touching
        // storage, and the drain observes the thieves admitted before it.
        let previous = self.state.fetch_or(RESIZE_CLAIMED_BIT, Ordering::Relaxed);
        let claim = ResizeGateClaim { state: &self.state };
        debug_assert_eq!(
            previous & RESIZE_CLAIMED_BIT,
            0,
            "the owner must not nest resize gate claims"
        );
        after_claim();

        let mut wait = ContentionWait::new();
        while self.state() != RESIZE_CLAIMED_BIT {
            wait.wait();
        }
        claim
    }

    pub(super) fn state(&self) -> usize {
        // SeqCst, not Acquire: reads the retry path's RMWs in the same total order.
        // Acquire here exhausts the loom model's branch budget. A SeqCst load
        // lowers to the same instruction as an Acquire load on x86-64 and AArch64.
        // Its acquire half still synchronizes with each departing thief's Release
        // `fetch_sub`, since every later RMW extends that release sequence.
        self.state.load(Ordering::SeqCst)
    }
}

#[cfg(loom)]
fn yield_now() {
    loom::thread::yield_now();
}

#[cfg(not(loom))]
fn yield_now() {
    std::thread::yield_now();
}
