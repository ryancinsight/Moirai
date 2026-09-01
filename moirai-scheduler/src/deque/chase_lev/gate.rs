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
        self.state.fetch_sub(STEAL_ACCESS_UNIT, Ordering::SeqCst);
    }
}

pub(super) struct ResizeGateClaim<'a> {
    state: &'a AtomicUsize,
}

impl Drop for ResizeGateClaim<'_> {
    fn drop(&mut self) {
        self.state.fetch_and(!RESIZE_CLAIMED_BIT, Ordering::SeqCst);
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
            let previous = self.state.fetch_add(STEAL_ACCESS_UNIT, Ordering::SeqCst);
            if previous & RESIZE_CLAIMED_BIT == 0 {
                return StealAccessGuard { state: &self.state };
            }
            self.state.fetch_sub(STEAL_ACCESS_UNIT, Ordering::SeqCst);
            on_backoff();
            yield_now();
        }
    }

    pub(super) fn claim(&self, after_claim: impl FnOnce()) -> ResizeGateClaim<'_> {
        let previous = self.state.fetch_or(RESIZE_CLAIMED_BIT, Ordering::SeqCst);
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
