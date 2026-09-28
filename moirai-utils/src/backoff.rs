//! The one "spin, then hand off" schedule the contended paths share.
//!
//! A contended hot path spins for a while and then hands the core off — to the
//! scheduler, to a condvar, or to a futex. Seven sites in the runtime spelled
//! that loop out, each with its own constant: the synchronized spin lock and its
//! stealing twin, the SPSC ring's blocking send/recv, the MPMC block policy, the
//! bounded queue's blocking `enqueue`, the futex mutex, and the runtime worker's
//! spin-then-park wait. The *loop* and the `1 << round` hint schedule are the
//! same at every one of them, so they live here once.
//!
//! What legitimately differs is kept at the call site:
//!
//! * the **budget** — how many rounds to spin — is each site's own, an impl of
//!   [`Spins`]. The numbers were tuned against different hand-off costs (a
//!   `yield_now` is far cheaper than a futex wait) and are deliberately *not*
//!   unified.
//! * the **hand-off action** — `yield_now`, `thread::park`, a condvar wait, a
//!   mutex lock — is each site's own, so [`spin_then`] reports that the budget is
//!   spent and lets the caller do it.
//!
//! [`spin_round`] only uses `core::hint::spin_loop`, so this module needs no
//! `std`.

/// A spin budget: how many rounds a hot loop spins before it hands off.
///
/// Implementors are zero-sized marker types, so the budget const-folds into the
/// loop and carries no runtime value — the same shape as
/// [`moirai_core`](https://docs.rs/moirai-core)'s `ResultWaitPolicy` and the
/// scheduler's `SPIN_LIMIT` type parameter.
pub trait Spins {
    /// Rounds to spin before the caller's hand-off action.
    const SPIN_ATTEMPTS: usize;
}

/// Spin one round's worth of hints through `hint`, returning how many were
/// performed.
///
/// `EXPONENTIAL` selects the `1 << round` schedule; otherwise the round spends a
/// single hint. The count is returned so a caller that also budgets *total*
/// hints (the synchronized spin lock yields after a running total) can reuse it
/// instead of re-deriving the schedule.
///
/// `round` must stay below the width of `usize` when `EXPONENTIAL` holds; every
/// caller bounds it by its own small [`Spins::SPIN_ATTEMPTS`] or by a cap.
#[inline]
pub fn spin_hints<const EXPONENTIAL: bool>(round: usize, mut hint: impl FnMut()) -> usize {
    let hints = if EXPONENTIAL { 1usize << round } else { 1 };
    for _ in 0..hints {
        hint();
    }
    hints
}

/// [`spin_hints`] driving the platform's `spin_loop` hint.
#[inline]
pub fn spin_round<const EXPONENTIAL: bool>(round: usize) -> usize {
    spin_hints::<EXPONENTIAL>(round, core::hint::spin_loop)
}

/// Spend one round of `S`'s budget, or report that it is exhausted.
///
/// Returns `false` after spinning a round (the caller should retry its
/// condition) and `true` once [`S::SPIN_ATTEMPTS`](Spins::SPIN_ATTEMPTS) rounds
/// are spent. The caller then performs its hand-off and, if its schedule
/// restarts, resets the round counter it passed in.
#[inline]
pub fn spin_then<const EXPONENTIAL: bool, S: Spins>(round: &mut usize) -> bool {
    spin_then_with::<EXPONENTIAL, S>(round, core::hint::spin_loop)
}

/// [`spin_then`] with an injected hint action.
///
/// A site whose hint action is a parameter — the work-stealing retry takes its
/// `spin` and `yield_now` actions as arguments so tests can count them — spends
/// its budget through this entry, keeping the shared schedule while leaving the
/// action the caller's.
#[inline]
pub fn spin_then_with<const EXPONENTIAL: bool, S: Spins>(
    round: &mut usize,
    hint: impl FnMut(),
) -> bool {
    if *round < S::SPIN_ATTEMPTS {
        spin_hints::<EXPONENTIAL>(*round, hint);
        *round += 1;
        false
    } else {
        true
    }
}
