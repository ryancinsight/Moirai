//! The block policy every blocked send and recv in this channel shares.
//!
//! A blocked path sleeps the same way whichever core it sits on: `1 << round`
//! spin-loop hints per round for [`MPMC_BLOCK_SPINS`] rounds, then a condvar
//! wait guarded by that side's waiter count. Four paths were spelling that out —
//! the ring's `send_bounded`/`recv_bounded` and the mutex path's
//! `send_unbounded`/`recv_unbounded` — so the policy lives here once and they
//! call into it. ADR-0016 item 2 asks for exactly this: one policy over both
//! cores, monomorphized rather than branched, so a caller that never blocks
//! compiles to the bare ring.
//!
//! The two cores spend a round differently, which is why the budget check stays
//! with the caller rather than inside [`backoff_step`]: the ring paths retry the
//! ring while holding no lock, while the mutex paths must release the channel
//! mutex *around* the spin so the other side can make progress.

#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, MutexGuard};

use super::channel::MpmcChannel;
use super::{MPMC_BLOCK_SPINS, MpmcState};

/// One round of the exponential backoff: `1 << *round` spin-loop hints, then
/// advance the round.
///
/// Callers own the [`MPMC_BLOCK_SPINS`] budget check — see the [module
/// docs](self) for why.
#[inline]
pub(super) fn backoff_step(round: &mut usize) {
    for _ in 0..(1 << *round) {
        std::hint::spin_loop();
    }
    *round += 1;
}

impl<T: Send> MpmcChannel<T> {
    /// Hold the channel mutex until `blocked` stops holding, backing off for
    /// [`MPMC_BLOCK_SPINS`] rounds before parking on `condvar`.
    ///
    /// `waiters` is the count the opposite side consults before it notifies, so
    /// it is incremented before the wait and decremented after it: a waiter that
    /// is not counted is a waiter that is never woken.
    pub(super) fn wait_while<'a>(
        &'a self,
        mut guard: MutexGuard<'a, MpmcState<T>>,
        condvar: &Condvar,
        waiters: &AtomicUsize,
        blocked: impl Fn(&MpmcState<T>) -> bool,
    ) -> MutexGuard<'a, MpmcState<T>> {
        let mutex = &self.state.0;
        let mut round = 0;

        while blocked(&guard) {
            if round < MPMC_BLOCK_SPINS {
                // Released across the backoff so the opposite side can take it
                // and publish the transition this thread is waiting for.
                drop(guard);
                backoff_step(&mut round);
                guard = mutex.lock().unwrap();
            } else {
                waiters.fetch_add(1, Ordering::AcqRel);
                guard = condvar.wait(guard).unwrap();
                waiters.fetch_sub(1, Ordering::AcqRel);
            }
        }

        guard
    }
}
