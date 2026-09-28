//! Contended-steal retry policy.

use moirai_scheduler::StealResult;

/// Lost-race processor hints emitted before yielding to another runnable thread.
///
/// This matches the established upper handoff window used by Moirai's
/// contended spin lock while keeping steal retries allocation- and sleep-free.
pub(super) const STEAL_SPINS_BEFORE_YIELD: usize = 1_000;

/// [`STEAL_SPINS_BEFORE_YIELD`] as the shared [`moirai_utils::backoff::Spins`]
/// budget, so the retry drives the one spin-then-hand-off schedule rather than
/// its own loop.
struct StealSpins;

impl moirai_utils::backoff::Spins for StealSpins {
    const SPIN_ATTEMPTS: usize = STEAL_SPINS_BEFORE_YIELD;
}

#[inline]
pub(super) fn steal_after_contention<T>(steal: impl FnMut() -> StealResult<T>) -> Option<T> {
    steal_after_contention_with(steal, std::hint::spin_loop, std::thread::yield_now)
}

#[inline]
pub(super) fn steal_after_contention_with<T>(
    mut steal: impl FnMut() -> StealResult<T>,
    mut spin: impl FnMut(),
    mut yield_now: impl FnMut(),
) -> Option<T> {
    let mut spins = 0usize;
    loop {
        match steal() {
            StealResult::Success(value) => return Some(value),
            StealResult::Empty => return None,
            StealResult::Retry => {
                // One round of this site's budget, driven by the injected hint
                // action: `spin_then_with` reports `true` once the budget is
                // spent, which is when the retry hands the core off.
                if moirai_utils::backoff::spin_then_with::<false, StealSpins>(&mut spins, &mut spin)
                {
                    spins = 0;
                    yield_now();
                }
            }
        }
    }
}
