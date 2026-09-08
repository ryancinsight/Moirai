//! Contended-steal retry policy.

use moirai_scheduler::StealResult;

/// Lost-race processor hints emitted before yielding to another runnable thread.
///
/// This matches the established upper handoff window used by Moirai's
/// contended spin lock while keeping steal retries allocation- and sleep-free.
pub(super) const STEAL_SPINS_BEFORE_YIELD: usize = 1_000;
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
            StealResult::Retry if spins < STEAL_SPINS_BEFORE_YIELD => {
                spins += 1;
                spin();
            }
            StealResult::Retry => {
                spins = 0;
                yield_now();
            }
        }
    }
}
