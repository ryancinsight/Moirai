//! Bounded cooperative waiting for Chase-Lev storage contention.

/// The longest pause batch used by the existing scheduler lock ladder. These
/// waits do not own the resource that makes progress, so they yield after one
/// such batch instead of consuming the lock's larger aggregate spin budget.
const SPINS_BEFORE_YIELD: u8 = 64;

#[derive(Default)]
pub(super) struct ContentionWait {
    spins: u8,
}

impl ContentionWait {
    pub(super) const fn new() -> Self {
        Self { spins: 0 }
    }

    #[inline]
    pub(super) fn wait(&mut self) {
        self.wait_with(spin_once, yield_thread);
    }

    #[inline]
    fn wait_with(&mut self, spin: impl FnOnce(), yield_now: impl FnOnce()) {
        if self.spins < SPINS_BEFORE_YIELD {
            self.spins += 1;
            spin();
        } else {
            self.spins = 0;
            yield_now();
        }
    }
}

#[cfg(loom)]
fn spin_once() {
    // Loom needs a scheduler point where production emits a processor hint.
    loom::thread::yield_now();
}

#[cfg(not(loom))]
fn spin_once() {
    std::hint::spin_loop();
}

#[cfg(loom)]
fn yield_thread() {
    loom::thread::yield_now();
}

#[cfg(not(loom))]
fn yield_thread() {
    std::thread::yield_now();
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, mem::size_of};

    use super::{ContentionWait, SPINS_BEFORE_YIELD};

    #[test]
    fn wait_spins_for_one_bounded_batch_then_yields_and_repeats() {
        let spins = Cell::new(0usize);
        let yields = Cell::new(0usize);
        let mut wait = ContentionWait::new();

        for _ in 0..SPINS_BEFORE_YIELD {
            wait.wait_with(
                || spins.set(spins.get() + 1),
                || yields.set(yields.get() + 1),
            );
        }
        assert_eq!(spins.get(), usize::from(SPINS_BEFORE_YIELD));
        assert_eq!(yields.get(), 0);

        wait.wait_with(
            || spins.set(spins.get() + 1),
            || yields.set(yields.get() + 1),
        );
        assert_eq!(spins.get(), usize::from(SPINS_BEFORE_YIELD));
        assert_eq!(yields.get(), 1);

        for _ in 0..=SPINS_BEFORE_YIELD {
            wait.wait_with(
                || spins.set(spins.get() + 1),
                || yields.set(yields.get() + 1),
            );
        }
        assert_eq!(spins.get(), 2 * usize::from(SPINS_BEFORE_YIELD));
        assert_eq!(yields.get(), 2);
    }

    #[test]
    fn wait_state_is_one_byte() {
        assert_eq!(size_of::<ContentionWait>(), size_of::<u8>());
    }
}
