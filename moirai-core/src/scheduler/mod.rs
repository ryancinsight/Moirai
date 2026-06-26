//! Scheduler identity.
//!
//! The canonical work-stealing scheduler abstraction lives in `moirai-executor`
//! as the [`WorkScheduler`] seam, implemented by `ThreadScheduler` and consumed
//! by `HybridExecutor`. This module retains only [`SchedulerId`], the scheduler
//! identifier used by metrics aggregation — the passive `Scheduler` trait,
//! `ScheduledTask`, and the standalone deques/config that once lived here were
//! dead duplicates of the executor's live types and have been removed.
//!
//! [`WorkScheduler`]: https://docs.rs/moirai-executor

use core::fmt;

/// A unique identifier for a scheduler instance.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchedulerId(usize);

impl SchedulerId {
    /// Creates a new scheduler ID.
    #[must_use]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Returns the raw ID value.
    #[must_use]
    pub const fn get(&self) -> usize {
        self.0
    }
}

impl fmt::Display for SchedulerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scheduler({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::SchedulerId;

    #[test]
    fn scheduler_id_round_trips_and_displays() {
        let id = SchedulerId::new(42);
        assert_eq!(id.get(), 42);
        assert_eq!(format!("{id}"), "Scheduler(42)");
    }
}
