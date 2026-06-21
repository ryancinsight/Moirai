//! Work-stealing scheduler, statistics, and coordinator.

mod core;

#[cfg(test)]
mod tests;

pub use core::{SchedulerStats, SchedulerStatsSnapshot, WorkStealingCoordinator, WorkStealingScheduler};
