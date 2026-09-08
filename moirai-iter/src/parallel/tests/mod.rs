use super::*;
use std::cell::RefCell;
use std::sync::Arc;

mod aggregation;

/// Source length that forces the scheduler-backed drive path.
///
/// `PARALLEL_DRIVE_THRESHOLD` is 1024 and a source is only split across the
/// scheduler above it, so every terminal test below this length exercises the
/// inline branch only. These regressions pin the terminal contracts on the
/// path that actually splits.
const ABOVE_DRIVE_THRESHOLD: usize = 8_192;

mod conversions;
mod indexed;
mod reductions;
mod search;
mod thresholds;
mod transforms;
