//! Per-function time model of the local benchmark suite.
//!
//! `scripts/bench_suite.py` bounds the whole suite by a 300-second wall clock,
//! so the runtime of every benchmark binary is a designed quantity: each
//! benchmark function costs its warm-up, its measurement window, and Criterion's
//! analysis, about 0.4 seconds in all. The suite registers 567 functions;
//! `WARM_UP + MEASUREMENT` for each, plus process start-up and setup, is the
//! model the budget is checked against, and the runner reports the measured
//! total.
//!
//! One model serves every target so that a new benchmark function costs a known
//! amount and the suite cannot outgrow its budget without the runner naming the
//! target that did. Input sizes are not part of the model: a function keeps the
//! regime it measures, and only the sampling window is shared.

use criterion::Criterion;
use std::time::Duration;

/// Samples per benchmark function; Criterion's minimum.
const SAMPLE_SIZE: usize = 10;
/// Warm-up window per benchmark function.
const WARM_UP: Duration = Duration::from_millis(75);
/// Measurement window per benchmark function.
const MEASUREMENT: Duration = Duration::from_millis(200);

/// Criterion configuration carrying the suite's time model.
///
/// Plots are skipped: they cost more than the measurement they illustrate and
/// nothing in the suite consumes them.
pub fn criterion() -> Criterion {
    Criterion::default()
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(WARM_UP)
        .measurement_time(MEASUREMENT)
        .without_plots()
}
