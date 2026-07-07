//! Histogram metric handle and derived statistics.

use std::sync::{Arc, Mutex, MutexGuard};

/// Error returned when a histogram sample violates the finite-value contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistogramError {
    /// The rejected sample value.
    pub value: f64,
}

/// Statistics from a histogram.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramStats {
    /// Number of samples.
    pub count: u64,
    /// Sum of all samples.
    pub sum: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Mean value.
    pub mean: f64,
    /// Population standard deviation.
    pub stddev: f64,
}

impl Default for HistogramStats {
    fn default() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            stddev: 0.0,
        }
    }
}

/// Running-moment state maintained with Welford's online algorithm.
///
/// The naive `E[x²] − mean²` formulation subtracts two nearly-equal large
/// numbers for large-mean/small-variance samples and cancels catastrophically
/// (variance collapses to 0 or negative once `mean² · ε` exceeds the true
/// variance). Welford accumulates the centered second moment `m2 = Σ(x−mean)²`
/// incrementally, whose relative error is O(n·ε·κ) with condition number
/// κ = √(1 + mean²/σ²) — accurate where the naive form has already lost every
/// significant digit.
#[derive(Debug, Default)]
struct HistogramState {
    count: u64,
    sum: f64,
    /// Running mean (Welford).
    mean: f64,
    /// Centered second moment Σ(x−mean)² (Welford).
    m2: f64,
    min: f64,
    max: f64,
}

/// A finite-sample histogram metric.
#[derive(Clone, Debug)]
pub struct Histogram {
    state: Arc<Mutex<HistogramState>>,
}

impl Histogram {
    /// Create an independent empty histogram.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(HistogramState::default())),
        }
    }

    /// Record a finite sample.
    ///
    /// Non-finite values are ignored to preserve the historical infallible API.
    /// Call [`Histogram::try_record`] when rejection must be observed.
    pub fn record(&self, value: f64) {
        let _ = self.try_record(value);
    }

    /// Record a finite sample and report invalid values.
    pub fn try_record(&self, value: f64) -> Result<(), HistogramError> {
        if !value.is_finite() {
            return Err(HistogramError { value });
        }

        let mut state = lock_histogram(&self.state);
        if state.count == 0 {
            state.min = value;
            state.max = value;
        } else {
            state.min = state.min.min(value);
            state.max = state.max.max(value);
        }
        state.count += 1;
        state.sum += value;
        // Welford update: delta uses the pre-update mean, delta2 the
        // post-update mean; their product accumulates the centered second
        // moment without forming cancellation-prone uncentered sums.
        let delta = value - state.mean;
        state.mean += delta / state.count as f64;
        let delta2 = value - state.mean;
        state.m2 += delta * delta2;
        Ok(())
    }

    /// Get current histogram statistics.
    #[must_use]
    pub fn stats(&self) -> HistogramStats {
        let state = lock_histogram(&self.state);
        if state.count == 0 {
            return HistogramStats::default();
        }

        let count = state.count;
        // Population variance from the centered second moment; `m2` is a sum
        // of non-negative terms up to rounding, so clamp only guards the last
        // ulp of rounding noise, not a structural cancellation.
        let variance = (state.m2 / count as f64).max(0.0);
        HistogramStats {
            count,
            sum: state.sum,
            min: state.min,
            max: state.max,
            mean: state.mean,
            stddev: variance.sqrt(),
        }
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

fn lock_histogram(state: &Mutex<HistogramState>) -> MutexGuard<'_, HistogramState> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
