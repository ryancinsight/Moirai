//! Point-in-time metric snapshots.

use std::collections::HashMap;

use crate::HistogramStats;

/// A value snapshot of all metrics at one collection point.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricsSnapshot {
    /// Unix timestamp in seconds.
    pub timestamp: u64,
    /// Counter values by metric name.
    pub counters: HashMap<String, u64>,
    /// Gauge values by metric name.
    pub gauges: HashMap<String, i64>,
    /// Histogram statistics by metric name.
    pub histograms: HashMap<String, HistogramStats>,
}
