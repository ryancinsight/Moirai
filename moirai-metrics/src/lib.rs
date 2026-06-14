//! Performance metrics and monitoring for Moirai.
//!
//! The crate exposes cloneable metric handles backed by shared atomic or
//! bounded mutex state. Metric snapshots are value copies suitable for export
//! without retaining collector locks.

mod collector;
mod counter;
mod exporter;
mod gauge;
mod histogram;
mod snapshot;

pub use collector::{Metrics, MetricsCollector};
pub use counter::Counter;
pub use exporter::PrometheusExporter;
pub use gauge::Gauge;
pub use histogram::{Histogram, HistogramError, HistogramStats};
pub use snapshot::MetricsSnapshot;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
