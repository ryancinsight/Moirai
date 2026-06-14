//! Metric registry and snapshot collection.

use std::{
    collections::HashMap,
    sync::{Mutex, MutexGuard},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{Counter, Gauge, Histogram, MetricsSnapshot};

/// A metrics collection system.
#[derive(Debug, Default)]
pub struct Metrics {
    collector: MetricsCollector,
}

impl Metrics {
    /// Create a new metrics system.
    #[must_use]
    pub fn new() -> Self {
        Self {
            collector: MetricsCollector::new(),
        }
    }

    /// Get or create a counter by name.
    #[must_use]
    pub fn counter(&self, name: &str) -> Counter {
        self.collector.counter(name)
    }

    /// Get or create a gauge by name.
    #[must_use]
    pub fn gauge(&self, name: &str) -> Gauge {
        self.collector.gauge(name)
    }

    /// Get or create a histogram by name.
    #[must_use]
    pub fn histogram(&self, name: &str) -> Histogram {
        self.collector.histogram(name)
    }

    /// Collect a snapshot of all registered metrics.
    #[must_use]
    pub fn collect(&self) -> MetricsSnapshot {
        self.collector.collect()
    }
}

/// A thread-safe metrics collector.
#[derive(Debug, Default)]
pub struct MetricsCollector {
    counters: Mutex<HashMap<String, Counter>>,
    gauges: Mutex<HashMap<String, Gauge>>,
    histograms: Mutex<HashMap<String, Histogram>>,
}

impl MetricsCollector {
    /// Create a new metrics collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            counters: Mutex::new(HashMap::new()),
            gauges: Mutex::new(HashMap::new()),
            histograms: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a counter by name.
    #[must_use]
    pub fn counter(&self, name: &str) -> Counter {
        let mut counters = lock_map(&self.counters);
        counters.entry(name.to_owned()).or_default().clone()
    }

    /// Get or create a gauge by name.
    #[must_use]
    pub fn gauge(&self, name: &str) -> Gauge {
        let mut gauges = lock_map(&self.gauges);
        gauges.entry(name.to_owned()).or_default().clone()
    }

    /// Get or create a histogram by name.
    #[must_use]
    pub fn histogram(&self, name: &str) -> Histogram {
        let mut histograms = lock_map(&self.histograms);
        histograms.entry(name.to_owned()).or_default().clone()
    }

    /// Collect all metrics into a value snapshot.
    #[must_use]
    pub fn collect(&self) -> MetricsSnapshot {
        let counters = lock_map(&self.counters)
            .iter()
            .map(|(name, counter)| (name.clone(), counter.get()))
            .collect();
        let gauges = lock_map(&self.gauges)
            .iter()
            .map(|(name, gauge)| (name.clone(), gauge.get()))
            .collect();
        let histograms = lock_map(&self.histograms)
            .iter()
            .map(|(name, histogram)| (name.clone(), histogram.stats()))
            .collect();

        MetricsSnapshot {
            timestamp: unix_timestamp_seconds(),
            counters,
            gauges,
            histograms,
        }
    }
}

fn lock_map<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn unix_timestamp_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}
