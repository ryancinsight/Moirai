//! Prometheus text exposition.

use std::fmt::Write;

use crate::{HistogramStats, MetricsSnapshot};

/// A zero-sized Prometheus text exporter.
#[derive(Debug, Default, Clone, Copy)]
pub struct PrometheusExporter;

impl PrometheusExporter {
    /// Create a new Prometheus exporter.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Export metrics in Prometheus text format.
    #[must_use]
    pub fn export(&self, snapshot: &MetricsSnapshot) -> String {
        let mut output = String::new();
        write_group(
            &mut output,
            "moirai_snapshot_timestamp_seconds",
            "gauge",
            snapshot.timestamp,
        );

        let mut counters: Vec<_> = snapshot.counters.iter().collect();
        counters.sort_unstable_by(|left, right| left.0.cmp(right.0));
        for (name, value) in counters {
            write_group(&mut output, &metric_name(name), "counter", *value);
        }

        let mut gauges: Vec<_> = snapshot.gauges.iter().collect();
        gauges.sort_unstable_by(|left, right| left.0.cmp(right.0));
        for (name, value) in gauges {
            write_group(&mut output, &metric_name(name), "gauge", *value);
        }

        let mut histograms: Vec<_> = snapshot.histograms.iter().collect();
        histograms.sort_unstable_by(|left, right| left.0.cmp(right.0));
        for (name, stats) in histograms {
            write_histogram(&mut output, &metric_name(name), stats);
        }

        output
    }
}

fn write_group<T>(output: &mut String, name: &str, metric_type: &str, value: T)
where
    T: std::fmt::Display,
{
    let _ = writeln!(output, "# TYPE {name} {metric_type}");
    let _ = writeln!(output, "{name} {value}");
}

fn write_histogram(output: &mut String, name: &str, stats: &HistogramStats) {
    write_group(output, &format!("{name}_count"), "gauge", stats.count);
    write_group(output, &format!("{name}_sum"), "gauge", stats.sum);
    write_group(output, &format!("{name}_min"), "gauge", stats.min);
    write_group(output, &format!("{name}_max"), "gauge", stats.max);
    write_group(output, &format!("{name}_mean"), "gauge", stats.mean);
    write_group(output, &format!("{name}_stddev"), "gauge", stats.stddev);
}

fn metric_name(name: &str) -> String {
    let mut sanitized = String::with_capacity(name.len().saturating_add(1));
    for (index, character) in name.chars().enumerate() {
        let valid = character.is_ascii_alphanumeric() || character == '_' || character == ':';
        let first_valid = character.is_ascii_alphabetic() || character == '_' || character == ':';
        if index == 0 && !first_valid {
            sanitized.push('_');
        }
        sanitized.push(if valid { character } else { '_' });
    }
    if sanitized.is_empty() {
        "_".to_owned()
    } else {
        sanitized
    }
}
