//! Metrics collector benchmark.
//!
//! The rows isolate real `moirai-metrics` storage operations: shared atomic
//! counter handles, fixed-size snapshot collection, and Prometheus text export.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use moirai_metrics::{MetricsCollector, MetricsSnapshot, PrometheusExporter};
use std::time::Duration;

const SAMPLE_SIZE: usize = 20;
const MEASUREMENT_MILLIS: u64 = 500;
const WARM_UP_MILLIS: u64 = 100;
const METRIC_COUNT: usize = 32;

fn build_collector() -> MetricsCollector {
    let collector = MetricsCollector::new();
    for index in 0..METRIC_COUNT {
        collector
            .counter(&format!("requests_total_{index}"))
            .add(index as u64);
        collector
            .gauge(&format!("workers_active_{index}"))
            .set(index as i64 - 4);
        collector
            .histogram(&format!("task_seconds_{index}"))
            .record(index as f64 / 10.0);
    }
    collector
}

fn assert_snapshot(snapshot: &MetricsSnapshot) -> usize {
    assert_eq!(snapshot.counters.len(), METRIC_COUNT);
    assert_eq!(snapshot.gauges.len(), METRIC_COUNT);
    assert_eq!(snapshot.histograms.len(), METRIC_COUNT);
    assert_eq!(snapshot.counters["requests_total_31"], 31);
    assert_eq!(snapshot.gauges["workers_active_0"], -4);
    assert_eq!(snapshot.histograms["task_seconds_10"].count, 1);
    snapshot.counters.len() + snapshot.gauges.len() + snapshot.histograms.len()
}

fn metrics_counter_handle_add_get(counter: &moirai_metrics::Counter) -> u64 {
    counter.increment();
    black_box(counter.get())
}

fn metrics_collector_snapshot(collector: &MetricsCollector) -> usize {
    let snapshot = collector.collect();
    black_box(assert_snapshot(&snapshot))
}

fn metrics_prometheus_export(exporter: &PrometheusExporter, snapshot: &MetricsSnapshot) -> usize {
    let output = exporter.export(snapshot);
    assert!(output.contains("requests_total_31 31"));
    assert!(output.contains("workers_active_0 -4"));
    assert!(output.contains("task_seconds_10_count 1"));
    black_box(output.len())
}

fn bench_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics_collector_comparison");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));

    let counter = moirai_metrics::Counter::new();
    group.bench_function("counter_handle_add_get", |bench| {
        bench.iter(|| metrics_counter_handle_add_get(&counter));
    });

    let collector = build_collector();
    group.bench_function("collector_snapshot_32_each", |bench| {
        bench.iter(|| metrics_collector_snapshot(&collector));
    });

    let snapshot = collector.collect();
    let exporter = PrometheusExporter::new();
    group.bench_function("prometheus_export_32_each", |bench| {
        bench.iter(|| metrics_prometheus_export(&exporter, &snapshot));
    });

    group.finish();
}

criterion_group!(benches, bench_metrics);
criterion_main!(benches);
