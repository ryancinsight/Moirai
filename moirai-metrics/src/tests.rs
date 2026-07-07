use super::{Histogram, Metrics, MetricsCollector, PrometheusExporter};

fn assert_close(actual: f64, expected: f64) {
    let tolerance = f64::EPSILON * 16.0 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual} expected {expected} tolerance {tolerance}"
    );
}

#[test]
fn metrics_handles_share_named_storage() {
    let metrics = Metrics::new();

    let first = metrics.counter("tasks_total");
    let second = metrics.counter("tasks_total");
    first.increment();
    second.add(4);

    let gauge = metrics.gauge("active_workers");
    gauge.set(3);
    metrics.gauge("active_workers").decrement();

    let histogram = metrics.histogram("task_seconds");
    histogram.record(1.0);
    metrics
        .histogram("task_seconds")
        .try_record(2.0)
        .expect("finite histogram sample must be accepted");

    let snapshot = metrics.collect();
    assert_eq!(snapshot.counters["tasks_total"], 5);
    assert_eq!(snapshot.gauges["active_workers"], 2);
    assert_eq!(snapshot.histograms["task_seconds"].count, 2);
    assert_eq!(snapshot.histograms["task_seconds"].sum, 3.0);
}

#[test]
fn histogram_stats_are_value_semantic() {
    let histogram = Histogram::new();
    for sample in [1.0, 2.0, 4.0] {
        histogram
            .try_record(sample)
            .expect("finite histogram sample must be accepted");
    }

    let stats = histogram.stats();
    assert_eq!(stats.count, 3);
    assert_eq!(stats.sum, 7.0);
    assert_eq!(stats.min, 1.0);
    assert_eq!(stats.max, 4.0);
    assert_close(stats.mean, 7.0 / 3.0);
    assert_close(stats.stddev, (14.0_f64 / 9.0).sqrt());

    assert!(histogram.try_record(f64::NAN).is_err());
    assert_eq!(histogram.stats(), stats);
}

#[test]
fn histogram_variance_survives_large_offset_small_spread() {
    // Large-mean/small-variance regime where the previous E[x²] − mean²
    // formulation cancels catastrophically: with mean = 1e9, mean² = 1e18 and
    // f64 ε ≈ 2.2e-16, the uncentered form's absolute error floor is
    // mean²·ε ≈ 220 — over 300× the true variance of 2/3 — so it returned 0
    // (after the negative clamp) with zero significant digits. Welford's
    // centered accumulation must recover the variance to near machine
    // precision.
    let offset = 1.0e9;
    let samples = [offset - 1.0, offset, offset + 1.0];
    let histogram = Histogram::new();
    for sample in samples {
        histogram
            .try_record(sample)
            .expect("finite histogram sample must be accepted");
    }

    let true_mean = offset;
    let true_var = 2.0 / 3.0; // population variance of {-1, 0, +1} around 0
    let stats = histogram.stats();
    assert_eq!(stats.count, 3);
    assert_eq!(stats.min, offset - 1.0);
    assert_eq!(stats.max, offset + 1.0);

    // Tolerance derived from Welford's error bound: relative error on m2 is
    // O(n·ε·κ) with condition number κ = √(1 + mean²/σ²) (Chan, Golub &
    // LeVeque 1983). Here n = 3, κ ≈ 1.22e9, so the variance tolerance is
    // n·ε·κ·σ² ≈ 5.4e-7 — nine orders of magnitude below the old formula's
    // ~220 error floor.
    let n = samples.len() as f64;
    let kappa = (1.0 + true_mean * true_mean / true_var).sqrt();
    let var_tol = n * f64::EPSILON * kappa * true_var;
    let variance = stats.stddev * stats.stddev;
    assert!(
        (variance - true_var).abs() <= var_tol,
        "variance {variance} must be within {var_tol} of {true_var}"
    );
    // Mean is conditioned at κ_mean ≈ 1 relative to its own magnitude.
    assert!((stats.mean - true_mean).abs() <= n * f64::EPSILON * true_mean);
}

#[test]
fn collector_snapshot_contains_registered_values() {
    let collector = MetricsCollector::new();
    collector.counter("requests_total").add(7);
    collector.gauge("queue_depth").set(-2);
    collector.histogram("latency_seconds").record(0.25);

    let snapshot = collector.collect();
    assert!(snapshot.timestamp > 0);
    assert_eq!(snapshot.counters["requests_total"], 7);
    assert_eq!(snapshot.gauges["queue_depth"], -2);
    assert_eq!(snapshot.histograms["latency_seconds"].count, 1);
    assert_eq!(snapshot.histograms["latency_seconds"].mean, 0.25);
}

#[test]
fn prometheus_exporter_emits_deterministic_values() {
    let collector = MetricsCollector::new();
    collector.counter("requests.total").add(3);
    collector.gauge("workers:active").set(2);
    collector.histogram("latency seconds").record(0.5);

    let output = PrometheusExporter::new().export(&collector.collect());
    assert!(output.contains("# TYPE requests_total counter"));
    assert!(output.contains("requests_total 3"));
    assert!(output.contains("# TYPE workers:active gauge"));
    assert!(output.contains("workers:active 2"));
    assert!(output.contains("# TYPE latency_seconds_count gauge"));
    assert!(output.contains("latency_seconds_count 1"));
    assert!(output.contains("latency_seconds_sum 0.5"));
}
