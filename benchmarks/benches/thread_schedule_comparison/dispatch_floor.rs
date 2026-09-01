//! Reproducible fixed-cost and policy-crossover instrument for indexed work.
//!
//! The earlier dispatch-floor observation did not retain its executable probe.
//! This replacement pins four workers, immutable input addresses, geometric
//! sizes, exact operation bodies, and Criterion windows. Runtime construction,
//! input allocation, and value validation happen outside timed regions.

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use moirai::Moirai;
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

const WORKER_THREADS: usize = 4;
const PRIMITIVE_COUNTS: &[usize] = &[1_024, 4_096, 16_384, 65_536];
const CROSSOVER_COUNTS: &[usize] = &[512, 1_024, 4_096, 8_192, 16_384, 32_768, 65_536];
const FMA_STEPS: usize = 24;

fn input(count: usize) -> Vec<f64> {
    (0..count)
        .map(|index| 1.0 + (index % 1_024) as f64 / 1_024.0)
        .collect()
}

fn expected_index_sum(count: usize) -> usize {
    count.wrapping_mul(count.wrapping_add(1)) / 2
}

fn one_multiply(value: f64) -> f64 {
    value * 1.000_000_119_209_289_6
}

fn square_root_plus_log_one_plus(value: f64) -> f64 {
    value.sqrt() + value.ln_1p()
}

fn chained_fused_multiply_add(mut value: f64) -> f64 {
    for _ in 0..FMA_STEPS {
        value = value.mul_add(1.000_000_119_209_289_6, 0.000_001);
    }
    value
}

fn serial_sum<F>(values: &[f64], body: F) -> f64
where
    F: Fn(f64) -> f64 + Copy,
{
    values
        .iter()
        .fold(0.0, |sum, &value| sum + body(black_box(value)))
}

fn parallel_sum<F>(runtime: &Moirai, values: &[f64], body: F) -> f64
where
    F: Fn(f64) -> f64 + Copy + Send + Sync,
{
    runtime
        .map_reduce_indexed(
            values.len(),
            0.0,
            |index| body(black_box(values[index])),
            |left, right| left + right,
        )
        .expect("indexed policy-crossover reduction must complete")
}

fn assert_reduction_matches(actual: f64, expected: f64, count: usize) -> f64 {
    // Every body is positive on [1, 2). For naive summation, Higham's
    // gamma_n = n*eps/(1-n*eps) bounds each reduction's forward error by
    // gamma_n * sum(abs(terms)). Serial and chunked sums may each attain that
    // bound, so twice gamma_n bounds their difference; the factor of two also
    // covers the rounded serial sum used as the scale estimate.
    let scaled_epsilon = count as f64 * f64::EPSILON;
    let gamma = scaled_epsilon / (1.0 - scaled_epsilon);
    let tolerance = 4.0 * gamma * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "reduction differs by {} beyond derived bound {tolerance}",
        (actual - expected).abs()
    );
    black_box(actual)
}

fn validate_for_each_exactly_once(runtime: &Moirai, count: usize) {
    let visits = (0..count).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
    runtime
        .for_each_indexed(count, |index| {
            visits[index].fetch_add(1, Ordering::Relaxed);
        })
        .expect("indexed fan-out validation must complete");
    assert!(
        visits
            .iter()
            .all(|visits| visits.load(Ordering::Relaxed) == 1),
        "indexed fan-out must visit every index exactly once"
    );
}

fn bench_primitives(c: &mut Criterion, runtime: &Moirai) {
    let mut group = c.benchmark_group("dispatch_floor/four_workers/primitives");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(250));

    for &count in PRIMITIVE_COUNTS {
        validate_for_each_exactly_once(runtime, count);
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("for_each_indexed", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    runtime
                        .for_each_indexed(count, |index| {
                            black_box(index);
                        })
                        .expect("indexed fan-out must complete")
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("map_reduce_indexed", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let sum = runtime
                        .map_reduce_indexed(
                            count,
                            0usize,
                            |index| black_box(index.wrapping_add(1)),
                            usize::wrapping_add,
                        )
                        .expect("indexed primitive reduction must complete");
                    assert_eq!(sum, expected_index_sum(count));
                    black_box(sum)
                });
            },
        );
    }

    group.finish();
}

fn bench_policy_body<F>(c: &mut Criterion, runtime: &Moirai, name: &str, body: F)
where
    F: Fn(f64) -> f64 + Copy + Send + Sync,
{
    let mut group = c.benchmark_group(format!(
        "dispatch_floor/four_workers/policy_crossover/{name}"
    ));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(250));

    for &count in CROSSOVER_COUNTS {
        let values = input(count);
        let expected = serial_sum(&values, body);
        assert_reduction_matches(parallel_sum(runtime, &values, body), expected, count);
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("serial", count),
            values.as_slice(),
            |b, values| {
                b.iter(|| assert_reduction_matches(serial_sum(values, body), expected, count));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("parallel", count),
            values.as_slice(),
            |b, values| {
                b.iter(|| {
                    assert_reduction_matches(parallel_sum(runtime, values, body), expected, count)
                });
            },
        );
    }

    group.finish();
}

pub(super) fn bench(c: &mut Criterion) {
    let runtime = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("dispatch-floor runtime must start");

    bench_primitives(c, &runtime);
    bench_policy_body(c, &runtime, "one_multiply", one_multiply);
    bench_policy_body(
        c,
        &runtime,
        "square_root_plus_log_one_plus",
        square_root_plus_log_one_plus,
    );
    bench_policy_body(
        c,
        &runtime,
        "twenty_four_chained_fused_multiply_adds",
        chained_fused_multiply_add,
    );

    runtime.shutdown();
}
