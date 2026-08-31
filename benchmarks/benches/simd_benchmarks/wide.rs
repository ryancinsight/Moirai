use super::{SIMD_MEASUREMENT_SECONDS, SIMD_SAMPLE_SIZE, SIMD_WARM_UP_MILLIS};
use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use moirai_utils::simd::add;
use std::time::Duration;

fn generate_wide_test_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
    let b: Vec<f64> = (0..size).map(|i| (size - i) as f64 * 0.1).collect();
    (a, b)
}

/// Benchmark wide real vector addition operations.
pub(super) fn bench_vector_addition_wide(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_addition_wide");
    group.sample_size(SIMD_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(SIMD_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(SIMD_WARM_UP_MILLIS));

    for size in [64, 256, 1024, 4096, 16384].iter() {
        let (a, b) = generate_wide_test_data(*size);
        let mut result = vec![0.0; *size];
        let expected: Vec<f64> = a
            .iter()
            .zip(b.iter())
            .map(|(left, right)| left + right)
            .collect();

        add(&a, &b, &mut result);
        assert_eq!(result, expected);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("wide_vectorized", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    add(black_box(&a), black_box(&b), black_box(&mut result));
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| {
                for i in 0..*size {
                    result[i] = black_box(a[i] + b[i]);
                }
            });
        });
    }

    group.finish();
}
