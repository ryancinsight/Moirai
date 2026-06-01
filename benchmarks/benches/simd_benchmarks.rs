//! SIMD performance benchmarks for Moirai utilities.
//!
//! This benchmark suite measures the performance of vectorized operations
//! compared to scalar implementations across different data sizes and
//! CPU architectures.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use moirai_utils::global_simd_counter;
use moirai_utils::simd::*;
use std::time::Duration;

const SIMD_SAMPLE_SIZE: usize = 10;
const SIMD_MEASUREMENT_SECONDS: u64 = 1;
const SIMD_WARM_UP_MILLIS: u64 = 250;

/// Generate test data for benchmarks.
fn generate_test_data(size: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.1).collect();
    (a, b)
}

/// Generate 4x4 matrix test data.
fn generate_matrix_data() -> ([f32; 16], [f32; 16]) {
    let a = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let b = [
        16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    ];
    (a, b)
}

/// Benchmark vector addition operations.
fn bench_vector_addition(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_addition");
    group.sample_size(SIMD_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(SIMD_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(SIMD_WARM_UP_MILLIS));

    for size in [64, 256, 1024, 4096, 16384].iter() {
        let (a, b) = generate_test_data(*size);
        let mut result = vec![0.0f32; *size];

        group.throughput(Throughput::Elements(*size as u64));

        // Benchmark safe vectorized addition
        group.bench_with_input(
            BenchmarkId::new("safe_vectorized", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    add(black_box(&a), black_box(&b), black_box(&mut result));
                });
            },
        );

        // Benchmark scalar implementation for comparison
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| {
                for i in 0..*size {
                    result[i] = black_box(a[i] + b[i]);
                }
            });
        });

        if has_native_vector_path::<f32>() && *size % 8 == 0 {
            group.bench_with_input(
                BenchmarkId::new("native_checked", size),
                size,
                |bench, _| {
                    bench.iter(|| {
                        add(black_box(&a), black_box(&b), black_box(&mut result));
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark vector addition with scalar tails after a native vector prefix.
fn bench_vector_prefix_tail_addition(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_prefix_tail_addition");
    group.sample_size(SIMD_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(SIMD_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(SIMD_WARM_UP_MILLIS));

    for size in [65, 4099, 16385].iter() {
        let (a, b) = generate_test_data(*size);
        let mut result = vec![0.0f32; *size];
        let expected: Vec<f32> = a
            .iter()
            .zip(b.iter())
            .map(|(left, right)| left + right)
            .collect();

        add(&a, &b, &mut result);
        assert_eq!(result, expected);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("generic_prefix_tail", size),
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

/// Benchmark vector multiplication operations.
fn bench_vector_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_multiplication");
    group.sample_size(SIMD_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(SIMD_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(SIMD_WARM_UP_MILLIS));

    for size in [64, 256, 1024, 4096, 16384].iter() {
        let (a, b) = generate_test_data(*size);
        let mut result = vec![0.0f32; *size];

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("safe_vectorized", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    mul(black_box(&a), black_box(&b), black_box(&mut result));
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| {
                for i in 0..*size {
                    result[i] = black_box(a[i] * b[i]);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark dot product operations.
fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    group.sample_size(SIMD_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(SIMD_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(SIMD_WARM_UP_MILLIS));

    for size in [64, 256, 1024, 4096, 16384].iter() {
        let (a, b) = generate_test_data(*size);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("safe_vectorized", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(dot(black_box(&a), black_box(&b)));
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| {
                let result: f32 = a.iter().zip(b.iter()).map(|(x, y)| black_box(x * y)).sum();
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("iterator", size), size, |bench, _| {
            bench.iter(|| {
                let result: f32 = black_box(&a)
                    .iter()
                    .zip(black_box(&b).iter())
                    .map(|(x, y)| x * y)
                    .sum();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark matrix multiplication operations.
fn bench_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication_4x4");
    group.sample_size(SIMD_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(SIMD_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(SIMD_WARM_UP_MILLIS));
    let (a, b) = generate_matrix_data();
    let mut result = [0.0f32; 16];

    group.throughput(Throughput::Elements(16));

    group.bench_function("safe_vectorized", |bench| {
        bench.iter(|| {
            matrix_mul_square::<f32, 4>(black_box(&a), black_box(&b), black_box(&mut result));
        });
    });

    group.bench_function("scalar", |bench| {
        bench.iter(|| {
            for i in 0..4 {
                for j in 0..4 {
                    let mut sum = 0.0;
                    for k in 0..4 {
                        sum += black_box(a[i * 4 + k] * b[k * 4 + j]);
                    }
                    result[i * 4 + j] = sum;
                }
            }
        });
    });

    if has_native_vector_path::<f32>() {
        group.bench_function("native_checked", |bench| {
            bench.iter(|| {
                matrix_mul_square::<f32, 4>(black_box(&a), black_box(&b), black_box(&mut result));
            });
        });
    }

    group.finish();
}

/// Benchmark statistical operations.
fn bench_statistical_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_operations");
    group.sample_size(SIMD_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(SIMD_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(SIMD_WARM_UP_MILLIS));

    for size in [64, 256, 1024, 4096, 16384].iter() {
        let (data, _) = generate_test_data(*size);

        group.throughput(Throughput::Elements(*size as u64));

        // Sum benchmarks
        group.bench_with_input(
            BenchmarkId::new("sum_vectorized", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(sum(black_box(&data)));
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("sum_scalar", size), size, |bench, _| {
            bench.iter(|| {
                let result: f32 = black_box(&data).iter().sum();
                black_box(result);
            });
        });

        // Mean benchmarks
        group.bench_with_input(
            BenchmarkId::new("mean_vectorized", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(mean(black_box(&data)));
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("mean_scalar", size), size, |bench, _| {
            bench.iter(|| {
                let sum: f32 = black_box(&data).iter().sum();
                let result = sum / data.len() as f32;
                black_box(result);
            });
        });

        // Variance benchmarks (only for sizes that work well with SIMD)
        if *size >= 64 {
            group.bench_with_input(
                BenchmarkId::new("variance_vectorized", size),
                size,
                |bench, _| {
                    bench.iter(|| {
                        black_box(variance(black_box(&data)));
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("variance_scalar", size),
                size,
                |bench, _| {
                    bench.iter(|| {
                        let mean: f32 = black_box(&data).iter().sum::<f32>() / data.len() as f32;
                        let result: f32 = black_box(&data)
                            .iter()
                            .map(|x| (x - mean).powi(2))
                            .sum::<f32>()
                            / data.len() as f32;
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark SIMD capability detection.
fn bench_capability_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("capability_detection");
    group.sample_size(SIMD_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(SIMD_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(SIMD_WARM_UP_MILLIS));

    group.bench_function("avx2_detection", |bench| {
        bench.iter(|| {
            black_box(has_avx2_support());
        });
    });

    group.bench_function("neon_detection", |bench| {
        bench.iter(|| {
            black_box(has_neon_support());
        });
    });

    group.finish();
}

/// Benchmark performance counter operations.
fn bench_performance_counters(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_counters");
    group.sample_size(SIMD_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(SIMD_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(SIMD_WARM_UP_MILLIS));

    let counter = global_simd_counter();

    group.bench_function("record_vectorized_op", |bench| {
        bench.iter(|| {
            counter.record_vectorized_op(black_box(1024));
        });
    });

    group.bench_function("record_scalar_op", |bench| {
        bench.iter(|| {
            counter.record_scalar_op(black_box(1024));
        });
    });

    group.bench_function("get_stats", |bench| {
        bench.iter(|| {
            black_box(counter.get_stats());
        });
    });

    group.bench_function("simd_utilization_ratio", |bench| {
        bench.iter(|| {
            black_box(counter.simd_utilization_ratio());
        });
    });

    group.finish();
}

/// Comprehensive SIMD vs scalar comparison.
fn bench_comprehensive_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comprehensive_comparison");
    group.sample_size(SIMD_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(SIMD_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(SIMD_WARM_UP_MILLIS));

    // Reset counters for clean measurement
    global_simd_counter().reset();

    let size = 4096;
    let (a, b) = generate_test_data(size);
    let mut result = vec![0.0f32; size];

    group.throughput(Throughput::Elements(size as u64));

    // Mixed workload benchmark
    group.bench_function("mixed_workload_vectorized", |bench| {
        bench.iter(|| {
            // Addition
            add(black_box(&a), black_box(&b), black_box(&mut result));

            // Multiplication
            mul(black_box(&a), black_box(&b), black_box(&mut result));

            // Dot product
            let _dot = dot(black_box(&a), black_box(&b));

            // Statistical operations
            let _sum = sum(black_box(&a));
            let _mean = mean(black_box(&a));

            black_box(());
        });
    });

    group.bench_function("mixed_workload_scalar", |bench| {
        bench.iter(|| {
            // Addition
            for i in 0..size {
                result[i] = black_box(a[i] + b[i]);
            }

            // Multiplication
            for i in 0..size {
                result[i] = black_box(a[i] * b[i]);
            }

            // Dot product
            let _dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

            // Statistical operations
            let _sum: f32 = a.iter().sum();
            let _mean = _sum / a.len() as f32;

            black_box(());
        });
    });

    group.finish();
}

fn generate_wide_test_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
    let b: Vec<f64> = (0..size).map(|i| (size - i) as f64 * 0.1).collect();
    (a, b)
}

/// Benchmark wide real vector addition operations.
fn bench_vector_addition_wide(c: &mut Criterion) {
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

criterion_group!(
    simd_benches,
    bench_vector_addition,
    bench_vector_prefix_tail_addition,
    bench_vector_addition_wide,
    bench_vector_multiplication,
    bench_dot_product,
    bench_matrix_multiplication,
    bench_statistical_operations,
    bench_capability_detection,
    bench_performance_counters,
    bench_comprehensive_comparison
);

criterion_main!(simd_benches);
