//! Diagnostic benchmark for public result-handle overhead.
//!
//! This target separates the one-shot result slot from scheduler submission.
//! It is not a competitive benchmark; it exists to locate the next bottleneck
//! in `Moirai::spawn_fn(...).join()` without changing the public workload.

#[path = "result_handle_diagnostics/mod.rs"]
mod diagnostics;

use criterion::{criterion_group, criterion_main, Criterion};
use diagnostics::{
    benchmark_result_handle_diagnostics, BENCHMARK_MEASUREMENT_SECONDS, BENCHMARK_SAMPLE_SIZE,
    BENCHMARK_WARM_UP_MILLIS,
};
use std::time::Duration;

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(BENCHMARK_SAMPLE_SIZE)
        .measurement_time(Duration::from_secs(BENCHMARK_MEASUREMENT_SECONDS))
        .warm_up_time(Duration::from_millis(BENCHMARK_WARM_UP_MILLIS))
        .without_plots();
    targets = benchmark_result_handle_diagnostics
}

criterion_main!(benches);
