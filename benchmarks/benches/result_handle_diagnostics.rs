//! Diagnostic benchmark for public result-handle overhead.
//!
//! This target separates the one-shot result slot from scheduler submission.
//! It is not a competitive benchmark; it exists to locate the next bottleneck
//! in `Moirai::spawn_fn(...).join()` without changing the public workload.

#[path = "result_handle_diagnostics/mod.rs"]
mod diagnostics;

#[path = "common/time_model.rs"]
mod time_model;
use criterion::{criterion_group, criterion_main};
use diagnostics::benchmark_result_handle_diagnostics;

criterion_group! {
    name = benches;
    config = time_model::criterion();
    targets = benchmark_result_handle_diagnostics
}

criterion_main!(benches);
