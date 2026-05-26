use moirai::Moirai;
use moirai_core::TaskBuilder;
use moirai_utils::simd;
use rayon::prelude::*;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

const READY_COUNT: usize = 257;
const MAP_REDUCE_COUNT: usize = 4_096;
const WORKER_THREADS: usize = 4;
const CPU_WORK: usize = 8;

fn benchmark_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn read_benchmark(relative: &str) -> String {
    fs::read_to_string(benchmark_path(relative)).expect("benchmark source must be readable")
}

fn read_result_handle_diagnostics() -> String {
    let mut source = read_benchmark("benches/result_handle_diagnostics.rs");
    for relative in [
        "benches/result_handle_diagnostics/mod.rs",
        "benches/result_handle_diagnostics/types.rs",
        "benches/result_handle_diagnostics/support.rs",
        "benches/result_handle_diagnostics/result_paths.rs",
        "benches/result_handle_diagnostics/async_state.rs",
        "benches/result_handle_diagnostics/scheduler_paths.rs",
        "benches/result_handle_diagnostics/scheduler_submission_diagnostics.rs",
        "benches/result_handle_diagnostics/scheduler_lifecycle.rs",
        "benches/result_handle_diagnostics/wrapper_registry.rs",
        "benches/result_handle_diagnostics/benchmark.rs",
    ] {
        source.push('\n');
        source.push_str(&read_benchmark(relative));
    }
    source
}

fn manifest_section<'a>(content: &'a str, header: &str) -> &'a str {
    let Some(start) = content.find(header) else {
        return "";
    };
    let body = &content[start + header.len()..];
    let end = body.find("\n[").unwrap_or(body.len());
    &body[..end]
}

fn manifest_section_declares_dependency(section: &str, dependency: &str) -> bool {
    let spaced = format!("{dependency} ");
    let assigned = format!("{dependency}=");
    section.lines().any(|line| {
        let trimmed = line.trim_start();
        !trimmed.starts_with('#')
            && (trimmed.starts_with(&spaced) || trimmed.starts_with(&assigned))
    })
}

fn expected_ready_sum(count: usize) -> usize {
    count * (count + 1) / 2
}

fn cpu_work(seed: usize) -> u64 {
    let mut value = seed as u64;
    for index in 0..CPU_WORK {
        value = value.wrapping_add((index as u64).wrapping_mul(31));
    }
    value
}

fn expected_cpu_work_sum(work_items: usize) -> u64 {
    let work_items = work_items as u64;
    let per_item_offset = 31u64 * (CPU_WORK as u64) * ((CPU_WORK - 1) as u64) / 2;
    work_items * per_item_offset + work_items * (work_items - 1) / 2
}
