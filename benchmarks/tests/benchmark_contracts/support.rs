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
    let normalized = relative.replace('\\', "/");
    if normalized == "../moirai-core/src/task.rs" {
        let mut content = String::new();
        for file in [
            "../moirai-core/src/task/mod.rs",
            "../moirai-core/src/task/handle.rs",
            "../moirai-core/src/task/traits.rs",
            "../moirai-core/src/task/builder.rs",
            "../moirai-core/src/task/future.rs",
            "../moirai-core/src/task/id_and_context.rs",
            "../moirai-core/src/task/ext.rs",
        ] {
            if let Ok(c) = fs::read_to_string(benchmark_path(file)) {
                content.push_str(&c);
                content.push('\n');
            }
        }
        return content.replace("\r\n", "\n").replace('\r', "\n");
    }
    if normalized == "../moirai-core/src/scheduler.rs" {
        let mut content = String::new();
        for file in [
            "../moirai-core/src/scheduler/mod.rs",
            "../moirai-core/src/scheduler/coordinator.rs",
            "../moirai-core/src/scheduler/traits.rs",
            "../moirai-core/src/scheduler/buffer.rs",
            "../moirai-core/src/scheduler/config.rs",
            "../moirai-core/src/scheduler/deque.rs",
            "../moirai-core/src/scheduler/task.rs",
        ] {
            if let Ok(c) = fs::read_to_string(benchmark_path(file)) {
                content.push_str(&c);
                content.push('\n');
            }
        }
        return content.replace("\r\n", "\n").replace('\r', "\n");
    }
    if normalized == "../moirai-executor/src/hybrid/mod.rs" {
        let mut content = String::new();
        for file in [
            "../moirai-executor/src/hybrid/mod.rs",
            "../moirai-executor/src/hybrid/async_state.rs",
        ] {
            if let Ok(c) = fs::read_to_string(benchmark_path(file)) {
                content.push_str(&c);
                content.push('\n');
            }
        }
        return content.replace("\r\n", "\n").replace('\r', "\n");
    }
    if normalized == "../moirai-executor/src/schedule/runtime/mod.rs" {
        let mut content = String::new();
        for file in [
            "../moirai-executor/src/schedule/runtime/mod.rs",
            "../moirai-executor/src/schedule/runtime/scheduler.rs",
            "../moirai-executor/src/schedule/runtime/types.rs",
            "../moirai-executor/src/schedule/runtime/worker.rs",
            "../moirai-executor/src/schedule/runtime/tests.rs",
            "../moirai-executor/src/schedule/runtime/scheduler/diagnostics.rs",
        ] {
            if let Ok(c) = fs::read_to_string(benchmark_path(file)) {
                content.push_str(&c);
                content.push('\n');
            }
        }
        return content.replace("\r\n", "\n").replace('\r', "\n");
    }
    if normalized == "../moirai-iter/src/async_iter.rs" {
        let mut content = String::new();
        for file in [
            "../moirai-iter/src/async_iter.rs",
            "../moirai-iter/src/async_iter_tests.rs",
        ] {
            if let Ok(c) = fs::read_to_string(benchmark_path(file)) {
                content.push_str(&c);
                content.push('\n');
            }
        }
        return content.replace("\r\n", "\n").replace('\r', "\n");
    }
    if normalized == "../moirai-iter/src/parallel/adapters.rs" {
        let mut content = String::new();
        for file in [
            "../moirai-iter/src/parallel/adapters/mod.rs",
            "../moirai-iter/src/parallel/adapters/map.rs",
            "../moirai-iter/src/parallel/adapters/filter.rs",
            "../moirai-iter/src/parallel/adapters/flat.rs",
            "../moirai-iter/src/parallel/adapters/ref_ops.rs",
            "../moirai-iter/src/parallel/adapters/slice_ops.rs",
            "../moirai-iter/src/parallel/adapters/blocks.rs",
            "../moirai-iter/src/parallel/adapters/chunks.rs",
            "../moirai-iter/src/parallel/adapters/pair.rs",
            "../moirai-iter/src/parallel/adapters/position.rs",
            "../moirai-iter/src/parallel/adapters/side_effect.rs",
            "../moirai-iter/src/parallel/adapters/stride.rs",
            "../moirai-iter/src/parallel/adapters/window.rs",
        ] {
            if let Ok(c) = fs::read_to_string(benchmark_path(file)) {
                content.push_str(&c);
                content.push('\n');
            }
        }
        return content.replace("\r\n", "\n").replace('\r', "\n");
    }
    if normalized == "../moirai-scheduler/src/lib.rs" {
        let mut content = String::new();
        for file in [
            "../moirai-scheduler/src/lib.rs",
            "../moirai-scheduler/src/deque.rs",
            "../moirai-scheduler/src/reclaim.rs",
            "../moirai-scheduler/src/scheduler.rs",
        ] {
            if let Ok(c) = fs::read_to_string(benchmark_path(file)) {
                content.push_str(&c);
                content.push('\n');
            }
        }
        return content.replace("\r\n", "\n").replace('\r', "\n");
    }
    if normalized == "../moirai-scheduler/src/numa_scheduler.rs" {
        let mut content = String::new();
        for file in [
            "../moirai-scheduler/src/numa_scheduler/mod.rs",
            "../moirai-scheduler/src/numa_scheduler/scheduler.rs",
            "../moirai-scheduler/src/numa_scheduler/topology.rs",
            "../moirai-scheduler/src/numa_scheduler/queue.rs",
            "../moirai-scheduler/src/numa_scheduler/backoff.rs",
            "../moirai-scheduler/src/numa_scheduler/tests.rs",
        ] {
            if let Ok(c) = fs::read_to_string(benchmark_path(file)) {
                content.push_str(&c);
                content.push('\n');
            }
        }
        return content.replace("\r\n", "\n").replace('\r', "\n");
    }

    fs::read_to_string(benchmark_path(relative))
        .expect("benchmark source must be readable")
        .replace("\r\n", "\n")
        .replace('\r', "\n")
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
        "benches/result_handle_diagnostics/scheduler_tail_paths.rs",
        "benches/result_handle_diagnostics/wrapper_primitives.rs",
        "benches/result_handle_diagnostics/wrapper_registry.rs",
        "benches/result_handle_diagnostics/scheduled_wrapper_paths.rs",
        "benches/result_handle_diagnostics/registry_paths.rs",
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
