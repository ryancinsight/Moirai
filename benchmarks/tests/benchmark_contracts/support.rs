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

/// Pre-split module aliases: contract tests address a module by its original
/// single-file path; after a vertical split the alias maps to every leaf file
/// so source markers keep matching the whole module. A new split adds one
/// table row, not a new branch.
const SPLIT_MODULE_ALIASES: &[(&str, &[&str])] = &[
    (
        "../moirai/src/lib.rs",
        &[
            "../moirai/src/lib.rs",
            "../moirai/src/runtime.rs",
            "../moirai/src/builder.rs",
            "../moirai/src/global.rs",
            "../moirai/src/prelude.rs",
            "../moirai/src/tests.rs",
        ],
    ),
    (
        "../moirai-core/src/communication.rs",
        &[
            "../moirai-core/src/communication/mod.rs",
            "../moirai-core/src/communication/message.rs",
            "../moirai-core/src/communication/broadcast.rs",
            "../moirai-core/src/communication/collective.rs",
            "../moirai-core/src/communication/ring_buffer.rs",
            "../moirai-core/src/communication/pubsub.rs",
            "../moirai-core/src/communication/router.rs",
            "../moirai-core/src/communication/tests.rs",
        ],
    ),
    (
        "../moirai-core/src/unified_channel.rs",
        &[
            "../moirai-core/src/unified_channel/mod.rs",
            "../moirai-core/src/unified_channel/error.rs",
            "../moirai-core/src/unified_channel/config.rs",
            "../moirai-core/src/unified_channel/stats.rs",
            "../moirai-core/src/unified_channel/core.rs",
            "../moirai-core/src/unified_channel/sender.rs",
            "../moirai-core/src/unified_channel/receiver.rs",
            "../moirai-core/src/unified_channel/tests.rs",
        ],
    ),
    (
        "../moirai-core/src/task.rs",
        &[
            "../moirai-core/src/task/mod.rs",
            "../moirai-core/src/task/handle.rs",
            "../moirai-core/src/task/traits.rs",
            "../moirai-core/src/task/builder.rs",
            "../moirai-core/src/task/future.rs",
            "../moirai-core/src/task/id_and_context.rs",
            "../moirai-core/src/task/ext.rs",
        ],
    ),
    (
        "../moirai-core/src/scheduler.rs",
        &[
            "../moirai-core/src/scheduler/mod.rs",
            "../moirai-core/src/scheduler/traits.rs",
            "../moirai-core/src/scheduler/config.rs",
            "../moirai-core/src/scheduler/task.rs",
        ],
    ),
    (
        "../moirai-executor/src/hybrid/mod.rs",
        &[
            "../moirai-executor/src/hybrid/mod.rs",
            "../moirai-executor/src/hybrid/async_state.rs",
            "../moirai-executor/src/hybrid/control.rs",
            "../moirai-executor/src/hybrid/spawner.rs",
            "../moirai-executor/src/hybrid/manager.rs",
            "../moirai-executor/src/hybrid/tests.rs",
        ],
    ),
    (
        "../moirai-executor/src/registry/mod.rs",
        &[
            "../moirai-executor/src/registry/mod.rs",
            "../moirai-executor/src/registry/state.rs",
            "../moirai-executor/src/registry/token.rs",
            "../moirai-executor/src/registry/registry.rs",
            "../moirai-executor/src/registry/diagnostics.rs",
            "../moirai-executor/src/registry/tests.rs",
        ],
    ),
    (
        "../moirai-executor/src/schedule/runtime/mod.rs",
        &[
            "../moirai-executor/src/schedule/runtime/mod.rs",
            "../moirai-executor/src/schedule/runtime/scheduler/mod.rs",
            "../moirai-executor/src/schedule/runtime/scheduler/core.rs",
            "../moirai-executor/src/schedule/runtime/scheduler/scope.rs",
            "../moirai-executor/src/schedule/runtime/types.rs",
            "../moirai-executor/src/schedule/runtime/worker.rs",
            "../moirai-executor/src/schedule/runtime/tests.rs",
            "../moirai-executor/src/schedule/runtime/scheduler/diagnostics.rs",
        ],
    ),
    (
        "../moirai-iter/src/async_iter.rs",
        &[
            "../moirai-iter/src/async_iter.rs",
            "../moirai-iter/src/async_iter_tests.rs",
        ],
    ),
    (
        "../moirai-iter/src/multi_system.rs",
        &[
            "../moirai-iter/src/multi_system/mod.rs",
            "../moirai-iter/src/multi_system/config.rs",
            "../moirai-iter/src/multi_system/context.rs",
            "../moirai-iter/src/multi_system/profile.rs",
            "../moirai-iter/src/multi_system/allocation.rs",
            "../moirai-iter/src/multi_system/scheduler.rs",
            "../moirai-iter/src/multi_system/resource.rs",
            "../moirai-iter/src/multi_system/optimizer.rs",
            "../moirai-iter/src/multi_system/balancer.rs",
            "../moirai-iter/src/multi_system/iter.rs",
            "../moirai-iter/src/multi_system/tests.rs",
        ],
    ),
    (
        "../moirai-iter/src/distributed.rs",
        &[
            "../moirai-iter/src/distributed/mod.rs",
            "../moirai-iter/src/distributed/config.rs",
            "../moirai-iter/src/distributed/context.rs",
            "../moirai-iter/src/distributed/scheduler.rs",
            "../moirai-iter/src/distributed/balancer.rs",
            "../moirai-iter/src/distributed/failure.rs",
            "../moirai-iter/src/distributed/iter.rs",
            "../moirai-iter/src/distributed/tests.rs",
        ],
    ),
    (
        "../moirai-iter/src/execution/mod.rs",
        &[
            "../moirai-iter/src/execution/mod.rs",
            "../moirai-iter/src/execution/base.rs",
            "../moirai-iter/src/execution/parallel.rs",
            "../moirai-iter/src/execution/async_ctx.rs",
            "../moirai-iter/src/execution/hybrid.rs",
            "../moirai-iter/src/execution/tests.rs",
        ],
    ),
    (
        "../moirai-iter/src/parallel/adapters.rs",
        &[
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
        ],
    ),
    (
        "../moirai-scheduler/src/lib.rs",
        &[
            "../moirai-scheduler/src/lib.rs",
            "../moirai-scheduler/src/deque/mod.rs",
            "../moirai-scheduler/src/deque/reclaim.rs",
            "../moirai-scheduler/src/deque/chase_lev.rs",
            "../moirai-scheduler/src/deque/block_based.rs",
            "../moirai-scheduler/src/deque/split.rs",
            "../moirai-scheduler/src/deque/tests.rs",
        ],
    ),
    (
        "../moirai-scheduler/src/numa_scheduler.rs",
        &[
            "../moirai-scheduler/src/numa_scheduler/mod.rs",
            "../moirai-scheduler/src/numa_scheduler/topology.rs",
            "../moirai-scheduler/src/numa_scheduler/backoff.rs",
        ],
    ),
    (
        "../moirai-pal/src/reactor.rs",
        &[
            "../moirai-pal/src/reactor/mod.rs",
            "../moirai-pal/src/reactor/core.rs",
            "../moirai-pal/src/reactor/future.rs",
            "../moirai-pal/src/reactor/task.rs",
            "../moirai-pal/src/reactor/metrics.rs",
            "../moirai-pal/src/reactor/tls.rs",
            "../moirai-pal/src/reactor/tests.rs",
        ],
    ),
    (
        "../moirai-core/src/dtype.rs",
        &[
            "../moirai-core/src/dtype/mod.rs",
            "../moirai-core/src/dtype/base.rs",
            "../moirai-core/src/dtype/integer.rs",
            "../moirai-core/src/dtype/float.rs",
            "../moirai-core/src/dtype/context.rs",
            "../moirai-core/src/dtype/tests.rs",
        ],
    ),
    (
        "../moirai-core/src/memory.rs",
        &[
            "../moirai-core/src/memory/mod.rs",
            "../moirai-core/src/memory/allocator.rs",
            "../moirai-core/src/memory/pool.rs",
            "../moirai-core/src/memory/buffer.rs",
            "../moirai-core/src/memory/tests.rs",
        ],
    ),
    (
        "../moirai-core/src/ipc.rs",
        &[
            "../moirai-core/src/ipc/mod.rs",
            "../moirai-core/src/ipc/error.rs",
            "../moirai-core/src/ipc/memory.rs",
            "../moirai-core/src/ipc/queue.rs",
            "../moirai-core/src/ipc/tests.rs",
        ],
    ),
    (
        "../moirai-async/src/executor.rs",
        &[
            "../moirai-async/src/executor/mod.rs",
            "../moirai-async/src/executor/core.rs",
            "../moirai-async/src/executor/task.rs",
            "../moirai-async/src/executor/stats.rs",
            "../moirai-async/src/executor/waker.rs",
            "../moirai-async/src/executor/handle.rs",
            "../moirai-async/src/executor/result_slot.rs",
        ],
    ),
    (
        "../moirai-async/src/timer.rs",
        &[
            "../moirai-async/src/timer/mod.rs",
            "../moirai-async/src/timer/delay.rs",
            "../moirai-async/src/timer/registration.rs",
            "../moirai-async/src/timer/driver.rs",
            "../moirai-async/src/timer/timeout.rs",
            "../moirai-async/src/timer/interval.rs",
            "../moirai-async/src/timer/limiter.rs",
            "../moirai-async/src/timer/wheel.rs",
        ],
    ),
    (
        "../moirai-async/src/fs.rs",
        &[
            "../moirai-async/src/fs/mod.rs",
            "../moirai-async/src/fs/options.rs",
            "../moirai-async/src/fs/stats.rs",
            "../moirai-async/src/fs/file.rs",
            "../moirai-async/src/fs/ops.rs",
        ],
    ),
    (
        "../moirai-async/src/io.rs",
        &[
            "../moirai-async/src/io/mod.rs",
            "../moirai-async/src/io/traits.rs",
            "../moirai-async/src/io/ext.rs",
            "../moirai-async/src/io/compat.rs",
        ],
    ),
    (
        "../moirai-async/src/net.rs",
        &[
            "../moirai-async/src/net/mod.rs",
            "../moirai-async/src/net/listener.rs",
            "../moirai-async/src/net/stream.rs",
            "../moirai-async/src/net/socket.rs",
            "../moirai-async/src/net/types.rs",
        ],
    ),
    (
        "../moirai-async/src/sync.rs",
        &[
            "../moirai-async/src/sync/mod.rs",
            "../moirai-async/src/sync/broadcast.rs",
            "../moirai-async/src/sync/notify.rs",
            "../moirai-async/src/sync/rwlock.rs",
            "../moirai-async/src/sync/semaphore.rs",
            "../moirai-async/src/sync/watch.rs",
        ],
    ),
];

fn read_benchmark(relative: &str) -> String {
    let normalized = relative.replace('\\', "/");
    if let Some((_, leaves)) = SPLIT_MODULE_ALIASES
        .iter()
        .find(|(alias, _)| *alias == normalized)
    {
        let mut content = String::new();
        for file in *leaves {
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
