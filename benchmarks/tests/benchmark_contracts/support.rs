use moirai::Moirai;
use moirai_core::TaskBuilder;
use moirai_utils::simd;
use rayon::prelude::*;
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

const READY_COUNT: usize = 257;
const MAP_REDUCE_COUNT: usize = 4_096;
const WORKER_THREADS: usize = 4;
const CPU_WORK: usize = 8;

/// Run-time path resolution, retained for the one audit that asserts a file's
/// *absence*: `include_str!` cannot express "this path must not exist", since
/// naming a missing file is a compile error rather than a readable fact. Every
/// audit that reads content goes through [`EMBEDDED_SOURCES`] instead.
///
/// `env!("CARGO_MANIFEST_DIR")` resolves when this binary is compiled, so this
/// single check still reports on whichever worktree last built the binary.
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
        "../moirai-core/src/channel/unified.rs",
        &[
            "../moirai-core/src/channel/unified/mod.rs",
            "../moirai-core/src/channel/unified/core.rs",
            "../moirai-core/src/channel/unified/sender.rs",
            "../moirai-core/src/channel/unified/receiver.rs",
            "../moirai-core/src/channel/unified/tests.rs",
        ],
    ),
    (
        "../moirai-core/src/channel/stats.rs",
        &["../moirai-core/src/channel/stats.rs"],
    ),
    (
        "../moirai-core/src/channel/config.rs",
        &["../moirai-core/src/channel/config.rs"],
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
        &["../moirai-core/src/scheduler/mod.rs"],
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
            "../moirai-executor/src/schedule/runtime/scheduler/data_parallel.rs",
            "../moirai-executor/src/schedule/runtime/scheduler/scope.rs",
            "../moirai-executor/src/schedule/runtime/types.rs",
            "../moirai-executor/src/schedule/runtime/worker.rs",
            "../moirai-executor/src/schedule/runtime/worker/indexed.rs",
            "../moirai-executor/src/schedule/runtime/worker/wait.rs",
            "../moirai-executor/src/schedule/runtime/tests.rs",
            "../moirai-executor/src/schedule/runtime/scheduler/diagnostics.rs",
        ],
    ),
    (
        "../moirai-iter/src/async_iter.rs",
        &["../moirai-iter/src/async_iter_tests.rs"],
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
            "../moirai-scheduler/src/deque/chase_lev/storage.rs",
            "../moirai-scheduler/src/deque/split.rs",
            "../moirai-scheduler/src/deque/tests.rs",
        ],
    ),
    (
        "../moirai-scheduler/src/numa.rs",
        &[
            "../moirai-scheduler/src/numa/mod.rs",
            "../moirai-scheduler/src/numa/topology.rs",
            "../moirai-scheduler/src/numa/backoff.rs",
        ],
    ),
    (
        "../moirai-pal/src/reactor.rs",
        &[
            "../moirai-pal/src/reactor/mod.rs",
            "../moirai-pal/src/reactor/core.rs",
            "../moirai-pal/src/reactor/metrics.rs",
            "../moirai-pal/src/reactor/tls.rs",
            "../moirai-pal/src/reactor/tests.rs",
        ],
    ),
    (
        "../moirai-pal/src/fs.rs",
        &[
            "../moirai-pal/src/fs/mod.rs",
            "../moirai-pal/src/fs/file.rs",
            "../moirai-pal/src/fs/path.rs",
            "../moirai-pal/src/fs/tests.rs",
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
    (
        "../moirai-transport/src/route.rs",
        &[
            "../moirai-transport/src/route.rs",
            "../moirai-transport/src/route/process_client.rs",
        ],
    ),
    (
        "../moirai-utils/src/simd/scalar.rs",
        &[
            "../moirai-utils/src/simd/scalar.rs",
            "../moirai-utils/src/simd/scalar/capability.rs",
            "../moirai-utils/src/simd/scalar/validation.rs",
        ],
    ),
    (
        "benches/simd_benchmarks.rs",
        &[
            "benches/simd_benchmarks.rs",
            "benches/simd_benchmarks/wide.rs",
        ],
    ),
];

/// Audited sources, embedded at compile time.
///
/// A contract asserts that named shapes are present in, or absent from, a
/// source file. Reading that file at run time made the verdict independent of
/// the binary: Cargo saw no dependency on the audited source, so an edit never
/// rebuilt the test, and the root the binary read was `CARGO_MANIFEST_DIR`
/// baked at compile time — under one shared `CARGO_TARGET_DIR` a binary built
/// in one worktree audited that worktree from any other.
///
/// `include_str!` closes both: Cargo tracks each included file as a build
/// dependency, so editing an audited source rebuilds this binary, and the
/// paths resolve relative to *this file* rather than to an environment
/// variable, so the audit reads the tree it was compiled from. A path naming
/// a file that no longer exists is a compile error instead of a silently
/// empty read.
///
/// Paths are written relative to the `benchmarks` package root, matching how
/// contracts address them; `../../` re-anchors that on this file's directory.
macro_rules! embedded_sources {
    ($($relative:literal),+ $(,)?) => {
        const EMBEDDED_SOURCES: &[(&str, &str)] = &[
            $(($relative, include_str!(concat!("../../", $relative))),)+
        ];
    };
}

embedded_sources![
    "../CHANGELOG.md",
    "../Cargo.toml",
    "../GAP_ANALYSIS.md",
    "../PERFORMANCE_RESULTS.md",
    "../docs/adr-008-checklist.md",
    "../docs/adr/0008-scheduler-route-consumption-and-transport-ownership-boundary.md",
    "../docs/checklist.md",
    "../docs/moirai_rayon_tokio_comparison.md",
    "../docs/rayon_adapter_surface_audit.md",
    "../docs/rayon_tokio_gap_audit.md",
    "../moirai-async/Cargo.toml",
    "../moirai-async/src/executor/core.rs",
    "../moirai-async/src/executor/handle.rs",
    "../moirai-async/src/executor/mod.rs",
    "../moirai-async/src/executor/result_slot.rs",
    "../moirai-async/src/executor/stats.rs",
    "../moirai-async/src/executor/task.rs",
    "../moirai-async/src/executor/waker.rs",
    "../moirai-async/src/fs/file.rs",
    "../moirai-async/src/fs/mod.rs",
    "../moirai-async/src/fs/ops.rs",
    "../moirai-async/src/fs/options.rs",
    "../moirai-async/src/fs/stats.rs",
    "../moirai-async/src/fs/tests.rs",
    "../moirai-async/src/io/compat.rs",
    "../moirai-async/src/io/ext.rs",
    "../moirai-async/src/io/mod.rs",
    "../moirai-async/src/io/tests.rs",
    "../moirai-async/src/io/traits.rs",
    "../moirai-async/src/net/listener.rs",
    "../moirai-async/src/net/mod.rs",
    "../moirai-async/src/net/socket.rs",
    "../moirai-async/src/net/stream.rs",
    "../moirai-async/src/net/tests.rs",
    "../moirai-async/src/net/types.rs",
    "../moirai-async/src/sync/broadcast.rs",
    "../moirai-async/src/sync/mod.rs",
    "../moirai-async/src/sync/notify.rs",
    "../moirai-async/src/sync/rwlock.rs",
    "../moirai-async/src/sync/semaphore.rs",
    "../moirai-async/src/sync/watch.rs",
    "../moirai-async/src/timer/delay.rs",
    "../moirai-async/src/timer/driver.rs",
    "../moirai-async/src/timer/interval.rs",
    "../moirai-async/src/timer/limiter.rs",
    "../moirai-async/src/timer/mod.rs",
    "../moirai-async/src/timer/registration.rs",
    "../moirai-async/src/timer/timeout.rs",
    "../moirai-async/src/timer/wheel.rs",
    "../moirai-core/Cargo.toml",
    "../moirai-core/src/channel/config.rs",
    "../moirai-core/src/channel/stats.rs",
    "../moirai-core/src/channel/unified/core.rs",
    "../moirai-core/src/channel/unified/mod.rs",
    "../moirai-core/src/channel/unified/receiver.rs",
    "../moirai-core/src/channel/unified/sender.rs",
    "../moirai-core/src/channel/unified/tests.rs",
    "../moirai-core/src/communication/broadcast.rs",
    "../moirai-core/src/communication/collective.rs",
    "../moirai-core/src/communication/message.rs",
    "../moirai-core/src/communication/mod.rs",
    "../moirai-core/src/communication/pubsub.rs",
    "../moirai-core/src/communication/ring_buffer.rs",
    "../moirai-core/src/communication/router.rs",
    "../moirai-core/src/communication/tests.rs",
    "../moirai-core/src/ipc/error.rs",
    "../moirai-core/src/ipc/memory.rs",
    "../moirai-core/src/ipc/mod.rs",
    "../moirai-core/src/ipc/queue.rs",
    "../moirai-core/src/ipc/tests.rs",
    "../moirai-core/src/lib.rs",
    "../moirai-core/src/memory/allocator.rs",
    "../moirai-core/src/memory/buffer.rs",
    "../moirai-core/src/memory/mod.rs",
    "../moirai-core/src/memory/pool.rs",
    "../moirai-core/src/memory/tests.rs",
    "../moirai-core/src/scheduler/mod.rs",
    "../moirai-core/src/task/builder.rs",
    "../moirai-core/src/task/ext.rs",
    "../moirai-core/src/task/future.rs",
    "../moirai-core/src/task/handle.rs",
    "../moirai-core/src/task/id_and_context.rs",
    "../moirai-core/src/task/mod.rs",
    "../moirai-core/src/task/traits.rs",
    "../moirai-executor/Cargo.toml",
    "../moirai-executor/src/hybrid/async_state.rs",
    "../moirai-executor/src/hybrid/control.rs",
    "../moirai-executor/src/hybrid/manager.rs",
    "../moirai-executor/src/hybrid/mod.rs",
    "../moirai-executor/src/hybrid/spawner.rs",
    "../moirai-executor/src/hybrid/tests.rs",
    "../moirai-executor/src/lib.rs",
    "../moirai-executor/src/registry/diagnostics.rs",
    "../moirai-executor/src/registry/mod.rs",
    "../moirai-executor/src/registry/registry.rs",
    "../moirai-executor/src/registry/state.rs",
    "../moirai-executor/src/registry/tests.rs",
    "../moirai-executor/src/registry/token.rs",
    "../moirai-executor/src/schedule/class/mod.rs",
    "../moirai-executor/src/schedule/job/mod.rs",
    "../moirai-executor/src/schedule/mod.rs",
    "../moirai-executor/src/schedule/queue/mod.rs",
    "../moirai-executor/src/schedule/route/decision.rs",
    "../moirai-executor/src/schedule/route/ids.rs",
    "../moirai-executor/src/schedule/route/mod.rs",
    "../moirai-executor/src/schedule/route/policy.rs",
    "../moirai-executor/src/schedule/route/router.rs",
    "../moirai-executor/src/schedule/route/summary.rs",
    "../moirai-executor/src/schedule/route/tests.rs",
    "../moirai-executor/src/schedule/route/topology.rs",
    "../moirai-executor/src/schedule/runtime/mod.rs",
    "../moirai-executor/src/schedule/runtime/scheduler/core.rs",
    "../moirai-executor/src/schedule/runtime/scheduler/data_parallel.rs",
    "../moirai-executor/src/schedule/runtime/scheduler/diagnostics.rs",
    "../moirai-executor/src/schedule/runtime/scheduler/mod.rs",
    "../moirai-executor/src/schedule/runtime/scheduler/scope.rs",
    "../moirai-executor/src/schedule/runtime/tests.rs",
    "../moirai-executor/src/schedule/runtime/types.rs",
    "../moirai-executor/src/schedule/runtime/worker.rs",
    "../moirai-executor/src/schedule/runtime/worker/indexed.rs",
    "../moirai-executor/src/schedule/runtime/worker/wait.rs",
    "../moirai-gpu/Cargo.toml",
    "../moirai-gpu/src/task.rs",
    "../moirai-iter/Cargo.toml",
    "../moirai-iter/src/async_iter/adapters.rs",
    "../moirai-iter/src/async_iter/consumers.rs",
    "../moirai-iter/src/async_iter/mod.rs",
    "../moirai-iter/src/async_iter/parallel.rs",
    "../moirai-iter/src/async_iter/sources.rs",
    "../moirai-iter/src/async_iter/traits.rs",
    "../moirai-iter/src/async_iter_tests.rs",
    "../moirai-iter/src/base.rs",
    "../moirai-iter/src/base/tests.rs",
    "../moirai-iter/src/cache.rs",
    "../moirai-iter/src/channel_fusion.rs",
    "../moirai-iter/src/execution/async_ctx.rs",
    "../moirai-iter/src/execution/base.rs",
    "../moirai-iter/src/execution/hybrid.rs",
    "../moirai-iter/src/execution/mod.rs",
    "../moirai-iter/src/execution/parallel.rs",
    "../moirai-iter/src/execution/tests.rs",
    "../moirai-iter/src/facade/mod.rs",
    "../moirai-iter/src/iter_ops.rs",
    "../moirai-iter/src/iter_ops/parallel.rs",
    "../moirai-iter/src/iter_ops/streaming.rs",
    "../moirai-iter/src/iter_ops/tests.rs",
    "../moirai-iter/src/lib.rs",
    "../moirai-iter/src/parallel.rs",
    "../moirai-iter/src/parallel/adapters/blocks.rs",
    "../moirai-iter/src/parallel/adapters/chunks.rs",
    "../moirai-iter/src/parallel/adapters/filter.rs",
    "../moirai-iter/src/parallel/adapters/flat.rs",
    "../moirai-iter/src/parallel/adapters/map.rs",
    "../moirai-iter/src/parallel/adapters/mod.rs",
    "../moirai-iter/src/parallel/adapters/pair.rs",
    "../moirai-iter/src/parallel/adapters/position.rs",
    "../moirai-iter/src/parallel/adapters/ref_ops.rs",
    "../moirai-iter/src/parallel/adapters/side_effect.rs",
    "../moirai-iter/src/parallel/adapters/slice_ops.rs",
    "../moirai-iter/src/parallel/adapters/stride.rs",
    "../moirai-iter/src/parallel/adapters/window.rs",
    "../moirai-iter/src/parallel/consumers.rs",
    "../moirai-iter/src/parallel/fallible.rs",
    "../moirai-iter/src/parallel/indexed.rs",
    "../moirai-iter/src/parallel/sorting.rs",
    "../moirai-iter/src/parallel/sources.rs",
    "../moirai-iter/src/parallel/split.rs",
    "../moirai-iter/src/parallel/tests.rs",
    "../moirai-iter/src/parallel/traits.rs",
    "../moirai-iter/src/simd_iter.rs",
    "../moirai-metrics/Cargo.toml",
    "../moirai-metrics/src/collector.rs",
    "../moirai-metrics/src/counter.rs",
    "../moirai-metrics/src/exporter.rs",
    "../moirai-metrics/src/gauge.rs",
    "../moirai-metrics/src/histogram.rs",
    "../moirai-metrics/src/lib.rs",
    "../moirai-metrics/src/tests.rs",
    "../moirai-pal/Cargo.toml",
    "../moirai-pal/src/fs/file.rs",
    "../moirai-pal/src/fs/mod.rs",
    "../moirai-pal/src/fs/path.rs",
    "../moirai-pal/src/fs/tests.rs",
    "../moirai-pal/src/lib.rs",
    "../moirai-pal/src/net.rs",
    "../moirai-pal/src/reactor/core.rs",
    "../moirai-pal/src/reactor/kqueue_transition.rs",
    "../moirai-pal/src/reactor/metrics.rs",
    "../moirai-pal/src/reactor/mod.rs",
    "../moirai-pal/src/reactor/registration.rs",
    "../moirai-pal/src/reactor/tests.rs",
    "../moirai-pal/src/reactor/tls.rs",
    "../moirai-pal/src/timer.rs",
    "../moirai-pal/src/timer/tests.rs",
    "../moirai-pal/src/unix/epoll.rs",
    "../moirai-pal/src/unix/kqueue.rs",
    "../moirai-pal/src/windows/poll.rs",
    "../moirai-parallel/benches/par_benchmarks.rs",
    "../moirai-parallel/src/ops.rs",
    "../moirai-parallel/src/policy.rs",
    "../moirai-scheduler/Cargo.toml",
    "../moirai-scheduler/src/deque/chase_lev.rs",
    "../moirai-scheduler/src/deque/chase_lev/storage.rs",
    "../moirai-scheduler/src/deque/mod.rs",
    "../moirai-scheduler/src/deque/reclaim.rs",
    "../moirai-scheduler/src/deque/split.rs",
    "../moirai-scheduler/src/deque/tests.rs",
    "../moirai-scheduler/src/lib.rs",
    "../moirai-scheduler/src/numa/backoff.rs",
    "../moirai-scheduler/src/numa/mod.rs",
    "../moirai-scheduler/src/numa/topology.rs",
    "../moirai-sync/Cargo.toml",
    "../moirai-transport/Cargo.toml",
    "../moirai-transport/src/lib.rs",
    "../moirai-transport/src/network.rs",
    "../moirai-transport/src/payload.rs",
    "../moirai-transport/src/process.rs",
    "../moirai-transport/src/remote_task.rs",
    "../moirai-transport/src/remote_task/capability.rs",
    "../moirai-transport/src/remote_task/server.rs",
    "../moirai-transport/src/remote_task/tests.rs",
    "../moirai-transport/src/route.rs",
    "../moirai-transport/src/route/process_client.rs",
    "../moirai-transport/src/route/tests.rs",
    "../moirai-transport/src/safe_channel.rs",
    "../moirai-transport/src/transport.rs",
    "../moirai-utils/Cargo.toml",
    "../moirai-utils/src/lib.rs",
    "../moirai-utils/src/simd/arch/mod.rs",
    "../moirai-utils/src/simd/mod.rs",
    "../moirai-utils/src/simd/scalar.rs",
    "../moirai-utils/src/simd/scalar/capability.rs",
    "../moirai-utils/src/simd/scalar/validation.rs",
    "../moirai-utils/src/simd/tests.rs",
    "../moirai/Cargo.toml",
    "../moirai/src/builder.rs",
    "../moirai/src/global.rs",
    "../moirai/src/lib.rs",
    "../moirai/src/prelude.rs",
    "../moirai/src/routed.rs",
    "../moirai/src/runtime.rs",
    "../moirai/src/tests.rs",
    "Cargo.toml",
    "benches/async_fs_comparison.rs",
    "benches/async_fs_dir_comparison.rs",
    "benches/async_io_compat_comparison.rs",
    "benches/async_iterator_comparison.rs",
    "benches/async_tcp_backpressure_comparison.rs",
    "benches/async_tcp_cancel_safety_comparison.rs",
    "benches/async_tcp_comparison.rs",
    "benches/async_tcp_readiness_comparison.rs",
    "benches/async_udp_comparison.rs",
    "benches/cache_iterator_comparison.rs",
    "benches/channel_matrix.rs",
    "benches/example_pattern_comparison.rs",
    "benches/execution_context_comparison.rs",
    "benches/industry_comparison.rs",
    "benches/iter_ops_parallel_comparison.rs",
    "benches/iter_simd_comparison.rs",
    "benches/iterator_adapter_comparison.rs",
    "benches/metrics_collector_comparison.rs",
    "benches/moirai_benchmarks.rs",
    "benches/parallel_iterator_regression.rs",
    "benches/performance_benchmarks.rs",
    "benches/process_server_routed_execution.rs",
    "benches/process_server_scheduler_routing.rs",
    "benches/public_result_handle_comparison.rs",
    "benches/result_handle_diagnostics.rs",
    "benches/result_handle_diagnostics/async_state.rs",
    "benches/result_handle_diagnostics/benchmark.rs",
    "benches/result_handle_diagnostics/mod.rs",
    "benches/result_handle_diagnostics/registry_paths.rs",
    "benches/result_handle_diagnostics/result_paths.rs",
    "benches/result_handle_diagnostics/scheduled_wrapper_paths.rs",
    "benches/result_handle_diagnostics/scheduler_lifecycle.rs",
    "benches/result_handle_diagnostics/scheduler_paths.rs",
    "benches/result_handle_diagnostics/scheduler_submission_diagnostics.rs",
    "benches/result_handle_diagnostics/scheduler_tail_paths.rs",
    "benches/result_handle_diagnostics/support.rs",
    "benches/result_handle_diagnostics/types.rs",
    "benches/result_handle_diagnostics/wrapper_primitives.rs",
    "benches/result_handle_diagnostics/wrapper_registry.rs",
    "benches/simd_benchmarks.rs",
    "benches/simd_benchmarks/wide.rs",
    "benches/sorting_comparison.rs",
    "benches/thread_schedule_comparison.rs",
    "benches/thread_schedule_comparison/dispatch_floor.rs",
    "benches/thread_schedule_comparison/local_queue_capacity.rs",
    "benches/transport_archive_comparison.rs",
    "tests/benchmark_contracts/support.rs",
];

fn embedded_source(relative: &str) -> &'static str {
    EMBEDDED_SOURCES
        .iter()
        .find(|(path, _)| *path == relative)
        .map(|(_, content)| *content)
        .unwrap_or_else(|| {
            panic!("audited source {relative} must be listed in EMBEDDED_SOURCES")
        })
}

fn normalize_newlines(source: &str) -> String {
    source.replace("\r\n", "\n").replace('\r', "\n")
}

fn read_benchmark(relative: &str) -> String {
    let normalized = relative.replace('\\', "/");
    if let Some((_, leaves)) = SPLIT_MODULE_ALIASES
        .iter()
        .find(|(alias, _)| *alias == normalized)
    {
        let mut content = String::new();
        for file in *leaves {
            content.push_str(embedded_source(file));
            content.push('\n');
        }
        return normalize_newlines(&content);
    }

    normalize_newlines(embedded_source(&normalized))
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
