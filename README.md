# Moirai

[![crates.io](https://img.shields.io/crates/v/moirai-runtime.svg)](https://crates.io/crates/moirai-runtime)
[![docs.rs](https://docs.rs/moirai-runtime/badge.svg)](https://docs.rs/moirai-runtime)
[![Rust Workspace](https://github.com/ryancinsight/Moirai/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/ryancinsight/Moirai/actions/workflows/rust-ci.yml)

Moirai is a unified scheduler/router for Rust work placement. It routes admitted
work across local CPU worker threads, sync/blocking/async-ready work classes,
supervised process routes, server routes, and per-process async lanes while using
zero-cost, monomorphized policy types at hot boundaries. Rayon and Tokio parity
are benchmark gates; the architecture target is a single scheduler hierarchy that
can grow into GPU, TPU, NPU, and server placement without duplicating algorithms
or fabricating execution.

The facade crate publishes as **`moirai-runtime`** (the `moirai` registry name
belongs to an unrelated project) and keeps the Rust library name `moirai`.

## Quick Start

```toml
[dependencies]
moirai-runtime = "0.5"
```

The crate is imported as `moirai`:

```rust
use moirai::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let moirai = Moirai::builder()
        .worker_threads(8)
        .async_threads(4)
        .thread_name_prefix("moirai-worker")
        .build()?;

    // Spawn a closure onto the unified scheduler.
    let parallel = moirai.spawn_fn(|| 7usize);

    // Spawn a future onto the async lane.
    let handle = moirai.spawn_async(async { 42usize });

    // Indexed map/reduce through the same scheduler.
    let sum = moirai.map_reduce_indexed(
        5,
        0usize,
        |index| (index + 1) * (index + 1),
        usize::wrapping_add,
    )?;
    assert_eq!(sum, 55);

    let parallel_result = parallel.join().expect("parallel handle attached")?;
    let async_result = handle.join().expect("async handle attached")?;
    assert_eq!(parallel_result, 7);
    assert_eq!(async_result, 42);

    moirai.shutdown();
    Ok(())
}
```

`TaskHandle::join` returns `Option<Result<T, TaskError>>`: `Some(Ok(_))` on
success, `Some(Err(_))` when the task failed, and `None` when the task was
detached.

Default features: `async`, `iter`, `parallel`, `local`, `mnemosyne-memory`,
`melinoe`. Optional: `distributed`, `network`, `metrics`, `numa`, `gpu`,
`encryption`, `compression`, `tokio-compat`, `no-std`, and `full`.

Minimum supported Rust version: **1.95**. The pinned build toolchain is 1.97.0
(`rust-toolchain.toml`).

## Crates

| Crate | Role |
|-------|------|
| [`moirai-runtime`](https://docs.rs/moirai-runtime) | Facade: `Moirai` runtime, builder, prelude (library name `moirai`) |
| [`moirai-core`](https://docs.rs/moirai-core) | Task/executor/scheduler traits, channels, communication patterns, memory pools |
| [`moirai-executor`](https://docs.rs/moirai-executor) | Hybrid executor: work-class routing, blocking lane, scheduler scopes |
| [`moirai-scheduler`](https://docs.rs/moirai-scheduler) | Chase-Lev / split work-stealing deques and CPU topology discovery |
| [`moirai-transport`](https://docs.rs/moirai-transport) | In-memory, IPC, and network transports; archive message pairs |
| [`moirai-sync`](https://docs.rs/moirai-sync) | `FutexMutex`, `WaitGroup`, `ConcurrentHashMap`, sharded resource pool |
| [`moirai-async`](https://docs.rs/moirai-async) | Async runtime integration and `AsyncRead`/`AsyncWrite` |
| [`moirai-async-macros`](https://docs.rs/moirai-async-macros) | `#[main]` attribute macro for an async entry point |
| [`moirai-iter`](https://docs.rs/moirai-iter) | Parallel/async/hybrid iterator combinators |
| [`moirai-parallel`](https://docs.rs/moirai-parallel) | Synchronous data-parallel slice primitives (rayon-style surface) |
| [`moirai-pal`](https://docs.rs/moirai-pal) | Platform abstraction: epoll / kqueue / `WSAPoll` readiness |
| [`moirai-metrics`](https://docs.rs/moirai-metrics) | Cloneable metric handles and value-copy snapshots |
| [`moirai-gpu`](https://docs.rs/moirai-gpu) | Hephaestus-backed GPU task scheduling and launch-shape planning |
| [`moirai-utils`](https://docs.rs/moirai-utils) | Cache alignment, atomics, lock-free queues, prefetch |
| [`moirai-crypto`](https://docs.rs/moirai-crypto) | Pure-Rust rustls `CryptoProvider` (RustCrypto, no C toolchain) |
| [`moirai-tls`](https://docs.rs/moirai-tls) | Async TLS client over Moirai sockets, no Tokio |
| [`moirai-http`](https://docs.rs/moirai-http) | Minimal async HTTP/1.1 client over `moirai-tls` |

## Runtime

- **Hybrid executor**: sync, blocking, and async-ready work classes on one
  scheduler facade; blocking admission uses a bounded lane isolated from the
  compute worker pool.
- **Work-stealing scheduler**: per-worker Chase-Lev deques indexed by
  `Priority::index`, with lock-free stealing. The Chase-Lev steal/pop ordering
  protocol has an exhaustive `loom` model; the bounded MPMC waiter and SPSC
  ring publication protocols, plus async executor wake deduplication, are
  modeled in the channel and async executor tests.
- **Route topology** (`distributed` feature): sealed zero-sized policies select
  `SchedulerRoute::{Thread, Process, Server, Accelerator}` without
  `dyn RoutePolicy`.
- **Process/server boundary**: fixed-format remote tasks execute through
  transport-backed routes behind sealed capability tokens; arbitrary closure
  remoting is intentionally rejected.
- **CPU topology**: Themis topology detection supplies default worker
  counts. Workers are not bound to processors, so the scheduler reports no
  per-worker NUMA assignment; its same-node steal tier activates only for an
  assignment a caller can vouch for (ADR-037).
- **Local CPU layer**: `ThreadScheduler` owns worker queues, work-class routing,
  scoped batches, and indexed fan-out/reduction.
- **Route layer**: `HybridRouter<P>` selects thread/process/server/accelerator
  routes through sealed zero-sized policies.
- **Transport layer**: `moirai-transport` consumes route metadata, archives
  payload bytes, and executes admitted fixed-format process/server tasks.
- **Accelerator layer**: `moirai-gpu` plans topology-aware launch shapes and
  schedules typed tasks through Hephaestus's generic `ComputeDevice` seam.
  `WgpuContext` and `CudaContext` select complete Hephaestus providers; provider
  acquisition failures remain typed failures rather than CPU fallbacks.
- **Memory boundary**: archive payloads move as owned bytes across
  thread/process/server/device regions; cross-process and cross-device pointer
  transfer is rejected.

## Python bindings

`moirai-python` exposes PyO3 wrappers over `moirai::Moirai`; it implements no
separate scheduler, planner, or backend, and it is not published to crates.io.
GitHub Releases tagged `moirai-python-v<version>` attach validated CPython
3.10–3.13 wheels for Linux, Windows, and macOS and publish the same artifacts to
PyPI through OIDC.

```bash
py -3.13 -m pip install moirai-python

# Source checkout verification
py -3.13 -m pip install -e moirai-python
py -3.13 -m unittest discover moirai-python\tests
```

## Examples

Examples are registered on the facade package, so they run with `-p moirai-runtime`:

```bash
cargo run -p moirai-runtime --example basic_usage
cargo run -p moirai-runtime --example web_crawler_parallel
cargo run -p moirai-runtime --example realtime_chat_server
cargo run -p moirai-runtime --example rayon_parallel_patterns
```

- [Basic usage](moirai/examples/basic_usage.rs), [blocking channels](moirai/examples/blocking_channels.rs),
  [sync primitives](moirai/examples/sync_primitives.rs), [iterator showcase](moirai/examples/iterator_showcase.rs)
- [Web crawler](moirai/examples/web_crawler_parallel.rs), [video processing pipeline](moirai/examples/video_processing_pipeline.rs)
- [Financial transaction processing](moirai/examples/financial_transaction_processing.rs),
  [high-frequency data pipeline](moirai/examples/high_frequency_data_pipeline.rs)
- [Chat server](moirai/examples/realtime_chat_server.rs), [load balancing](moirai/examples/network_service_load_balancing.rs),
  [IoT device management](moirai/examples/iot_device_management.rs)
- [Rayon-style patterns](moirai/examples/rayon_parallel_patterns.rs),
  [Tokio-style patterns](moirai/examples/tokio_task_fanout.rs),
  [Moirai vs Tokio/Rayon](moirai/examples/moirai_vs_tokio_rayon_comparison.rs)

Some examples require non-default features (`gpu_acceleration` needs `gpu`;
`async_timer` and `tokio_task_fanout` need `async`).

## Performance evidence

Performance claims are limited to executable Criterion targets and
value-semantic tests. The active evidence surfaces are:

- **Thread scheduling**: `thread_schedule_comparison`, `industry_comparison`,
  and `public_result_handle_comparison` compare Moirai scoped work, indexed
  reduction, mixed workloads, and public result handles against Rayon/Tokio
  reference rows.
- **Iterator paths**: `parallel_iterator_regression`,
  `iterator_adapter_comparison`, `iter_ops_parallel_comparison`,
  `cache_iterator_comparison`, and `async_iterator_comparison` provide same-run
  Rayon/Tokio comparisons with checksum/value assertions before timing.
- **Collective operations**: `collective_ops_comparison` measures
  scatter/gather/traverse over the `ChunkedVec` layout.
- **Process/server routing**: `process_server_scheduler_routing` validates
  deterministic route summaries; `process_server_routed_execution` executes
  fixed-format `SumU64` requests through real server and supervised process
  routes.
- **Async I/O**: `async_fs_*`, `async_tcp_*`, `async_udp_comparison`, and
  `async_io_compat_comparison` compare Moirai-owned facade behavior against
  Tokio references where the semantics match.

TPU and NPU placement are not claimed as implemented scheduler execution. GPU
evidence includes the `moirai-gpu::occupancy` launch-shape planner, its
value-semantic tests, and the generic Hephaestus task/context adapter tests.

## Testing

```bash
cargo nextest run --workspace --all-features
cargo test --doc --workspace --all-features
cargo bench -p moirai-benchmarks --no-run
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full gate sequence.

## Safety

- The facade crate (`moirai-runtime`) contains no `unsafe` blocks and denies
  `missing_docs` and `unsafe_op_in_unsafe_fn`.
- Lower-level crates use `unsafe` for lock-free data structures, platform I/O,
  and FFI, isolated behind safe APIs. `// SAFETY:` comments are the required
  convention for new and touched code; coverage of existing blocks is partial
  and is being raised (see [CONTRIBUTING.md](CONTRIBUTING.md)).
- Concurrency correctness is covered by exactly-once stress tests and bounded
  `loom` interleaving models for the Chase-Lev steal/pop, MPMC waiter, SPSC
  ring publication, and async executor wake-deduplication protocols.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

## Acknowledgments

- [Rayon](https://github.com/rayon-rs/rayon) — parallel computing patterns
- [Tokio](https://github.com/tokio-rs/tokio) — async runtime design
- [Go](https://golang.org/) — coroutines and channels
- [OpenMP](https://www.openmp.org/) — parallel patterns
