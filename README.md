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

Workspace 0.9.3 aligns its private allocator edge with Mnemosyne-memory 0.10.
The public GPU budget remains on mnemosyne-memory-core 0.3; scheduler, bridge,
feature and Rust API behavior stay unchanged. The lock records the reviewed
integrated Mnemosyne 0.10 provider; downstream publication remains ordered by
the dependency owner.

## Browser I/O

Workspace 0.9.4 extends the default-off private Windows diagnostic with bounded
controller focus and accelerator observations. Synchronous input callbacks
retain values and return before tracing; they never change Handled, focus,
input or deadlines. Installed-runtime and Metis form acceptance remain separate
checks under the [diagnostic contract](moirai-pal/src/windows/webview/host/diagnostics/design.md).
The same patch restores synchronous native-parent focus entry through
MoveFocus(PROGRAMMATIC), with weak callback ownership and typed native errors.
The unchanged-host Windows regression leaves focus at the parent and fails;
[corrected provider Windows execution](https://github.com/ryancinsight/Moirai/actions/runs/38749728275)
passes the real parent-entry regression and all nine installed fixtures.
Full Metis form acceptance remains open: run 38750658426 still retains the
second dose after a Lost/Got/Lost controller transition. A real MTA UIA fixture
isolates field delivery; it does not replace the complete Metis actor/journey.
Its controlled ordinary baseline completes 16 replacements/64 preservation
checks while selected mode fails. The private-parent experiment also fails
selected delivery and is withdrawn. The next native comparison isolates actual
controller callbacks from startup/CDP observers. Four-mode Windows execution
passes ordinary and controller-only, while startup-only and full selected
fail before a complete replacement journey. Two more controls distinguish
native events and CDP activation from DOM sampling; full selected and ordinary
acceptance remain required, with exact Unicode and every deadline unchanged.
The document-binding reuse experiment also fails and is withdrawn. Two more
controls distinguish DOM domain activation and one complete snapshot from
continuous sampling. These controls retain the whole native field journey;
full observation and both Metis form journeys remain required.
The domain/snapshot controls pass. Caller-owned wait coalescing in source
9cf4d47a bounds recurring work, but Windows run 38779895376 still fails the
last startup-only dose (15 edits/60 preservation checks). Full selected mode
completes 16/64; its observer exhausts its quota during the run. That does not prove
observation stays active through the journey. The root cause remains open.
An opt-in, fixture-only one-shot DOMSnapshot now separates live input values
from UIA reports without injected JavaScript. Its separate diagnostic run
preserves every original gate and deadline. Run38785314523 passes all eight
original modes (128 edits/512 preservation checks), but the separate probe
fails concentration4 after6/24. UIA and a timely live snapshot both report2;
the renderer responds while the original sampler remains active. This point
rules out stale UIA alone, without identifying the delivery cause. The next
opt-in variant reuses canonical controller callbacks with their own quota.
Source aa532b88/run38788877397 passes45 policy cases and seven original modes,
but startup fails concentration0 after10/40 (UIA4). Its separate16/64 pass
never triggers a snapshot;254 retained controller callbacks exhaust their own
quota with7 omissions. These different invocations do not establish a cause.
The private correction checks the original wait deadline and ready public
events at outgoing SDK admission, retaining an unissued stage for a later
caller turn. Baseline source44184f40/run38795557575 reproduces outgoing DOM
work before a ready resize returns. Source2fcef45d/run38798617700 passes the
corrected native regression in6.837s, all50 policy cases, and the native runtime
query. Ordinary mode still fails the last dose, startup fails the first weight,
and the separate probe retains the first patient in both UIA and timely live DOM
after requesting the second. Its complete controller history ends in LostFocus;
failed input targets a different owned HWND from prior key callbacks, on the
same browser process/thread/layout. The fixture now enters the known native
parent once after UIA selection and checks preservation of that same field.
[Windows run 38803044085](https://github.com/ryancinsight/Moirai/actions/runs/38803044085)
on source `6343610d` records 69 PASS test executions. Each of the eight original
modes and the separate probe passes 16 replacements/64 preservation checks;
its collected schema3 completion also verifies 16 entries/12 non-first entries.
The log separately retains 16 entered and 16 same-field-focused observations
per invocation. SDK admission and runtime availability pass, as does the actual
Rust bridge in 7.232s. The separate snapshot probe never triggers; UIA and
diagnostic quotas leave partial histories, so this supplies no new live-DOM
or complete-sampler proof. Both full Metis journeys remain open under the
[entry contract](moirai-pal/src/windows/webview/host/focus/tests/client/design.md).

A separate default-off [late-input diagnostic](moirai-pal/src/webview_startup/late_input/design.md)
retains copied controller snapshots after an actual configured key-down,
including later layout-dependent keys outside the exhausted startup quota.
Portable and strict MSVC checks pass; installed execution and both complete
Metis journeys remain required. No additional input or callback work supplies
this evidence.

Workspace 0.9.2 corrects native WebView2 method admission after an installed
Rust consumer rejects the host object through a DOM `EventTarget` cast. The
private interface checks real callable methods and preserves their original
receiver and listener teardown ownership. Public Rust signatures and message
bounds remain unchanged. Fourteen actual Chrome contracts and strict WASM and
Windows compilation pass. The [installed Windows regression](https://github.com/ryancinsight/Moirai/actions/runs/38539086879/job/115769121797)
passes real bidirectional values, inclusive byte bounds, close/remount and fresh
profile cleanup. Metis adoption and complete consumer journeys remain gates
under the [bridge contract](moirai-pal/src/wasm/webview/design.md).

Workspace 0.9.1 adds private fixture-only WebView2 startup observations with
bounded resource, exception and selected DOM evidence. Both explicit private
selection and a dedicated diagnostic tracing filter are required. Ordinary
applications do not capture DOM content. The
[source contract](moirai-pal/src/windows/webview/host/diagnostics/design.md)
defines activation, callback ownership and trace quotas. The protected provider
run verifies selected native DOM/shutdown; the subsequent consumer trace
identifies the native bridge admission failure.

Workspace 0.9.0 adds the Rust-owned page-side WebView2 bridge, real browser
permission requests, explicit callback teardown and native Number formatting.
Message admission shares the native 64 KiB limits; permission guards remain
bounded across remount and media leases stop every acquired track on drop.
An opt-in empty WebView2 profile supplies per-request denial evidence and an
actual browser-exit wait before caller cleanup. `ProcessSpec::working_directory`
sets the supervised child's directory without changing the parent's directory.
[ADR 0077](docs/adr/0077-rust-page-provider-ownership.md) and its source-adjacent
design define ownership, failure reporting and pending runtime acceptance.
Real Windows page handoff and consented positive permission cleanup remain open.

Workspace 0.8.3 fixes the inclusive WebView2 capture byte boundary and wires
its production PNG validator. Workspace 0.8.0 adds `moirai_pal::wasm::WebHttp`, a bounded GET/POST Fetch
provider. Declared and streamed response sizes are checked independently;
future drop aborts the request, and one finite deadline covers headers and body.
Document listeners unregister through owned handles. `WebLocation` and DOM
observations supply mechanisms while Metis retains application policy.
[ADR 0069](docs/adr/0069-bounded-browser-http.md) specifies the contracts.

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
`melinoe`. Optional: `distributed`, `network`, `metrics`, `gpu`,
`encryption`, `compression`, `tokio-compat`, `no-std`, and `full`.

When an application calls Melinoe's partition functions directly, initialize
the shared provider before the first call:

```rust
moirai::initialize();
```

Moirai's own partition helpers perform this initialization automatically.

Minimum supported Rust version: **1.95**. The pinned build toolchain is 1.97.0
(`rust-toolchain.toml`).

`moirai-webview` 0.2.0 supplies Rust-owned AppKit application, Edit and Window
menus and serves packaged Apple resources with Fetch-compatible
status and MIME/length headers. AppKit and UIKit use the same response law;
the native Metis starter runs its Rust/WASM greeting through this boundary.
Host-owned `MoiraiExternalLinks` bundle metadata admits exact HTTPS popup
destinations for an OS browser handoff. Undeclared external navigation is denied;
the packaged main frame remains intact (ADR 0074).

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
| [`moirai-gpu`](https://docs.rs/moirai-gpu) | GPU launch-shape planning over themis topology and mnemosyne budgets |
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
  counts. Workers run unbound by default and the scheduler then reports no
  per-worker NUMA assignment. With `WorkerPlacement::Pinned`, each worker binds
  itself to one logical processor at startup, construction fails if any
  binding is refused, and the same-node steal tier reads the enforced
  assignment (ADR-037).
- **Local CPU layer**: `ThreadScheduler` owns worker queues, work-class routing,
  scoped batches, and indexed fan-out/reduction.
- **Route layer**: `HybridRouter<P>` selects thread/process/server/accelerator
  routes through sealed zero-sized policies.
- **Transport layer**: `moirai-transport` consumes route metadata, archives
  payload bytes, and executes admitted fixed-format process/server tasks.
- **Accelerator layer**: `moirai-gpu::occupancy` plans topology-aware launch
  shapes. Device acquisition and kernel execution belong to Hephaestus, which
  depends on Moirai; a device operation runs on Moirai as an ordinary task
  (ADR 0041). TPU/NPU providers and device-specific kernel evidence remain
  open.
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

GPU launch-shape planning is covered by the occupancy tests. Device execution
evidence is Hephaestus-owned; TPU/NPU placement and device-specific rendering
remain open.

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
