# Moirai Development Checklist

**Target**: Unreleased

## MOI-PAR-062 — borrowing parallel scope [minor] — in progress

- Owner: Codex `/root` (composed from preserved peer work).
- Scope: the `moirai-parallel` borrowing scope facade, its direct dependency
  ownership, value-semantic tests, and release documentation.
- Acceptance: multiple tasks borrow caller-owned values, complete before the
  scope returns, preserve a body return value, and compile without exposing an
  escaping scheduler lifetime.
- [x] Preserve and complete the peer's public scope facade.
- [x] Add direct `moirai-core` ownership and the higher-ranked lifetime bound.
- [x] Add borrowed completion and return-value coverage.
- [x] Pass focused local and exact-head hosted gates; merge Moirai.

## MOI-SCHED-061 — bounded indexed admission [patch] — in progress

- Owner: Codex `/root` (stale-peer takeover after one hour without a write or
  commit in the claimed scope).
- Scope: indexed scheduler admission, its diagnostics and value-semantic
  saturation tests, release documentation, and the downstream Kwavers
  serialization workaround. Other scheduler policies are non-goals.
- Acceptance: a full worker admission queue executes each rejected indexed
  chunk exactly once on the caller, map-reduce preserves the mathematical
  result, caller-run panics become `SpawnFailed(Panicked)` only after scheduled
  scope work drains, the scheduler remains reusable, and the recovery event is
  observable without allocating on the healthy path.
- [x] Preserve the stale peer's caller-runs intent.
- [x] Add one shared panic boundary for inline indexed work.
- [x] Add a relaxed monotonic admission diagnostic.
- [x] Add deterministic saturated fan-out, reduction, panic, and reuse coverage.
- [x] Pass focused local and exact-head hosted gates; merge Moirai.
- [ ] Advance Kwavers to the merged Moirai pin, remove the test-serialization
      workaround, pass the affected therapy lane, and merge Kwavers.

## MOI-REL-060 — Python wheel releases [patch] — blocked

- Owner: Codex `/root`.
- Scope: `moirai-python` distribution metadata and documentation, a pinned
  cross-platform release workflow, the protected GitHub publishing
  environment, release-facing root documentation, the Linux shared-memory
  size boundary that blocks the binding gate, and this owner-keyed entry.
  Other native runtime behavior and workspace crates are non-goals.
- Acceptance: a GitHub Release tagged `moirai-python-v<version>` builds locked
  wheels for supported CPython versions on Linux, Windows, and macOS; installs
  and imports each wheel; validates distribution metadata against the tag;
  attests and attaches the exact artifacts to the GitHub Release; and publishes
  those same wheels to PyPI through GitHub OIDC.
- [x] Make Cargo the Python distribution version source of truth.
- [x] Add pinned cross-platform wheel CI and release workflows.
- [x] Synchronize Python, root, changelog, toolchain, and Nextest contracts.
- [x] Build, install, import, and exercise a production wheel locally.
- [x] Pass workflow lint and focused Rust/Python binding gates.
- [x] Protect the `pypi` environment with the `moirai-python-v*` tag policy.
- [x] Pass exact-head hosted CI.
- [x] Merge the release PR.
- [ ] Register the PyPI pending trusted publisher after account verification.
- Blocker: PyPI rejects `ryanclanton@outlook.com`; registration reopens when
  the account has a PyPI-accepted email address and completes verification.
- Evidence: checksum-verified actionlint 1.7.12 accepts both workflows; locked
  Cargo metadata and Rust 1.95 formatting, warning-denied all-target Clippy,
  configured Nextest 1/1, doctests, and warning-clean rustdoc pass. A locked
  CPython 3.13 wheel builds as `moirai-python` 0.4.0, installs into an isolated
  environment, imports, reports the requested two-worker native lifecycle, and
  passes both Python tests. GitHub environment `pypi` accepts only
  `moirai-python-v*` tags. Hosted run `29799529159` then exposed an unchecked
  Unix `usize`-to-`off_t` conversion in `moirai-core`; the owner-local fix
  validates zero and out-of-domain lengths before acquiring a shared-memory
  descriptor and covers both boundaries through the public `SharedMemory`
  contract. The Windows host passes warning-denied all-target core Clippy and
  70/70 configured Nextest cases. Replacement hosted run `29800011266` passes
  the Windows wheel job and exposes a pre-existing unconditional non-Linux
  `AtomicBool` import in the Linux binding closure; that import is now
  target-gated. Exact-head hosted run `29800253930` passes formatting,
  warning-denied binding lint, native binding and Unix shared-memory boundary
  tests, binding doctests, and all three production wheel build/install/import
  smoke jobs. PR #82 carries the merge-ready delivery. PyPI publisher
  registration remains blocked on account verification.

## MOI-SCOPE-059 — scoped multi-job memory safety [patch] — done

- Owner: Codex Tyche integration.
- Scope: `moirai-executor/src/schedule/{job,runtime/scheduler}` and focused
  scoped-dispatch regression coverage.
- [x] Reproduce Tyche's multi-job borrowed-slice access violation without
      Tyche.
- [x] Publish the final zero count while holding the wait lock, and require
      both caller and worker waiters to acquire that lock before destroying the
      stack-owned scope state.
- [x] Verify the 64-round borrowed-chunk regression, the bounded one-completion
      Loom model, and the complete executor package.
- Evidence: configured Nextest passes `moirai-executor` 88/88 with one
  cfg-gated skip; warning-denied all-target/all-feature Clippy, rustfmt,
  doctests, and rustdoc pass. The bounded Loom model passes 1/1. Miri cannot
  reach the regression on Windows because Themis NUMA detection calls the
  unsupported `GetNumaHighestNodeNumber` FFI; it reports no result for this
  invariant. PR 81 contains the delivery; Moirai has no repository engineering
  workflow beyond the non-gating Copilot workflow.

## Moirai 0.4.0 release artifact closure [patch]

- [x] Take over the stale release-artifact increment after one hour without
      file or commit activity.
- [x] Synchronize the workspace version, changelog release section, checklist
      target, and benchmark artifact contract at 0.4.0.
- [x] Verify the focused artifact contract, formatting, and warning-denied
      benchmark test target before publication.
- Evidence: configured Nextest passes the artifact contract 1/1 with 67
  unrelated cases filtered; nightly rustfmt and warning-denied Clippy for the
  benchmark contract target pass.

## MOI-ASYNC-058 — synchronization stabilization [patch]

- [x] Take over the stale uncommitted synchronization/codec lane on `main`.
  Scope: `moirai-async/src/sync/{broadcast,mpsc,wait_queue}.rs`, timer
  regression coverage, `moirai-http/src/codec.rs`, affected examples, and this
  provider PM scope.
- [x] Verify FIFO/cancellation behavior, broadcast retention, and the
  cancellation-compaction regression through value-semantic configured Nextest
  coverage; run warning-denied Clippy and formatter checks before publication.
- [x] Absorb audit findings into the provider backlog/gap register and delete
  the untracked report instead of retaining a parallel status artifact.
- Evidence: configured Nextest passes `moirai-async` 88/88 and `moirai-http`
  9/9; warning-denied workspace Clippy, rustfmt, rustdoc, and doctests pass.

## Provider default-source convergence [major]

- [x] Remove direct Themis, Mnemosyne, and Melinoe revisions plus the local
  Melinoe patch from the workspace dependency SSOT.
- [x] Record ADR 016: merged Mnemosyne 0.5/Core 0.2 requires Rust 1.95, so the
  workspace advances from 0.3.1 to 0.4.0 without a compatibility branch.
- [x] Refresh the lockfile against merged provider heads and prove one source
  identity for Melinoe, Themis, and Mnemosyne.
- [x] Verify the focused GPU consumer with Rust 1.95 compilation,
  warning-denied Clippy, configured Nextest, doctests, and rustdoc.
- [x] [patch] Preserve SPSC send-before-close ordering with a value-semantic
  drain regression that contains no wall-clock synchronization.
- Evidence: Rust 1.95 accepts `moirai-gpu` and Rust 1.94 rejects the declared
  package graph; warning-denied focused Clippy passes; Nextest passes 10/10;
  doctests pass 0/0; rustdoc is warning-clean; each provider resolves to one
  lock source identity; the SemVer major comparison reports no API check
  failures.

## Phase 30: NUMA helper removal and channel hierarchy closure

- [x] [major] Delete the unconsumed `moirai_iter::numa` API and its obsolete
  Rayon comparison benchmark. Themis owns placement, Mnemosyne owns allocation,
  and `moirai-parallel` owns scheduler-backed data-parallel work; no source
  consumer imports the removed iterator helper.
- [x] [patch] Split hybrid and MPMC channel implementation into vertical state,
  send, receive, future, and test modules. `HybridChannel<T>` is a zero-sized
  factory; endpoint halves own each live synchronization primitive.
- Evidence: `cargo nextest run -p moirai-core` 69/69;
  `-p moirai-iter` 185/185 (2 cfg-skips); `-p moirai-benchmarks` 68/68;
  warning-denied `moirai-core` Clippy; and formatter checks pass. This is
  compile-time/API-surface and value-semantic test evidence, not a throughput
  claim.

## Phase 29: Indexed caller-region flattening

- [x] [patch] Mark the caller lane while it participates in indexed fan-out
  and map/reduce so nested regions flatten on every outer lane, not only
  scheduler workers.
- [x] Verify exact caller/worker lane identity and value sums through nextest,
  then run warning-denied Clippy.
- [x] Pin the commit in RITK and pass the unchanged masked-CMA consumer gate.
- Evidence: `cargo nextest run -p moirai-executor` passed 83/83 with one
  cfg-gated skip; warning-denied all-target/all-feature Clippy passed. RITK
  pins merged Moirai main in its Atlas checkout action, and `cargo nextest run
  -p ritk-registration masked_cache --all-features --status-level fail` passes.

## Phase 28: Melinoe executor capability migration

- [x] [major] Construct Melinoe's validated `ParallelExecutor` next to the real
  Moirai scheduler bridge with an explicit exact-once, completion, and lifetime
  safety proof.
- [x] [major] Remove the raw function-pointer registration call and update the
  workspace Melinoe contract to 0.9.0 and Mnemosyne facade to 0.3.0.
- [x] Verify the real Melinoe routing path plus Moirai executor Clippy, 83/83
  nextest (one cfg-gated test skipped), doctests, and rustdoc against Melinoe
  `bb07447`, Themis `6140468`, and Mnemosyne 0.3.0 at `df2994f`.

## Phase 27: GPU pollster boundary removal
- [x] [patch] Added `moirai_executor::block_on` as the Moirai-owned
  current-thread parking wait primitive for synchronous async boundaries.
- [x] [patch] Replaced `moirai-gpu`'s `GpuTaskAdapter` `pollster::block_on`
  call with `moirai_executor::block_on` and removed `pollster` from the
  `wgpu-backend` feature dependency list.
- [x] [patch] Added a benchmark source contract that rejects reintroducing
  `pollster` into `moirai-gpu`'s manifest or task adapter.
- Evidence: `rustup run nightly rustfmt --edition 2021
  moirai-executor\src\lib.rs moirai-gpu\src\task.rs
  benchmarks\tests\benchmark_contracts\source_contracts.rs`;
  `rustup run nightly cargo check -p moirai-gpu --features wgpu-backend`;
  `rustup run nightly cargo check -p moirai-executor --no-default-features`;
  `rustup run nightly cargo tree -p moirai-gpu --features wgpu-backend -i
  pollster` reports no matching package; `rustup run nightly cargo nextest run
  -p moirai-benchmarks gpu_task_adapter_uses_moirai_block_on_not_pollster
  --status-level fail --no-fail-fast` passed 1/1.

## Phase 26: Socket stale-wake regression ✅
- [x] [patch] Added a real loopback TCP regression for the July 2 stale-wake
  async bug: `timeout(stream.read(...))` completes through the timer while the
  socket read waker remains registered, then peer readability wakes the stale
  reactor slot after the async task has completed.
- Evidence: `rustup run nightly cargo nextest run -p moirai-async
  timeout_read_stale_socket_wake_does_not_repoll_completed_task
  --status-level fail --no-fail-fast`.

## Phase 25: Transport stale export cleanup
- [x] [patch] Removed `moirai_transport::core_zero_copy`, a stale re-export of
  deleted `moirai_core::communication::zero_copy`, so Atlas consumers compile
  against the current `moirai_core::communication` surface.
- Evidence: `rustup run nightly cargo fmt -p moirai-transport --check`;
  `rustup run nightly cargo check -p moirai-transport`.

## Phase 24: Stateful Chunk Parallel Provider API ✅
- [x] [patch] Added `moirai_parallel::for_each_chunk_mut_with_state` so
  consumers can run mutable chunk kernels with one reusable scratch state per
  scheduled worker shard.
- [x] [patch] Re-exported the API from `moirai-parallel` and covered it with a
  value-semantic test that proves every chunk is written from reusable state.
- [x] [patch] Added `for_each_chunk_triple_mut_enumerated_with` and
  `for_each_chunk_quad_mut_enumerated_with` for provider-owned fused updates
  across three or four equal-length mutable output buffers.
- [x] [patch] Re-exported the multi-output chunk APIs and covered them with
  value-semantic tests that prove chunk indices and all output buffers are
  written from the caller-provided closure.
- Evidence: `cargo fmt -p moirai-parallel --check`; `cargo check -p
  moirai-parallel`; `cargo nextest run -p moirai-parallel
  for_each_chunk_ --status-level fail --no-fail-fast` (6/6).

## Phase 23: Task Registry Stable Slot Access ✅
- [x] [patch] Kept `TaskStateBlock` slot storage private behind
  `UnsafeCell`-based `get`/`insert`/`clear`/`states` methods so lifecycle
  tokens receive stable `NonNull<TaskState>` pointers without exposing block
  internals.
- [x] [patch] Routed registry production paths, diagnostics, active/completed
  counts, and cleanup through the block accessor API.
- [x] [patch] Updated benchmark source-contract assertions to pin the dense
  `UnsafeCell<Option<TaskState>>` representation and zero-allocation stable-slot
  invariant.
- Evidence: `rustup run nightly cargo fmt -p moirai-executor --check`; `rustup
  run nightly cargo check -p moirai-executor --all-targets`; `rustup run
  nightly cargo clippy -p moirai-executor --all-targets -- -D warnings`;
  `rustup run nightly cargo nextest run -p moirai-executor` (61/61, 1 skipped);
  `rustup run nightly cargo doc -p moirai-executor --no-deps`.

## Phase 22: Sharded Task Registry ✅
- [x] [patch] Replaced the hybrid executor's single `Arc<Mutex<TaskRegistry>>`
  with `Arc<ShardedTaskRegistry>` so task registration and metadata reads route
  through per-shard registry locks.
- [x] [patch] Added sharded registry coverage proving dense global IDs,
  global-to-local metadata reporting, lifecycle-token completion, and unknown
  task lookup behavior.
- [x] [patch] Updated manager status/stat/wait paths to use the sharded
  registry facade directly and removed stale warning sources.
- Evidence: `rustup run nightly cargo fmt -p moirai-executor --check`; `rustup
  run nightly cargo check -p moirai-executor --all-targets`; `rustup run
  nightly cargo clippy -p moirai-executor --all-targets -- -D warnings`;
  `rustup run nightly cargo nextest run -p moirai-executor` (62/62, 1 skipped);
  `rustup run nightly cargo doc -p moirai-executor --no-deps`; `git diff
  --check`.

## Phase 21: Executor Lockfile and Rustdoc Hygiene ✅
- [x] [patch] Synchronized `Cargo.lock` with the existing `cfg(loom)`
  `moirai-executor` dev-dependency so locked builds resolve the model-checking
  dependency edge.
- [x] [patch] Removed redundant explicit `WorkScheduler` Rustdoc link targets
  from `moirai-executor::hybrid`, keeping the package rustdoc gate
  warning-clean.
- Evidence: `rustup run nightly cargo fmt -p moirai-executor --check`;
  `rustup run nightly cargo check -p moirai-executor --all-targets`; `rustup
  run nightly cargo clippy -p moirai-executor --all-targets -- -D warnings`;
  `rustup run nightly cargo nextest run -p moirai-executor`; `rustup run
  nightly cargo test --doc -p moirai-executor`; `rustup run nightly cargo doc
  -p moirai-executor --no-deps`; `git diff --check`.

## Phase 20: Async RwLock Waiter Map ✅
- [x] [patch] Completed the `moirai-async::sync::RwLock` waiter storage
  migration from `VecDeque` tuples to keyed `BTreeMap<u64, RwWaiter>` state.
- [x] [patch] Routed read/write future poll, cancellation, writer grant, and
  reader-batch wakeup through the same waiter state so the O(log n) removal
  contract compiles and preserves FIFO-by-monotonic-id handoff.
- [x] [patch] Fixed the `ConnectionId` Rustdoc peer-address link to
  `std::net::SocketAddr`, keeping rustdoc warning-clean.
- Evidence: `rustup run nightly cargo fmt -p moirai-async --check`; `rustup run
  nightly cargo check -p moirai-async --all-targets`; `rustup run nightly cargo
  clippy -p moirai-async --all-targets --all-features -- -D warnings`; `rustup
  run nightly cargo nextest run -p moirai-async`; `rustup run nightly cargo test
  --doc -p moirai-async`; `rustup run nightly cargo doc -p moirai-async
  --all-features --no-deps`; `git diff --check`.

## Phase 19: Concurrent Stream Module Export ✅
- [x] [minor] Completed the `parallel_stream` -> `stream` module rename by
  exporting `moirai_iter::stream` from `moirai-iter`.
- [x] [minor] Renamed the stream extension trait and methods to
  `ConcurrentStreamExt` / `concurrent_*`, matching the bounded-concurrency
  contract rather than promising CPU parallelism for every async item future.
- [x] [minor] Added fused `concurrent_filter_map` and `concurrent_filter`
  stream adapters with value-semantic coverage.
- Evidence: `cargo fmt --check -p moirai-iter`; `cargo clippy -p moirai-iter
  --all-targets --all-features -- -D warnings`; `cargo nextest run -p
  moirai-iter stream` -> 10 passed; `cargo doc -p moirai-iter --all-features
  --no-deps`.

## Phase 18: Default Provider Feature Contract
- [x] [patch] Added default `parallel` and `mnemosyne-memory` features to every
  Moirai package. Existing Mnemosyne-backed crates forward `mnemosyne-memory`
  to the established `mnemosyne` provider feature; non-provider leaf crates use
  zero-dependency markers.
- [x] [patch] Applied rustfmt-required import/closure formatting in existing
  Moirai iterator/reactor files so the formatting gate is clean.
- Evidence: `cargo metadata --no-deps --locked --format-version 1`; full Atlas
  feature-policy metadata audit; `cargo fmt --check`; `git diff --check`.
  Residual: compile/test gates were blocked before rustc by denied access to
  `target/debug/.cargo-lock`.

## Phase 17: Mnemosyne Worker Maintenance Integration ✅
- [x] Registered Moirai's global scheduler as Melinoe's `std` partition executor via pushed Melinoe commit `8140882`, so branded partition writes route through Moirai workers.
- [x] Added a value-semantic scheduler test proving Melinoe partition routing writes every branded cell exactly once through the registered Moirai executor.
- [x] Removed dead thread-local cache declaration from `moirai-core::pool::GlobalPool::get`; the active implementation uses the global pool path.
- [x] Added `mnemosyne` as an optional `moirai-executor` dependency and default feature.
- [x] Forwarded the top-level `moirai/mnemosyne` feature into `moirai-executor/mnemosyne`.
- [x] Routed idle worker-loop maintenance through `mnemosyne::Mnemosyne` defragmentation sweeps using the provider's top-level backend selector.
- [x] Updated Moirai's Mnemosyne pin to `938d0c2bc094d3bbe7745d68d60e05a531e0cfc2` so the executor consumes the exported provider selector.
- [x] Verification: `cargo fmt --check`; `cargo check -p moirai --locked`; `cargo test -p moirai-executor --features mnemosyne --locked`; `cargo clippy -p moirai-executor -p moirai --all-targets --all-features --locked -- -D warnings`.
- Evidence: compiler diagnostics, value-semantic scheduler/executor tests under the Mnemosyne feature, and clippy diagnostics.

## Phase 16: Default Parallel Branding Integration ✅
- [x] Enabled `moirai-parallel` Mellinoe integration by default so the parallel crate exposes branded partitioning without opt-in feature plumbing.
- [x] Added `melinoe` to the `moirai` facade default feature set alongside existing `parallel` and `mnemosyne` defaults.
- [x] Replaced serial async iterator map/filter/for_each execution with bounded concurrent polling while preserving ordered map/filter results.
- [x] Verification: `cargo fmt --check`; `cargo test -p moirai-iter execution::tests::async_context --locked`; `cargo test -p moirai-parallel -p moirai --locked`; `cargo clippy -p moirai-iter -p moirai-parallel -p moirai --all-targets -- -D warnings`; `cargo test --locked --workspace --examples`.
- Evidence: value-semantic async ordering tests, Mellinoe partitioning tests under default features, clippy diagnostics, and workspace example execution.

## Phase 15: Code Quality & Design Principles Enforcement ✅
- [x] **MAJOR**: Fixed clippy errors (match_same_arms, manual_let_else) for clean builds
- [x] **MAJOR**: Implemented underscored parameters (priority/locality hints in HybridExecutor)
- [x] **MAJOR**: Extracted magic numbers to named constants (SSOT/SOC compliance)
- [x] Applied cargo fix and cargo fmt for consistent code style
- [x] Fixed mixed attribute styles and redundant code patterns
- [x] Ensured no prohibited naming patterns (*_old, *_new, *_enhanced, etc.)
- [x] Verified no deprecated/redundant components requiring removal
- [x] Applied design principles (SOLID, CUPID, GRASP, DRY, KISS, YAGNI)
- [x] Maintained single implementations with flexible configuration
- [x] Enforced zero-cost abstractions and stdlib iterator usage

## Phase 14: Critical Infrastructure Fixes ✅
- [x] **MAJOR**: Fixed HybridExecutor to actually execute tasks (auto-start workers)
- [x] **MAJOR**: Fixed spawn_blocking result communication via proper channels
- [x] **MAJOR**: Fixed spawn_async implementation with polling-based runtime
- [x] Fixed clippy warnings that prevented clean builds (-D warnings compliance)
- [x] Fixed method naming conflicts (XorShiftRng API)
- [x] Replaced cfg(disabled) with proper Cargo feature flags
- [x] Fixed documentation compilation errors and broken links
- [x] Verified all examples work end-to-end (basic_usage, async_timer)

## Phase 13: Code Optimization and Cleanup ✅
- [x] Review and clean codebase following design principles
- [x] Consolidate channel implementations (DRY/SSOT)
- [x] Extract common iterator patterns into base module
- [x] Simplify sync module - remove redundant wrappers
- [x] Implement ExecutionBase trait for all contexts
- [x] Fix all build errors across workspace
- [x] Apply SOLID, CUPID, GRASP, DRY, KISS, YAGNI principles
- [x] Update README with optimization details

## Phase 12: Iterator System Enhancements ✅
- [x] Advanced iterator combinators (chunks, windows, etc.)
- [x] SIMD-optimized iterators
- [x] Cache-optimized iteration patterns
- [x] Streaming and batching support
- [x] Channel fusion for zero-copy pipelines
- [x] Adaptive execution strategies
- [x] Prefetching and memory optimization
- [x] NUMA-aware iteration

## Phase 11: Zero-Copy Transport ✅
- [x] Memory-mapped ring buffers
- [x] Zero-copy channel implementation
- [x] Shared memory transport
- [x] RDMA-style operations
- [x] Efficient serialization
- [x] Adaptive batching
- [x] Flow control mechanisms

## Phase 10: Unified Transport Layer ✅
- [x] Transport trait abstraction
- [x] In-memory transport
- [x] IPC transport foundation
- [x] Network transport skeleton
- [x] Message routing
- [x] Connection management
- [x] Transport selection logic

## Phase 9: Advanced Scheduler ✅
- [x] NUMA-aware scheduler
- [x] CPU topology detection
- [x] Work migration policies
- [x] Adaptive load balancing
- [x] Priority scheduling
- [x] Deadline scheduling
- [x] Resource quotas

## Phase 8: Metrics System ✅
- [x] Core metrics collection
- [x] Task execution metrics
- [x] Scheduler performance metrics
- [x] Memory usage tracking
- [x] Latency histograms
- [x] Throughput monitoring
- [x] Metric aggregation

## Phase 7: Async Runtime ✅
- [x] Async executor implementation
- [x] Future polling mechanism
- [x] Async task spawning
- [x] Timer implementation
- [x] I/O reactor integration
- [x] Async synchronization primitives

## Phase 6: Synchronization Primitives ✅
- [x] Fast mutex implementation
- [x] Reader-writer locks
- [x] Condition variables
- [x] Barriers
- [x] Semaphores
- [x] Atomic operations
- [x] Lock-free data structures

## Phase 5: Coroutine Support ✅
- [x] Coroutine trait definition
- [x] Yield mechanism
- [x] Coroutine scheduler
- [x] State management
- [x] Coroutine handles
- [x] Integration with task system

## Phase 4: Error Handling ✅
- [x] Error type hierarchy
- [x] Result types
- [x] Error propagation
- [x] Panic handling
- [x] Error recovery
- [x] Diagnostic information

## Phase 3: Memory Pool ✅
- [x] Object pool implementation
- [x] Arena allocator
- [x] Memory recycling
- [x] Cache-aligned allocation
- [x] NUMA-aware allocation
- [x] Memory statistics

## Phase 2: Work-Stealing Scheduler ✅
- [x] Chase-Lev deque implementation
- [x] Worker thread management
- [x] Task stealing logic
- [x] Load balancing
- [x] Scheduler benchmarks

## Phase 1: Core Architecture ✅
- [x] Task abstraction
- [x] Executor trait
- [x] Basic scheduler interface
- [x] Thread pool implementation
- [x] Basic task spawning

## Next Steps
- [x] Comprehensive test suite (39+ core tests passing)
- [x] Example applications (basic_usage, async_timer working)
- [x] Documentation improvements (SSOT and consolidation notes)
- [ ] Performance benchmarks validation
- [ ] API stabilization  
- [ ] Production readiness review
- [x] SSOT consolidation: zero-copy communication primitives live under
      `moirai_core::communication`
- [x] Iterator windows/chunks consolidated under `moirai_iter::windows`
- [x] Placeholder cleanup: replaced stubs with explicit unsupported errors or working code
- [x] Zero-copy send returns value on failure to prevent data loss
- [x] **Critical Infrastructure**: Fixed executor to actually run tasks (was completely broken)
