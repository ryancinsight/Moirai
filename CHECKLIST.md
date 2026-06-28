# Moirai Development Checklist

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
- [x] SSOT consolidation: zero-copy moved to `moirai_core::communication::zero_copy`
- [x] Iterator windows/chunks consolidated under `moirai_iter::windows`
- [x] Placeholder cleanup: replaced stubs with explicit unsupported errors or working code
- [x] Zero-copy send returns value on failure to prevent data loss
- [x] **Critical Infrastructure**: Fixed executor to actually run tasks (was completely broken)
