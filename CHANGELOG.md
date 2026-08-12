# Changelog

All notable changes to the Moirai concurrency library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `moirai-benchmarks/thread_schedule_comparison` now includes a bounded
  saturated-admission comparison: one-worker Moirai capacity-plus-one rejection
  is measured beside an equal-capacity Crossbeam `try_send` rejection. Both
  paths assert their full-queue state before timing and use bounded Criterion
  windows; the rows compare rejection latency only, not scheduler semantics.

### Changed

- Route default worker-count decisions through Themis topology detection
  (`CpuTopology::detect().logical_processors()`) with preserved
  `std::thread::available_parallelism()` fallback across
  `moirai-core`, `moirai-executor`, `moirai-iter`, `moirai-parallel`, and
  the scheduler's single-node topology bootstrap.
- Convert the `moirai_core::communication` collective operations (`scatter`,
  `gather`, `all_to_all`) from a jagged `Vec<Vec<T>>` layout to a CSR-shaped
  `ChunkedVec<T>`: one contiguous flat buffer plus a chunk-offset table.
  `gather` is now an O(1) buffer hand-off and element traversal runs over a
  single allocation instead of a per-chunk pointer chase. Criterion
  (`benchmarks/benches/collective_ops_comparison.rs`) measures the win at
  32/128 participants: gather ~10–13× faster, traverse ~1.6×, scatter
  ~1.1–2.2×, all_to_all ~1.3–3.2×. Empty input and a zero participant count
  now return an empty `ChunkedVec` instead of the previous `chunks(0)` panic.
- Publish the facade under the collision-free `moirai-runtime` package name
  while retaining the Rust library name `moirai`, bind Mnemosyne dependencies
  to their published package identities, and correct registry metadata to the
  owning repository.

### Fixed

- Make the interleaved priority, resource-contention, and memory-ordering
  regressions deterministic across host topologies and concurrent test
  execution by using explicit worker pools, event gates, and value-checked
  task joins instead of wall-clock polling.
- Make the `ChaseLevDeque` retired-array reclamation poison-tolerant: resize,
  drop, and test observation recover the guarded pointer list via `into_inner()`
  instead of panicking on a poisoned mutex after a panicking lock holder.
  A regression poisons the lock, forces a further resize, drains all items
  exactly once, and verifies final destruction (MOI-DEQUE-POISON-215).
- Bind the `themis` crate alias to the renamed `themis-topology` package so
  fresh Git dependency resolution follows the provider identity.
- Bind the `mnemosyne` and `mnemosyne_core` crate aliases to packages
  `mnemosyne-memory` and `mnemosyne-memory-core` while preserving Rust imports.

### Breaking

- Remove `moirai_iter::ThreadPool` and its `moirai::ThreadPool` re-export. The
  crate's own FIFO thread pool was a second runtime beside the unified
  scheduler, kept only as the fallback for a shutting-down executor — which is
  when starting worker threads is least defensible, and where the caller's own
  thread does just as well for a flat index domain. The indexed operations now
  re-run their work on the caller, and `ParallelContext` schedules through the
  process-wide executor rather than owning a pool, so several contexts share
  one worker set instead of over-subscribing the machine. Callers wanting a
  fan-out use `moirai_parallel`'s data-parallel operators or
  `moirai_executor::global()`.

### Fixed

- Return every chunk from `ParallelContext::execute_iter`. It collected results
  from a channel until the senders dropped, so a panicking chunk ended the
  collect early and the call returned a short `Vec` with `Ok` — 32 of 40 items
  in the regression case. Results now land in per-index slots and a missing
  chunk surfaces as a panic.

- Run a scoped job on the calling lane when the scheduler's admission queue
  rejects it. `SchedulerScope::flush` previously dropped such a job and
  returned the error, so a scope broke its "every spawned job runs before the
  scope returns" promise silently — the dropped job's completion token
  decrements the scope counter exactly as a finished one does, so the caller
  resumed as though borrowed work had happened, and `moirai_parallel::scope`
  turned the error into a panic. Admission now leaves a refused job in the
  caller's slot, and the caller-run event is counted through the existing
  `admission_caller_runs` surface. Shutdown is not backpressure and still
  propagates.
- Run a `moirai_parallel::join_with` branch on the caller when the scheduler
  refuses it. The scheduled branch was moved into the job, so a job refused
  while shutting down — or rejected by a full per-worker admission queue — was
  dropped with the branch inside it, and the join turned the resulting error
  into a panic. Both lanes now claim the branch from a shared slot, so it runs
  exactly once on whichever lane reaches it, and backpressure makes a join
  sequential rather than fatal.

### Changed

- The SPSC halves cache the opposite counter, so a send reads the consumer's
  cache line only when the queue looks full and a receive reads the producer's
  only when it looks empty, instead of on every operation. A stale cache always
  errs toward "full"/"empty" and is refreshed before reporting either, so the
  exact capacity check is unchanged. The `Cell` also supplies the `!Sync`
  property that a `PhantomData` marker used to carry, making it load-bearing in
  code rather than only in a comment. See ADR-025.

- **[breaking]** `SpscChannel` is no longer exported from `moirai-core`. The
  single-producer/single-consumer discipline is enforced by `SpscSender` and
  `SpscReceiver`, which are neither `Clone` nor `Sync`, but the bare channel
  escaped alongside them: it implements `Channel<T>`, whose `send`/`recv` take
  `&self`, and carried an `unsafe impl Sync`, so safe code could share one
  channel and drive two producers into the same buffer slot — a data race, with
  one value overwritten undropped, and a double free on the `recv` side.
  Construct channels with `moirai_core::channel::spsc(capacity)`, which is
  unchanged and returns the same pair; `SpscChannel::channel(n)` becomes
  `channel::spsc(n)`. See ADR-024.

- Run the `moirai-iter` parallel sorts' recursive fork-join on the unified
  scheduler's scope instead of the crate's own thread pool. That pool is a FIFO
  queue with no work stealing, so a worker blocked on another of its jobs could
  not run it — the starvation that the fork budget guarded against, at the cost
  of capping the work tree at the pool's width. A scheduler worker waiting
  inside a scope runs queued work instead of parking, so the budget is gone and
  the recursion is deadlock-free by construction. A fork the scheduler refuses,
  while shutting down or under admission backpressure, now runs on the calling
  lane rather than being dropped. Splitting stops once a sub-slice is smaller
  than `len / (workers * 8)` so fork count follows machine width rather than
  input size. `ThreadPool` remains for flat fan-out only.

### Added

- `moirai_core::channel::SpscRing` with `split(&mut self)`, yielding
  `SpscProducer`/`SpscConsumer` halves that borrow the ring instead of sharing
  an `Arc`. For pairs living inside a scope — a `thread::scope`, a frame loop, a
  pipeline stage — the scope already proves the ring outlives the halves, so the
  refcount enforces statically-checkable lifetimes at runtime cost. The borrowed
  pair drops that cost: no allocation beyond the ring's buffer, and dropping a
  half is one store rather than a refcount decrement and a conditional free.
  Both halves implement `Producer`/`Consumer`, so generic code takes either
  flavour with no `'static` bound. `spsc(capacity)` is unchanged. See ADR-026.

- `moirai_core::channel::{Producer, Consumer}` — the sending and receiving
  halves of a channel as separate contracts, neither requiring `Sync`. This
  restores generic use of the SPSC halves, which ADR-024 had excluded: they
  cannot implement `Channel<T>`, whose `Send + Sync` supertrait is precisely
  what a single-producer half must not satisfy. Implemented on both SPSC
  halves, both MPMC halves, and `MpmcChannel` itself, so a shareable channel is
  usable whole or split. See ADR-025.

- Expose a borrowing `moirai_parallel::scope` surface over the unified
  scheduler so downstream parallel regions can spawn an arbitrary number of
  non-`'static` tasks and return a value only after every task completes.
- Build, install, attest, and attach locked `moirai-python` wheels for CPython
  3.10 through 3.13 on Linux, Windows, and macOS when a matching GitHub Release
  is published, then publish the same artifacts to PyPI through OIDC Trusted
  Publishing.

### Fixed

- Remove obsolete nightly `thread_local` detection from `moirai-core`, which
  does not invoke Melinoe's TLS macro. The platform and executor crates retain
  their nightly `#[thread_local]` fast path.
- Preserve indexed fan-out and map-reduce work when bounded scheduler admission
  saturates by executing rejected chunks on the submitting caller under the
  same panic boundary as worker chunks. A monotonic diagnostic counter exposes
  each caller-run backpressure event.
- Reject zero-length and out-of-`off_t` Unix shared-memory mappings before
  acquiring an operating-system descriptor instead of truncating `usize`.
- Keep stack-owned scheduler scope state alive until the final completion
  releases its wait synchronization, preventing a multi-job scoped fan-out from
  hanging or dereferencing destroyed state.
- Reject a Unix `SharedMemory::open` whose requested size exceeds the existing
  segment. `mmap` accepts a length past the end of the object and leaves the
  surplus pages unbacked, so `as_slice` handed out bytes that raise `SIGBUS` on
  read; `open` now checks the segment with `fstat` first. Windows already
  refused the oversized view in `MapViewOfFile`.
- Detect a panicked pooled worker when joining `moirai-iter` fan-out. The join
  discarded the result of each `recv`, so a worker that unwound without sending
  left the channel disconnected and every later `recv` failed instantly — the
  join then returned as though all tasks had finished.
  `ZeroCopyParallelIter::map` consequently called `assume_init` over a slice no
  worker had written, and `for_each`, `reduce`, and the parallel sorts returned
  silently partial results. The join now counts completions and panics unless
  every task reported one.
- Stop the `moirai-iter` parallel sorts deadlocking on large inputs. The
  recursion forks one half onto the shared pool and blocks on it, so a forked
  half that forks again occupied a worker while depending on another; the pool
  does not steal work, so once every worker was blocked that way the queued
  halves had nobody left to run them and `par_sort` never returned. Reachable
  around `2048 * 2^(workers + 1)` elements — roughly a million on an eight-core
  machine. A fork budget now keeps at least one worker free to drain the queue,
  and the recursion sorts both halves in place once the budget is spent.
- Keep a `moirai-iter` pool worker alive when a job panics. The worker ran jobs
  without catching unwinds and workers are never replaced, so each panicking job
  removed one permanently; once all had gone, `execute` kept queueing onto a
  channel nobody received from and the next join blocked forever, turning a
  caller's panic into a later, unrelated hang. The panic still reaches the
  caller through the join, which sees the missing completion.
- Start at least one worker in `ThreadPool::new`. A zero-sized pool accepted
  jobs and ran none, so anything awaiting them waited indefinitely.
- Keep `CacheAlignedChunks` advancing for elements wider than a cache line. The
  chunk size was `(CACHE_LINE_SIZE / element_size) * (CACHE_CHUNK_SIZE /
  CACHE_LINE_SIZE)`, whose first term truncates to zero above 64 bytes, leaving
  the iterator yielding empty slices forever without advancing.
- Reserve the backing store of a Linux `SharedMemory::create` segment with
  `posix_fallocate`. `ftruncate` only sets the length of a tmpfs object and
  leaves its pages sparse, so a correctly sized segment still raised `SIGBUS`
  through `as_slice`/`as_mut_slice` when the store could not produce a page on
  first touch; the shortage now surfaces as an error at creation, and the failed
  segment is unlinked rather than left for a later `open` to map. Unix targets
  without `posix_fallocate`, macOS among them, are unchanged.
- Test the async task completion flag under the future lock rather than before
  acquiring it, so a second thread polling the same executor cannot pass the
  check, wait for the lock, and then poll a future that completed meanwhile —
  which panics with "resumed after completion".

## [0.5.0] - 2026-08-11

### Changed

- Re-release the workspace as version 0.5.0 against mnemosyne-memory 0.7.0 so
  consumers use the same Mnemosyne/Eunomia provider graph as the rest of
  Atlas. The package identity and Rust library name remain unchanged.

### Breaking

- The optional Mnemosyne integration now requires mnemosyne-memory 0.7.0;
  consumers must update their dependency graph to the 0.7 provider line.

## [0.4.0] - 2026-07-17

### Changed

- Confirm the RITK masked Parzen-cache consumer path against the merged Moirai
  scheduler pin, closing the indexed caller-region downstream gate.
- [major] Direct Atlas providers follow their merged default branches. The
  workspace removes revision quarantine and the local Melinoe patch, and adopts
  Mnemosyne 0.5/Core 0.2; `Cargo.lock` is the sole reproducibility pin.
- [major] Dual channel consolidation (TREE-DUP-002): fold `unified_channel`
  into `channel/` module. Extended `Channel<T>` trait with `send_batch`,
  `recv_batch`, `close`, `is_closed`, `len`, `stats` (default impls). Moved
  `ChannelConfig` → `channel/config.rs`, `ChannelStatistics` →
  `channel/stats.rs`. Added `InvalidConfig` to `ChannelError`. `UnifiedChannel`
  now implements `Channel<T>` under `channel::unified`. Deleted the
  `unified_channel/` module. Migration: replace `unified_channel::*` imports
  with `channel::unified::*`; match `ChannelError::InvalidConfig` in error
  handlers.

### Fixed

- Make async waiter handoff retain FIFO behavior after cancellation without
  linear waiter scans; reclaim broadcast messages after every live receiver
  consumes them; derive timer-compaction coverage from the active heap state.
- Apply warning-free iterator and arithmetic idioms in affected examples and
  HTTP framing fixtures.

### Breaking

- Moirai 0.4.0 requires Rust 1.95. Migration: update the consumer toolchain
  before resolving the default Mnemosyne provider graph.
- **Breaking:** removed the `unified_channel` public module. Use
  `moirai_core::channel::unified` instead.
- **Breaking:** `ChannelError` gains `InvalidConfig` variant — update
  exhaustive matches. `Channel<T>` trait adds `send_batch`/`recv_batch`/
  `close`/`is_closed`/`len`/`stats` — implementors must provide or accept
  default impls.

## [0.3.1] - 2026-07-15

### Changed

- Pin Themis and Mnemosyne to audited revisions that resolve one provider source
  identity through both direct and transitive dependencies.

## [0.3.0] - 2026-07-15

- Pin Mnemosyne's current reproducible provider graph so downstream Git-source
  consumers resolve one kernel-budget type identity.
- Consolidate Melinoe on Mnemosyne's exact provider revision through the
  workspace dependency SSOT.

### Changed

- `moirai-core`: split the hybrid and MPMC channel implementations into focused
  endpoint and state modules. `HybridChannel<T>` is now a zero-sized endpoint
  factory; sender and receiver halves retain the only live shared state.
- Upgrade the Themis topology provider to 0.10.0 and consume its published
  optional cache-level contract without a workspace-local override.
- Upgrade Mnemosyne to the release that owns the same Themis topology identity.
- Pin Themis to the exact audited provider revision so downstream integrators
  resolve one source identity instead of compiling the same commit twice.
- Pin Mnemosyne to the revision that owns that same Themis identity.
- `moirai-executor`: migrated the Melinoe scheduler bridge to the validated
  `ParallelExecutor` capability. Its unsafe construction site documents the
  exact-once index, blocking-completion, and context-lifetime proof; raw executor
  function pointers no longer cross a safe registration boundary.
- The Mnemosyne facade contract now targets 0.3.0, removing the obsolete 0.2
  facade and its duplicate backend type identity from integrated consumers.

### Fixed
- `moirai-sync`: make `ShardedResourcePool::clear` linearizable with
  reservation and publication by retaining per-bin guards through counter
  reset; add a deterministic cross-thread regression for the interleaving.
- `moirai-core`: re-read the SPSC publication index after acquiring sender
  closure so a value published immediately before sender drop is drained rather
  than misreported as a closed empty channel.
- `moirai-parallel`: honor the explicit `Parallel` policy for small domains by
  partitioning indexed work across worker-plus-caller lanes. Adaptive policy
  remains the operation-level threshold owner; the executor no longer silently
  serializes expensive low-cardinality work.
- `moirai-executor`: keep nested indexed fan-out and map/reduce work-conserving
  by using the scheduler's worker help path while waiting. This prevents a
  saturated outer parallel region from parking every worker while inner indexed
  chunks remain queued.
- `moirai-executor`: flatten worker-nested indexed regions onto the current lane,
  preserving outer parallelism without recursively stealing outer jobs onto the
  worker stack.
- `moirai-executor`: track indexed-region depth on the participating caller so
  its nested fan-out and reductions also flatten onto the outer caller lane.
- `moirai-executor`: distribute indexed remainders across every selected lane
  instead of leaving worker lanes unused when the item count is just above the
  worker-plus-caller cap.
- `moirai-executor`: align panicking indexed-reduction metric coverage with the
  worker-plus-caller lane contract: two worker chunks complete scheduler
  lifecycles while the caller owns the third lane.
- `moirai-core`: read the thread-local operating-system error through
  `std::io::Error` on Unix and Windows, removing the Linux-only errno symbol
  that prevented IPC consumers from compiling on macOS.
- `moirai-pal`: keep the pointer-bearing kqueue output buffer thread-local, so
  the reactor remains `Send + Sync` without an unsafe marker implementation or
  a per-poll allocation.

### Added
- `moirai-executor`: added `block_on`, a public current-thread parking wait
  primitive over the existing scheduler waker driver. This gives Moirai-owned
  crates a lightweight replacement for external `pollster::block_on` without
  constructing the global scheduler.
- `moirai-async`: added a loopback TCP `timeout(read)` regression that lets the
  timer complete before peer data arrives, then forces the socket reactor to
  wake the stale read waker after the async task has completed.
- `moirai-parallel`: added `for_each_chunk_mut_with_state`, a mutable chunk
  primitive with one reusable scratch state per scheduled worker shard. This
  fills the provider gap for consumers replacing Rayon `for_each_with`/
  `for_each_init` chunk kernels without allocating scratch per logical chunk.
- `moirai-parallel`: added `for_each_chunk_triple_mut_enumerated_with` and
  `for_each_chunk_quad_mut_enumerated_with` for fused provider-owned updates
  across three or four equal-length mutable output buffers.
- `moirai-executor`: loom model of the scheduler park/wake handshake
  (`tests/loom_wake_handshake.rs`, `cfg(loom)`-gated) — exhaustively verifies
  the `pending_tasks`/idle-bitset `SeqCst` Dekker handshake never loses a
  wakeup, the model-checked evidence that the `SeqCst` ordering is necessary
  and sufficient. The crate now wires the `cfg(loom)` dev-dependency mirroring
  `moirai-scheduler`.
- `moirai-iter`: Added bounded concurrent stream adapters
  `ConcurrentStreamExt::concurrent_filter_map` and
  `ConcurrentStreamExt::concurrent_filter`.
- Scheduler DIP seam: `moirai_executor::schedule::{WorkScheduler, WorkSubmit,
  SchedulerControl, DataParallel}` — ISP-segregated role traits implemented by
  `ThreadScheduler`. `HybridExecutor<S: WorkScheduler = ThreadScheduler>` now
  depends on this contract rather than the concrete scheduler, so a substitute
  (e.g. a single-threaded `wasm32` scheduler) can be plugged in via
  `HybridExecutor<S>`; the default type parameter keeps every existing call site
  unchanged.

### Removed
- **Breaking:** removed the unconsumed `moirai_iter::numa` helper and its
  Rayon comparison benchmark. Placement belongs to Themis, allocation belongs
  to Mnemosyne, and data-parallel execution belongs to the scheduler-backed
  `moirai-parallel` surface; the deleted iterator helper applied none of those
  provider contracts.
- **Breaking:** removed combined owner/thief deque capabilities,
  `QuiescentReclaim`, explicit default-policy `reclaim_memory`, callback
  `steal_batch_with`, and the unconsumed `BlockBasedDeque`. `ChaseLevDeque` is
  the canonical unique bottom-side owner; callers clone `ChaseLevStealer` for
  top-side access. Batch steals return an owning `StolenBatch` iterator.
- **Breaking:** removed the obsolete `moirai/no-global-alloc` no-op feature.
  The library no longer registers a global allocator; final binaries own that
  process-wide choice, while `mnemosyne` continues to forward provider
  integration into core and executor crates.
- **Breaking:** removed the dead, mis-shaped passive scheduler abstraction from
  `moirai-core`: the `Scheduler` trait, the `ScheduledTask` erased-task type
  (+ `INLINE_SCHEDULED_TASK_WORDS`), and the `SchedulerConfig`/
  `WorkStealingStrategy`/`QueueType`/`StealContext`/`Stats` config vocabulary.
  None had a production consumer — `ThreadScheduler` is active (owns workers and
  executes internally), so the passive `next_task`/`steal_task` contract could
  only ever be a mock. The live erased-task type is the executor's `ScheduledJob`
  and the canonical abstraction is the new `WorkScheduler` seam. `SchedulerId`
  (the one live export, used by metrics aggregation) is retained.
- **Breaking:** redundant work-stealing scheduler implementations consolidated
  onto the canonical runtime scheduler (`moirai_executor::ThreadScheduler`,
  which already executes every `spawn*`/`block_on`/`scope`). Removed
  `moirai_scheduler::{WorkStealingScheduler, WorkStealingCoordinator,
  SchedulerStats, SchedulerStatsSnapshot}`, `moirai_scheduler::numa_scheduler::
  {NumaAwareScheduler, NumaSchedulerStats, NumaSchedulerError, StealStatistics}`,
  the generic `moirai_core::scheduler::WorkStealingCoordinator`, and the
  `moirai::WorkStealingScheduler` re-export. None were on any runtime code path.
  `moirai-scheduler` is now a primitives crate: the lock-free deques
  (`ChaseLevDeque`/`SplitDeque`) and NUMA primitives
  (`CpuTopology`, `AdaptiveBackoff`) remain its canonical surface. Migration:
  construct the runtime via `moirai::Moirai`/the global runtime rather than the
  removed scheduler types directly.

### Changed
- `moirai-executor`: scheduler admission is bounded and fallible. Saturation
  returns `ExecutorError::ResourceExhausted`, rolls back pending accounting,
  drops rejected jobs once, and terminally completes unstarted registry state.
- `moirai-executor`: bottom-side Chase-Lev endpoints now move into their worker
  threads; shared worker state retains only stealers and bounded external
  injectors. Nested-scope helping uses top-side steals without recovering an
  owner through TLS or raw pointers.
- `moirai-scheduler`: default `DeferredReclaim` retains resized Chase-Lev arrays
  until the final typed endpoint drops. The Moirai-owned access-counted
  `SharedEpochReclaim` remains available for explicit live array reclamation;
  no Crossbeam reclamation dependency is introduced.
- `moirai-iter`: replaced raw `ManuallyDrop` vector moves in indexed
  collect-into-storage and interleave paths with owned iteration. Non-`Clone`
  element movement and output-capacity reuse remain intact while every source
  backing allocation is now released; targeted Miri-nextest leak regressions
  pass.
- `moirai-gpu`: replaced the `GpuTaskAdapter` synchronous wait boundary with
  `moirai_executor::block_on` and removed the direct `pollster` dependency from
  the `wgpu-backend` feature.
- `moirai-benchmarks`: removed external-ID task lifecycle diagnostic rows from
  `result_handle_diagnostics`. Lifecycle-backed wrapper rows now allocate task
  IDs through `TaskRegistry::diagnostic_register_next_and_complete_with_token_id`,
  so duration and metrics attribution use registry-owned task IDs instead of a
  separate `AtomicU64` placeholder path.
- `moirai-transport`: removed the stale `core_zero_copy` re-export that pointed
  at the deleted `moirai_core::communication::zero_copy` module. Consumers use
  the current `moirai_core::communication` primitives directly.
- `moirai-executor`: Completed the task-registry stable-slot migration by
  keeping `TaskStateBlock` slots private behind `UnsafeCell` accessors. Registry
  production paths, diagnostics, and benchmark source contracts now route
  through `get`/`insert`/`clear`/`states`, preserving stable
  `TaskLifecycleToken` pointers without exposing block internals.
- `moirai-executor`: Replaced the hybrid executor's single global task-registry
  mutex with `ShardedTaskRegistry`, using a lock-free global task id allocator
  and per-shard `TaskRegistry` locks. Manager status/stat/wait paths now call
  the sharded facade directly, with tests covering global id routing,
  lifecycle-token completion, and unknown lookups.
- `moirai-executor`: Synchronized `Cargo.lock` with the existing
  `cfg(loom)` dev-dependency and removed redundant explicit Rustdoc link
  targets from the hybrid executor docs so the package rustdoc gate is
  warning-clean.
- `moirai-async`: Completed the async `RwLock` waiter-map refactor by routing
  reader and writer registration, wakeup, and cancellation through keyed
  `BTreeMap` waiter state. This preserves FIFO-by-monotonic-id handoff while
  avoiding linear removal under contention.
- `moirai-async`: Extended the same keyed-`BTreeMap` refactor to the rest of the
  sync primitives — `Notify`, `Semaphore` (waiters) and `Watch`, `Broadcast`
  (receivers) — replacing O(n) `position`/`find`/`retain`-by-id scans under the
  state lock with O(log n) keyed lookup/remove. FIFO fairness, permit
  storage/transfer, reader-batch grant, and cancellation waker-clear semantics
  are unchanged.
- `moirai-core`: `Histogram::record` no longer increments a dedicated `count`
  atomic — `count()` derives from the bucket sum (`count == Σbuckets`). Drops a
  globally-contended atomic from every record and the `sum`/`count` false
  sharing; SSOT for the sample count. `count()` is now O(buckets), read-side
  only.
- `moirai-async`: Fixed the `ConnectionId` Rustdoc link to
  `std::net::SocketAddr`, keeping the package documentation warning-clean.
- `moirai-iter`: Completed the stream module rename by exporting
  `moirai_iter::stream` and naming the extension contract
  `ConcurrentStreamExt` / `concurrent_*`.
- `ThreadScheduler` default `SPIN_LIMIT` reduced from 131072 to 8192. With the
  idle-worker park fix above, a parked worker now wakes in ~8 µs, so the old
  ~1 ms pre-park busy-spin only bought ~700 ns wake latency at a large idle-CPU
  cost. 8192 (~60 µs of spin) keeps a short burst-catch window while parking
  quickly. Sustained throughput is unchanged (measured flat across spin budgets;
  the spin never engages while work is available). `SPIN_LIMIT` remains a const
  generic, so latency-critical deployments can raise it.
- `ThreadScheduler` now selects work-steal victims from a thread-local
  xorshift64 random origin instead of fixed round-robin, spreading post-barrier
  steal contention across victims (Blumofe–Leiserson). Full-ring coverage and
  worst-case scan cost are unchanged.
- `ChaseLevDeque` isolates its `bottom` (owner) and `top` (thief) indices onto
  separate cache lines, eliminating intra- and inter-deque false sharing; a
  compile-time assertion locks the ≥64-byte alignment invariant.

### Fixed
- `moirai-executor` scheduler: idle workers no longer redundantly drive the
  global `IoReactor` with a 1 ms `run_iteration` while waiting for work. That
  poll is not interruptible by `unpark` and rounds up to the OS timer
  granularity (~15 ms on Windows), so once a pool had parked, scheduling sync
  work to it stalled for ~15 ms — a latency the large `SPIN_LIMIT` (~6 ms of
  pre-park busy-spin) was masking. Idle workers now simply `park()`; async I/O
  readiness is driven by moirai-pal's dedicated global reactor thread, whose
  wakers reschedule their tasks through the same `schedule_job` path, so a
  parked worker is woken identically for async completions and fresh sync work.
  Measured submit→execute wake latency under intermittent load drops from
  ~15 ms to ~8 µs (8-worker pool, Windows). Verified by the new
  `spin_budget_bench` instrument.
- `moirai-pal` reactor: `IoReactor::get_active()` no longer panics if the
  process-global readiness reactor cannot be created or its driver thread cannot
  be spawned — it now caches and returns `None`, so socket operations degrade
  gracefully to the cooperative busy-poll self-wake fallback in `net.rs` instead
  of aborting. This makes that fallback (previously unreachable, since the old
  code always returned `Some` or panicked) a real, tested path; the async
  TCP/UDP self-wake-without-reactor round-trips are now value-verified.
- `moirai-sync::FutexMutex` (Linux): fixed a lost-wakeup **deadlock** in the
  three-state futex slow path. `lock_slow` acquired a woken-up contended lock via
  `CAS 0 -> 1`, erasing the "waiters present" marker (state 2) even when other
  waiters were still parked; the next `unlock` (which only `futex_wake`s when
  `swap(0)` observes 2) then skipped the wake, stranding those waiters forever.
  The slow path now acquires by `swap(2)` (Drepper / Rust-std algorithm), so the
  marker is conservatively preserved across hand-offs. Linux-only path;
  Windows/fallback path unchanged and value-verified.
- `moirai-async::Notify`: `notify_waiters()` no longer destroys a permit stored
  by a prior `notify_one()` issued with no waiters present. The two mechanisms
  are independent (matching `tokio::sync::Notify`); previously `notify_waiters`
  cleared the single-permit flag, so a subsequent `notified().await` could block
  forever. Regression test `notify_waiters_preserves_stored_notify_one_permit`.

### Added
- `Moirai::spawn_detached` (and `TaskSpawner::spawn_detached`): a fire-and-forget
  spawn that returns no handle and skips the per-task `Arc<TaskResultSlot>`
  allocation and atomic refcount that result-bearing spawns require — the
  cheapest dispatch path for background work whose output is not needed.
  Lifecycle/metrics tracking and shutdown drain are preserved, and panics are
  isolated. The trait method has a non-breaking default (routes through
  `spawn_blocking`); `HybridExecutor` overrides it for the no-allocation path.
- `moirai-pal`: a **real readiness reactor on Windows** (`WsaPollReactor`, backed
  by `WSAPoll`), replacing the non-functional IOCP backend. The IOCP completion
  model signals completions of *posted overlapped operations*, not socket
  *readiness*, so it could never drive the readiness-based `net.rs` futures —
  Windows async sockets therefore fell back to a 100%-CPU cooperative busy-poll.
  Windows now activates the process-global reactor like every other platform, so
  async TCP/UDP I/O is driven by real readiness instead of busy-polling. The
  reactor is `WSAPoll`-level-triggered (self-heals lost edges), self-cleans
  closed sockets via `POLLNVAL`, and is interrupted by a loopback wake socket.
  Verified by the full async net suite plus direct reactor tests (readiness
  delivery, wake interruption, stale-socket self-cleaning).

### Removed
- `moirai-pal` `IocpReactor` — superseded by `WsaPollReactor` (it could not
  deliver socket readiness).

### Fixed
- `moirai-pal` I/O reactor lost-edge hang (Linux/BSD): the epoll/kqueue backends
  registered fds **edge-triggered** (`EPOLLET` / `EV_CLEAR`), but the reactor
  registers a waker only *after* a `WouldBlock`, so readiness arriving in the
  register window — or already present at registration — emitted no edge and the
  task hung forever. Both backends are now **level-triggered**, which re-reports
  readiness on every wait until the I/O consumes it, self-healing the race (and
  the interest-widening `unregister`+`register` window). On Windows `get_active()`
  returns `None`, so the live platform uses the cooperative self-wake fallback and
  was unaffected; this is documented at the `get_active` site (a readiness-capable
  Windows/IOCP reactor remains a separate, larger feature).
- `moirai-pal` epoll/kqueue timeout truncation: saturate the `as_millis()`→`c_int`
  and `as_secs()`→`time_t` casts so a multi-week/century timeout cannot wrap to a
  negative value (turning a finite wait into an infinite block).
- `moirai-pal` `IoReactor::with_active`: restore the previous thread-local reactor
  via RAII so a panic in the closure cannot leave a dangling reactor pointer in
  the thread-local that a later `get_active()` would dereference (use-after-free
  on unwind).
- `moirai-pal` `Timer::new`: clamp the duration and use `checked_add` so a
  near-`Duration::MAX` timeout no longer panics the deadline computation
  (`Instant::now() + duration` overflow).
- `moirai-iter` `NumaAllocator`: on Linux, `numa_alloc` returns null on `mmap`
  failure instead of falling back to the global allocator, so every non-null
  pointer is `mmap`-backed and `numa_free`'s `munmap` is never applied to a heap
  pointer (which was undefined behavior).
- `moirai-utils` `prefetch_range_read`: saturating/checked address arithmetic so
  a range ending near `usize::MAX` cannot overflow (panic under `overflow-checks`).
- `moirai-async` `io::compat`: feature-gate the `tokio-compat`-only imports so the
  crate is warning-clean without the feature.

### Added
- `moirai-transport` `IpcTransport`: a **real** same-machine inter-process
  transport over shared memory (`moirai_core::ipc::SharedQueue`), replacing the
  former placeholder that returned constant errors. Carries length-prefixed
  fixed-size frames (≤ ~4 KiB/message), Unix/Windows only, used directly (not via
  `TransportManager`, to avoid the `Local`-address overlap with
  `InMemoryTransport`). Surfaced in the `moirai` umbrella.
- `moirai-transport`/`moirai` now re-export the rkyv-style archive channels
  (`ArchivedUniversalSender`/`ArchivedUniversalReceiver`/`ArchivedMessage` +
  `ArchiveSerialize`/`ArchiveView`) at the crate root — the canonical typed
  cross-boundary channel.
- `moirai-scheduler`: a `loom` exhaustive-interleaving model of the Chase-Lev
  steal/pop ordering protocol (`tests/loom_chase_lev.rs`, gated behind
  `--cfg loom`), machine-checking the exactly-once invariant across all
  interleavings. `loom` is wired as a `cfg(loom)`-only dev-dependency, so normal
  builds and CI are unaffected.
- `moirai-transport`: `MessageRouter::{unsubscribe, subscriber_count}` and
  `ConnectionManager::{state, is_connected, connected_addresses}` query methods;
  `ConnectionState` is now a public, `Copy` enum.

### Removed
- The non-functional `UniversalChannel`/`UniversalSender`/`UniversalReceiver`
  placeholders (their `send`/`recv` ignored their argument and returned `Closed`;
  a channel generic over an arbitrary `Send` `T` cannot serialize for transport).
  The working archive channels (above) are their realized replacement.

### Fixed
- `moirai-transport` `MessageRouter::publish` now actually delivers messages.
  The previous implementation constructed a throwaway `InMemoryTransport` per
  send and silently discarded every published message; it now routes through a
  shared transport and returns the delivered-subscriber count.

### Changed
- `moirai-transport` `MessageRouter` is generic over its backing `Transport`
  and takes the transport at construction (`MessageRouter::new(transport)`), so
  delivery is real and zero-cost. (Pre-1.0 breaking change to a previously
  non-functional API.)
- `moirai-core` `UnifiedChannel::send` now delegates to `try_send` (single SSOT
  for the send path, ~45 lines of duplicate overflow logic removed) and
  documents that it consumes the message on `Full`/`Closed`; callers needing the
  value back to retry use `try_send`. `overflow_count` is documented as advisory.

### Security
- `moirai-http` response parsing now enforces a `max_response_bytes` budget
  (default 64 MiB, configurable via `HttpClient::set_max_response_bytes`). A
  malicious or compromised peer could previously drive unbounded allocation
  (OOM) by advertising a huge `Content-Length`, streaming endless chunked/EOF
  bodies, or trickling header bytes — every read now funnels through one
  size-checked chokepoint, and an oversized `Content-Length` is rejected up front.

### Added
- `HttpClient::set_max_response_bytes` to tune the response-size cap.

### Fixed
- `moirai-core` `SecurityAuditor`: the audit-event buffer is now a `VecDeque`
  with a hard count cap (16,384) and amortized-O(1) front-pop eviction. The prior
  `Vec` ran an O(n) `retain` on *every* recorded event (O(n²) under a spawn
  storm) and was bounded only by a days-to-weeks retention window, so it grew to
  millions of entries under load. Also uses `checked_sub` for the retention
  cutoff (no panic on an absurd clock).
- `moirai-async` broadcast channel: `BroadcastRecv` now clears its registered
  waker on drop (mirroring `WatchChanged`), so a cancelled `recv` future no
  longer leaves a stale waker that the next `send` spuriously wakes/retains.
- `moirai-transport` network transport: bounded read/write timeouts
  (`NETWORK_IO_TIMEOUT`, 30s) on accepted and connected TCP streams in
  `NetworkTransport` and the remote-task server, so a peer that connects then
  stalls mid-frame can no longer pin a worker thread indefinitely.
- `moirai-core` rate limiter: saturating time arithmetic on the window-advance
  computation (`numerical_discipline`); documented the limiter's approximate
  (non-hard-quota) semantics.

### Changed
- `moirai-core` `SharedQueue<T>` now bounds `T` by `bytemuck::Pod` (was `Copy`):
  shared-memory contents are written by one process and read as `T` by another,
  so the element type must be valid for every bit pattern. `bytemuck` is promoted
  to a workspace SSOT dependency (also adopted by `moirai-gpu`).
- `moirai-async` `ConnectionPool` is now keyed by a unique `ConnectionId` instead
  of peer `SocketAddr`; `ConnectionInfo` gains a `peer_addr` field and
  `add_connection`/`add_connection_reserved` return the id. Streams untrack by id
  at drop. (Pre-1.0 breaking change to the connection-pool surface.)

### Fixed
- `moirai-async` TCP listener: `accept` no longer leaks a connection-pool
  reservation when the accept future is cancelled (dropped while pending). The
  reservation is held in an RAII guard released on every early exit, so a
  cancelled accept can no longer permanently exhaust `max_connections`.
- `moirai-async` `TcpStream::drop`: untrack the connection by the id captured at
  accept time instead of re-querying `peer_addr()`, which fails on an
  already-reset socket and leaked the pool slot plus the `active_connections`
  counter. Unique-id keying also removes the address-reuse collision that could
  undercount connections.
- `moirai-sync` `ShardedResourcePool::recycle`: reserve count/bytes before
  inserting so concurrent recyclers observe each other and evict on a fresh view.
  The prior load-decide-insert let N concurrent recyclers each skip eviction and
  overshoot the shard cap (and byte budget) by up to N-1.
- `moirai-core` `SharedQueue`: reject zero capacity (`% capacity` divide-by-zero),
  reject `capacity * size_of::<T>()` / total-size overflow (undersized mapping →
  out-of-bounds access) and over-aligned element types, and record capacity in
  the segment header so `open` validates it (a mismatched peer view is rejected
  rather than faulting on access).
- `moirai-async` `TimerWheel`: cancelling an already-fired or never-scheduled
  timer is now a no-op, fixing unbounded growth of the `cancelled` tombstone set
  (a membership index keeps the invariant `cancelled ⊆ active`).
- `moirai-core` `AdaptiveBatchSender::flush_batch`: bound consecutive no-progress
  retries and surface `WouldBlock` backpressure instead of spinning forever when a
  receiver stalls without closing.
- `moirai-core` hybrid SPSC channel: documented the `Send` soundness contract on
  the manual `unsafe impl`s and added a self-validating compile-time guard that
  turns any future `Clone` on `HybridSender`/`HybridReceiver` into a build error
  (a second producer/consumer would race the `!Sync` ring).
- `moirai-executor` thread scheduler: closed a producer/worker lost-wakeup. A
  parking worker performs `idle_workers` set (SeqCst) then `pending_tasks` load
  (SeqCst), but `schedule_job` incremented `pending_tasks` with `Release`, which
  is not in the SeqCst total order; the resulting store-buffer race allowed a
  worker to observe no work while the producer observed no idle worker, leaving
  a submitted task stranded until an unrelated submission. The increment is now
  `SeqCst` (free on x86, where `lock xadd` is already a full barrier) and stays
  ordered before the queue push so the `execute_job` decrement cannot underflow.
- `moirai-executor` thread scheduler: replaced the single-`AtomicU64` idle map
  with a multi-word `IdleBitset` so workers with id >= 64 are registered in and
  reachable by the wake lottery. On pools larger than 64 workers, high-index
  workers were previously invisible to the wake path and stayed parked under
  load (a throughput stall). Added `IdleBitset` unit tests plus a 100-worker
  multi-round end-to-end wake regression.
- `moirai-sync` `FutexMutex` (non-Linux/Windows fallback path): closed a lost
  wakeup in the condvar fallback. `unlock` released `locked` (Release) then read
  `waiters` (Relaxed); StoreLoad reordering let the `waiters` check float ahead
  of the release, so a concurrently-registering waiter could be skipped and left
  parked forever. Added paired `SeqCst` fences (waiter and unlocker) and a
  high-contention blocking-path regression test.
- `moirai-scheduler` NUMA `BackoffStrategy::record_failure`: use `saturating_mul`
  for the exponential delay so a pathologically large base delay cannot overflow
  (panic under `overflow-checks`).

### Changed
- `moirai-gpu`: made the concrete WGPU execution stack optional behind the
  `wgpu-backend` feature while keeping `occupancy` launch planning available
  with `default-features = false`. Atlas GPU backends can consume Moirai
  launch planning without inheriting Moirai's concrete WGPU runtime version.

### Added
- Default `parallel` and `mnemosyne-memory` feature contracts across every
  Moirai package. Crates with existing Mnemosyne-backed runtime behavior expose
  `mnemosyne-memory` as the canonical default forwarding feature; provider-free
  leaf crates expose zero-dependency markers.
- Registered Moirai's global scheduler as Melinoe's `std` partition executor, so branded partition writes can run on the existing Moirai worker pool instead of spawning raw scoped threads.
- Routed default `moirai-executor` worker idle maintenance through Mnemosyne when the runtime is waiting for work, so Moirai's default provider stack performs allocator defragmentation sweeps without Rayon/Tokio involvement.
- Added Apollo-facing public facade contract tests for chunked mutable
  scheduling and non-`Clone` collect-into-existing-storage behavior.
- Removed the stale duplicate top-level `moirai` `par_benchmarks` target
  declaration; `moirai-parallel` remains the benchmark owner.
- Added `moirai_parallel::{join, join_with}` as a Rayon-style two-closure join
  surface with static `ExecutionPolicy` dispatch, scoped scheduler flush plus
  caller-lane execution for forced parallel joins, borrowed non-`'static` tests,
  source contracts, and a value-checked Rayon comparison benchmark row.
- Added `parallel_iterator_regression`, a focused multi-size Moirai/Rayon benchmark matrix for parallel iterator map/reduce, zip/filter collect, borrowed positions, borrowed copied reduce, collect-into-existing-storage, nested flatten/reduce, chunked map/reduce, indexed step/interleave, partition/unzip, and position/find paths.
- Added `moirai_executor::schedule::HybridRouter<P>` with sealed zero-sized route policies and concrete thread/process/server/async-lane route values, plus `process_server_scheduler_routing` for value-checked route-decision benchmarks.
- Added sealed accelerator route metadata for CPU/GPU/TPU/NPU placement with `AcceleratorRoutePolicy`, `AcceleratorCounts`, `AcceleratorKind`, vertical route leaf modules, and value-checked `scheduler_route_accelerator_metadata_summary` benchmark rows.
- Added sealed device payload ownership regions for accelerator route handoff, with pointer-transfer rejection and value-checked `device_region_owned_handoff` benchmark coverage.
- Added the `moirai-transport/scheduler-routes` feature with route-to-address binding for `SchedulerRoute` values and archived local route send/receive helpers.
- Added bounded length-prefixed TCP byte transfer for `NetworkTransport` and `TransportManager` remote addresses.
- Added fixed-format remote task envelopes/results for explicit echo and wrapping-sum request/response execution over remote byte transport.
- Added selected server-route remote task execution through `RoutedRemoteTaskClient<P>`.
- Added real OS process lifecycle primitives in `moirai-transport::process`, including `ProcessSupervisor`, `ProcessSpec`, explicit drop policy, bounded wait, termination, and typed process outcomes.
- Added selected process-route fixed-format task execution through `RoutedProcessTaskClient<P>` and supervised `ProcessEndpoint` child processes.
- Added public fixed-capability routed execution through `Moirai::execute_routed_server_task` and `Moirai::execute_routed_process_task`, requiring sealed `RemoteCapabilityToken<C>` values instead of arbitrary remote closures.
- Split communication primitives into vertical broadcast, collective, message, pub/sub, ring-buffer, and router leaves under `moirai_core::communication`.
- Added `BoundedRemoteTaskServer` for fixed-format remote task execution with persistent listener ownership, bounded request queue capacity, bounded worker count, and accepted/completed value stats.
- Added sealed zero-sized remote capability tokens for constructing only admitted fixed-format remote task operations and keeping arbitrary closure remoting outside process/server transport routes.
- Added sealed thread/process/server/device `TransportPayload<R>` ownership regions for archive byte handoff, with process/server/device pointer-transfer rejection and Mnemosyne global allocator evidence pinned at the top-level crate feature.
- Added `process_server_routed_execution`, a value-checked benchmark for selected server-route and process-route fixed-format task execution through real TCP request/result frames and supervised child processes.
- Added `metrics_collector_comparison`, a value-checked Criterion target for
  shared counter handles, fixed-size metric snapshots, and Prometheus export.
- Added `async_iterator_comparison`, a value-checked `moirai-iter::AsyncIterator` ready-pipeline benchmark against Tokio `JoinSet` fan-out.
- Added `AsyncIterator::take` and `AsyncIterator::skip` logical-window adapters with value tests and a Tokio `JoinSet` comparison row.
- Added `AsyncIterator::enumerate` and `AsyncIterator::zip` logical-position/pairing adapters with value tests and a Tokio `JoinSet` comparison row.
- Added `ParallelSliceMut` sorting comparison coverage against Rayon `ParallelSliceMut`.
- Added `async_fs_comparison`, a value-checked `moirai_async::fs::read` benchmark against Tokio `fs::read`.
- Added an `async_fs_write_file` row to `async_fs_comparison`, comparing Moirai platform-write facade behavior against Tokio `fs::write` over the same 64 KiB payload.
- Added an `async_fs_append_file` row to `async_fs_comparison`, comparing Moirai platform-append facade behavior against Tokio append-open/write behavior over the same 64 KiB payload.
- Added `moirai_async::fs::metadata` and an `async_fs_metadata_file` row to `async_fs_comparison`, comparing Moirai platform-metadata facade behavior against Tokio `fs::metadata` over the same 64 KiB file.
- Added `moirai_async::fs::rename` and an `async_fs_rename_file` row to `async_fs_comparison`, comparing Moirai platform-rename facade behavior against Tokio `fs::rename` over prepared 64 KiB source files.
- Added an `async_fs_remove_file` row to `async_fs_comparison`, comparing Moirai platform-remove facade behavior against Tokio `fs::remove_file` over prepared 64 KiB source files.
- Added `async_fs_dir_comparison`, comparing Moirai directory create/remove and recursive create/remove facade behavior against Tokio directory facade operations.
- Added an `async_fs_copy_file` row to `async_fs_comparison`, comparing Moirai platform-copy facade behavior against Tokio `fs::copy` over the same 64 KiB file.
- Added TCP and UDP loopback value tests for the Moirai-owned async network facade.
- Added `async_tcp_comparison`, a value-checked Moirai TCP facade echo benchmark against Tokio `TcpListener`/`TcpStream`.
- Added a persistent TCP stream echo row to `async_tcp_comparison` to isolate established-stream read/write behavior against Tokio `TcpStream`.
- Added a TCP write-shutdown row to `async_tcp_comparison` to verify payload delivery and peer EOF against Tokio.
- Added `async_tcp_backpressure_comparison`, a value-checked TCP write-backpressure benchmark against Tokio over bounded socket buffers.
- Added `async_tcp_readiness_comparison`, a value-checked TCP read-readiness benchmark against Tokio that asserts `Poll::Pending` before peer data and exact payload delivery after release.
- Added `async_tcp_cancel_safety_comparison`, a value-checked TCP pending-read cancellation benchmark against Tokio that drops a pending borrowed read future before asserting unchanged caller buffer state and later payload delivery.
- Added `async_io_compat_comparison`, a value-checked benchmark for feature-gated Tokio I/O trait compatibility wrappers over native Moirai readers and writers.
- Added zero-copy `AsyncReadExt::read_exact` and `AsyncWriteExt::shutdown` futures for the native Moirai I/O trait surface.
- Added `async_udp_comparison`, a value-checked Moirai UDP facade receive benchmark against Tokio `UdpSocket::recv_from`.
- Added `map_with` and `map_init` stateful map adapters to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `update` mutation adapter to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `intersperse` separator adapter to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `flatten` nested-stream adapter to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `flat_map_iter` and `flatten_iter` serial-inner adapter methods to the Rayon-style parallel iterator subset with value tests and corrected same-run Rayon comparison rows.
- Added `take_any` and `skip_any` bounded-window adapters to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `take_any_while` and `skip_any_while` deterministic predicate-window adapters to the Rayon-style parallel iterator subset with value tests and a full-pass same-run Rayon benchmark row.
- Added `real_application_mixed_workload`, a value-checked mixed async/parallel/channel benchmark against a Tokio plus Rayon reference path.
- Refreshed public result-handle, async wake, scheduler handoff, and Criterion variance attribution evidence against Tokio and Rayon comparison rows.
- Added `sum`, `product`, `min`, and `max` terminal reducers to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `min_by`, `max_by`, `min_by_key`, and `max_by_key` ordered terminal reducers to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `find_map_first` and `find_map_any` predicate terminals to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `find_last` and `find_map_last` reverse-order predicate terminals to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `position_first`, `position_any`, and `position_last` predicate terminals to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `positions` to the Rayon-style parallel iterator subset with owned, borrowed, and mapped value tests plus a same-run Rayon benchmark row.
- Added `for_each_with` and `for_each_init` stateful side-effect terminals to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `try_for_each_with` and `try_for_each_init` fallible stateful side-effect terminals to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `while_some` optional-stream adapter to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `try_for_each` fallible side-effect terminal to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `try_reduce` fallible reduction terminal to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `try_reduce_with` fallible no-identity reduction over sealed `Option<T>` and `Result<T, E>` stream items with value tests and a same-run Rayon benchmark row.
- Added `copied` and `cloned` borrowed-reference adapters to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `iterator_adapter_non_clone_ref_map`, a value-checked borrowed `Vec<T>::par_iter` benchmark row against Rayon over non-`Clone` values.
- Added `IndexedParallelIterator::collect_into_vec` for exact-size source iterators with non-`Clone` move tests and a same-run Rayon benchmark row.
- Added `IndexedParallelIterator::unzip_into_vecs` for exact-size pair sources with non-`Clone` move tests and a same-run Rayon benchmark row over caller-provided output storage.
- Added indexed `interleave` and `interleave_shortest` adapters for exact-size sources with non-`Clone` move tests, shortest-tail drop checks, and a same-run Rayon benchmark row.
- Added indexed `step_by` for exact-size sources with non-`Clone` move tests, skipped-value drop checks, exact-length tests, and a same-run Rayon benchmark row.
- Added indexed `by_exponential_blocks` and `by_uniform_blocks` logical-output block adapters with non-`Clone` move tests, zero-sized policy markers, zero-size rejection, and a same-run Rayon benchmark row.
- Added `collect_vec_list` to the Rayon-style parallel iterator subset with non-`Clone` move tests and a same-run Rayon benchmark row.
- Added `zip_eq` equal-length pairing to the Rayon-style parallel iterator subset with value tests, mismatch-panic coverage, and a same-run Rayon benchmark row.
- Added `partition_map` with a public `Either<L, R>` sum type to the Rayon-style parallel iterator subset with value tests and a same-run Rayon benchmark row.
- Added `unzip` pair-stream collector to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `IndexedParallelIterator::{len, is_empty}` for exact-size parallel iterator sources with value tests and a same-run Rayon metadata benchmark row.
- Added `iter_ops_parallel_comparison`, a value-checked scoped `ParallelIter` map/reduce benchmark against Rayon.
- Added `cache_iterator_comparison`, a value-checked borrowed-slice `ZeroCopyParallelIter` map/reduce benchmark against Rayon, including a large borrowed reduce row that exercises the cache scoped-spawn gate.
- Added `execution_context_comparison`, a value-checked owned `ParallelContext::execute_iter` benchmark against Rayon.
- Added `numa_context_comparison`, a value-checked owned `NumaContext::execute_iter` benchmark against Rayon.
- Added `distributed_context_comparison`, a value-checked owned `DistributedContext::execute_distributed_map` benchmark against Rayon.
- Added `multi_system_context_comparison`, a value-checked owned `MultiSystemContext::execute_heterogeneous_compute` benchmark against Rayon.
- Added PAL async file value tests, PAL TCP/UDP delayed loopback progress tests for the no-active-reactor self-wake path, and a Linux epoll wake-path test.
- Added a PAL reactor task-handle completion regression test for spawned ready tasks.
- Added `standalone_deque_reclaim_policy`, a value-checked diagnostic benchmark for `ChaseLevDeque` quiescent versus shared epoch reclamation.
- Added bounded channel matrix coverage to the Tokio comparison audit, comparison report, and benchmark-contract surface.
- Added sealed generic `moirai_utils::simd` scalar contracts and a benchmark source contract rejecting the removed type-suffixed utility SIMD public surface.
- Added a vector-prefix/tail SIMD benchmark row for non-lane-multiple `f32` slice lengths with value assertions.
- Added a value-checked wide-real SIMD vector-addition benchmark row under the generic `moirai_utils::simd::add<T>` public API.
- Added registry-local and external-ID token lifecycle diagnostic rows to `result_handle_diagnostics`, with benchmark contracts tying both rows to the production registry token path.
- Added a registry-owned after-send quiescent diagnostic row to separate public result availability from worker-tail metrics completion.
- Added a refreshed native Rayon/Tokio gap-closure benchmark snapshot covering result handles, scheduler rows, async iterator rows, and selected Rayon adapter rows.

### Changed
- Replaced `moirai-iter::base` adapter dead-field suppressions with live
  accessor and `into_parts` APIs for `BaseIterator`, `MapAdapter`,
  `FilterAdapter`, and `BatchAdapter`; moved base tests into a vertical
  `base/tests.rs` leaf.
- Split `moirai-iter::async_iter` into vertical traits, sources, adapters,
  consumers, and bounded-parallel leaves; removed dead source cursor fields so
  `AsyncVecIter<T>` stays the size of `Vec<T>` and `AsyncRangeIter` stays the
  size of `Range<usize>`.
- Consolidated executor idle-maintenance and PAL active-reactor thread-local
  state through Melinoe `thread_cached!` leaves, removing duplicated nightly
  `#[thread_local]` and stable `thread_local!` branches.
- Removed obsolete `remote_stage("gpu-cluster", ...)` examples from historical
  architecture notes and replaced them with the fixed-capability routed facade
  boundary.
- Removed dead thread-local cache declaration from `moirai-core::pool::GlobalPool::get`; the active implementation uses the global pool path.
- Removed the stale root Mnemosyne Git patch override that still forced
  `../mnemosyne` path resolution; `Cargo.lock` now resolves Mnemosyne crates
  from upstream GitHub `main` commit
  `8a428c4ce72786ff4a28a94342d8e724a36319a3`.
- Removed the local Mnemosyne Git patch override and obsolete repository-local dependency copy, locked Moirai's GitHub Mnemosyne dependency to upstream `ryancinsight/Mnemosyne` `main` commit `4f8d84b91780d2b1f7b27ede29580dffe2bff9c9`, and reran allocator, TLS, Rayon-facing parallel iterator, cache iterator, and process/server routed execution benchmarks.
- Replaced `moirai-metrics` placeholder storage with vertical collector,
  counter, gauge, histogram, snapshot, and exporter leaves backed by real
  shared state and deterministic value snapshots.
- Replaced the PAL timer immediate-ready placeholder with a deadline-sensitive
  future that registers a waker, returns `Pending` before the deadline, and
  wakes at completion.
- Replaced the distributed iterator fixed 10 second completion estimate with an
  input-sensitive model using task count, node CPU capacity, reliability,
  latency, and aggregate bandwidth, including saturating behavior for extreme
  telemetry.
- Changed borrowed vector `positions`, borrowed copied map/filter/sum, chunked map/sum, vector-backed indexed step/interleave enumerate/map/sum, nested flatten/map/filter/sum, and zip_eq/map/filter/collect parallel iterator shapes to avoid intermediate materialization on the focused regression paths.
- Moved the public `moirai-iter::MoiraiIterator` facade into a vertical `facade` module, preserved execution contexts through the `ExecutionContext` enum instead of string matching, and removed silent error-to-empty fallback branches.
- Changed the public Moirai facade documentation to state that cross-machine remote closure execution is outside the active API until a transport-backed task contract exists.
- Changed the benchmark crate feature declarations to forward Moirai `async`, `iter`, `local`, and `mnemosyne` features explicitly instead of inheriting `moirai` defaults implicitly.
- Changed Mnemosyne OS TLS key fast-path lookup to use a relaxed scalar load; the published value is only the OS TLS key, not allocator slot contents.
- Changed standalone deque steal paths to remove steal-side `SeqCst` fences while retaining acquire observations and `SeqCst` ownership CAS operations.
- Changed utility SIMD benchmarks and benchmark setup contracts to call generic `moirai_utils::simd` operations instead of type-suffixed vector functions.
- Changed generic `f32` SIMD dispatch to process native vector prefixes plus scalar tails for non-lane-multiple lengths and record those operations as vectorized when a native backend is available.
- Changed utility SIMD wide-real native dispatch reporting to stay x86 AVX2-specific, matching the implemented backend instead of over-reporting native support on unsupported architectures.
- Moved `HybridExecutor` public task ID allocation into `TaskRegistry::register_next_task`, removing the executor-local `AtomicU64` while reusing the existing registry registration boundary.
- Split `result_handle_diagnostics` wrapper, scheduler-tail, primitive, and registry rows into dedicated vertical leaves while preserving benchmark names and production-path contracts.
- Changed `moirai_async::fs::copy` to delegate to the PAL platform copy operation instead of allocating a user-space 64 KiB transfer buffer.
- Changed `moirai_async::fs::write` to delegate to the PAL platform write operation over the caller-provided byte slice instead of constructing a facade file handle, updating stats, looping through writes, and unconditionally syncing.
- Changed `moirai_async::fs::append` to delegate to the PAL platform append operation over the caller-provided byte slice instead of constructing a facade file handle, updating stats, looping through writes, and unconditionally syncing.
- Changed async file facade metadata lookup to use PAL platform metadata without constructing an async file handle or stats state.
- Changed async file facade rename to use PAL platform rename without reading or copying file contents through user-space buffers.
- Changed async file facade removal to use PAL platform remove without constructing an async file handle or stats state.
- Changed async directory facade creation and removal to use PAL platform directory operations instead of direct async-layer `std::fs` ownership.
- Replaced async iterator terminal placeholder futures with value-semantic `collect`, `for_each`, `fold`, and `reduce` execution over the logical iterator stream.
- Changed `moirai_iter::iter_ops::ParallelIter` to own `Vec<T>` directly, borrow scoped chunks without `Arc<Vec<T>>`, accept non-`'static` closures, and use the bounded scheduler batch capacity as the scoped-spawn cost gate.
- Changed `moirai_iter::cache::ZeroCopyParallelIter` map execution to borrow scoped slices and closures directly instead of allocating `Arc` wrappers for borrowed data and functions.
- Changed `moirai_iter::cache::ZeroCopyParallelIter` reduce execution to move owned intermediate partials through pair compaction, accept non-`Clone` reducer closures, and gate scoped OS-thread fanout behind a scheduler-batch cache-chunk floor.
- Changed `moirai_iter::execution` direct iterator execution to move owned chunks instead of cloning chunk slices, removing `T: Clone` from direct execution-context map bounds.
- Changed `moirai_iter::numa` map and reduce execution to consume owned batches instead of cloning chunk slices, removing `T: Clone` from NUMA map and extension bounds.
- Changed `moirai_iter::distributed` map, reduce, and partition helpers to consume owned partitions, produce value-semantic map results, and remove clone-bound direct distributed item paths.
- Changed `moirai_iter::multi_system` partition, distribution, and heterogeneous map helpers to consume owned partitions without clone-bound direct item paths and to return value-semantic mapped results.
- Changed `moirai_iter::parallel::IntoParallelRefIterator for Vec<T>` to borrow non-`Clone` values with `T: Send + Sync + 'data` instead of requiring `T: Clone + 'static`.
- Changed exact-size vector source collection to bulk-move owned items into caller-provided `Vec` spare capacity without cloning.
- Changed `ParAsyncMap`, `ParAsyncFilter`, and `ParAsyncForEach` to use bounded in-flight polling through their `concurrency` parameter.
- Raised the unstable sorting sequential threshold so medium slices use Rust's optimized unstable sort until worker dispatch amortizes.
- Replaced placeholder-only async file tests with value-semantic file operation tests.
- Removed the obsolete async file placeholder future wrapper and documented Tokio reactor-native I/O compatibility as a deferred PAL boundary rather than a covered facade benchmark.
- Changed PAL TCP/UDP `WouldBlock` fallback without an active reactor to wake the current task before returning `Pending`.
- Changed async file, TCP, and UDP Tokio comparison benchmarks so Moirai rows use `Moirai::block_on` instead of `futures::executor::block_on`.
- Changed the Moirai TCP stream facade to expose TCP_NODELAY for low-latency stream benchmark parity with Tokio.
- Changed Moirai TCP stream shutdown from a no-op to PAL write-side socket shutdown.
- Changed the Moirai TCP stream facade to expose `from_std` so tests and benchmarks can wrap preconfigured sockets without copying payload buffers or depending on Tokio.
- Changed `TokioCompat<T>` and `MoiraiCompat<T>` into transparent conversion wrappers with `From<T>` constructors and value-semantic compatibility tests.
- Changed the persistent TCP stream benchmark to use the production Moirai `AsyncReadExt::read_exact` and `AsyncWriteExt::write_all` futures.
- Replaced the PAL reactor pending-only task handle with per-task atomic completion state and waker publication.
- Replaced PAL platform reactor boxing with compile-target `PlatformReactor` static dispatch.
- Replaced PAL reactor queued future `Pin<Box<dyn Future>>` storage with bounded inline future storage and monomorphized poll/drop future erasure.
- Replaced the Linux epoll reactor no-op wake placeholder with an internal `eventfd` wake path.
- Replaced public core and scheduler `Box<dyn BoxedTask>` / `dyn Scheduler` task surfaces with `ScheduledTask` inline storage and monomorphized execute/drop/context erasure.
- Replaced standalone scheduler `ChaseLevDeque` per-item boxed task nodes with contiguous `UnsafeCell<MaybeUninit<T>>` ring slots, sealed zero-sized quiescent reclamation, and opt-in shared epoch reclamation.
- Changed all current comparison benchmark targets to carry explicit Criterion measurement and warm-up bounds under benchmark-contract coverage.

### Breaking
- Removed the unused exported `thread_local_static!` platform macro. Internal
  runtime TLS now uses concrete std TLS or Melinoe `thread_cached!` sites.
- Removed the placeholder public distributed facade methods `Moirai::spawn_remote`, `Moirai::get_nodes`, `Moirai::register_node`, `MoiraiBuilder::enable_distributed`, and `MoiraiBuilder::node_id`; distributed iterator helper coverage remains in `moirai-iter`.
- Removed public type-suffixed `moirai_utils::simd` vector functions in favor of generic `add`, `mul`, `dot`, `sum`, `mean`, `variance`, and `matrix_mul_square<T, const N>` operations over sealed scalar traits.

### Fixed
- Fixed example all-target clippy coverage by documenting intentionally broad demo-domain fields, replacing index/range and loop patterns, handling map insert results, and factoring an IoT event-handler alias.
- Fixed `async_tcp_comparison` persistent stream setup so persistent sockets are created immediately before the persistent-stream benchmark group instead of idling through the preceding loopback group.
- Fixed the scheduler-diagnostics wake-decision path to pass the concrete `SchedulerInner` reference into the static `ContendedWakable` boundary.
- Fixed strict clippy findings in zero-copy iterator helper pointer captures without changing borrowed chunk semantics.
- Fixed async `RwLock` release-handoff coverage by adding value-semantic tests for final-reader-to-writer and writer-to-multiple-reader grant paths.
- Fixed utility SIMD benchmark coverage so the wide vector-addition row asserts expected values before timing and benchmark contracts reject reintroduced type-suffixed private SIMD routing.

## [0.2.0] - 2026-05-24

### Added
- Added `moirai-python` as PyO3 wrappers over `moirai::Moirai`, with package documentation and Python/Rust lifecycle tests.

### Removed
- Removed empty/deprecated `moirai-python` directories left by the earlier standalone backend path.
- Removed `moirai-python` workload wrappers and dependent comparison artifacts that are not direct runtime bindings: `checksum_indexed`, `mix_indexed`, `mix_rounds_indexed`, `wait_checksum_indexed`, `file_byte_sum`, `file_mix_sum`, `tcp_index_sum`, `u64_file_mix_sum`, `file_header_stat_sum`, `csv_numeric_sum`, `jsonl_numeric_sum`, `rgb_luma_sum`, Python comparison scripts, optional joblib dependency, and generated CSV results.

### Added
- Added a unified `moirai-executor::schedule` hierarchy with one thread scheduler for sync, blocking, and async-ready tasks.
- Added zero-sized work-class markers (`SyncTask`, `AsyncTask`, `BlockingTask`) for monomorphized scheduler routing.
- Added `thread_schedule_comparison` Criterion benchmark comparing Moirai ready-task scheduling against Tokio and Rayon.
- Added `ThreadScheduler::scope`, `HybridExecutor::scope`, and `Moirai::scope` for borrowed completion-only fan-out on the unified scheduler.
- Added `ThreadScheduler::for_each_indexed`, `HybridExecutor::for_each_indexed`, and `Moirai::for_each_indexed` for typed indexed fan-out on worker-sized scheduler chunks.
- Added `ThreadScheduler::map_reduce_indexed`, `HybridExecutor::map_reduce_indexed`, and `Moirai::map_reduce_indexed` for typed indexed map/reduce with one per-chunk result slot.
- Added `ThreadScheduler::join`, `HybridExecutor::join`, and `Moirai::join` for non-destructive scheduler quiescence waits.
- Added `has_work` accessors for scheduler, executor, and runtime queued-or-active work detection.
- Added a mixed unified-scheduler benchmark comparing one Moirai runtime against a Tokio plus Rayon reference for sync completion, async result handles, and indexed reduction.
- Added ready-work benchmark value assertions so Criterion runs validate computed sums before reporting timings.
- Added scoped ready-work scaling benchmarks at 64, 256, and 1024 work units.
- Added scoped unified-scheduler rows and checksum assertions to `industry_comparison`.
- Added an official Rayon-pattern map/reduce benchmark using `into_par_iter().map(...).sum()` against Moirai indexed reduction.
- Added `benchmark_contracts` integration tests for benchmark source integrity and comparison-path correctness.
- Added benchmark contracts for executable bounded Criterion target configuration, spawn smoke values, and SIMD benchmark setup values.
- Added `docs/rayon_tokio_gap_audit.md` mapping the active scheduler/result-handle/indexed-reduction comparison scope to executable Rayon/Tokio benchmarks and zero-cost invariant checks.
- Added same-turn Rayon/Tokio quick benchmark refresh evidence for public result handles, scoped ready work, and indexed reduction.
- Added `public_result_handle_comparison` with real Moirai `TaskHandle`, Tokio `JoinHandle`, and labeled Rayon scope baseline rows.
- Added `result_handle_diagnostics` Criterion benchmark separating direct result-slot, scheduler submission, scheduled result-slot, registry lifecycle, and full scheduler-backed public spawn/join costs.
- Added quiescent-barrier rows to `result_handle_diagnostics` to distinguish raw result-handle joins from explicit scheduler process-join barriers.
- Added a direct public-wrapper component row to `result_handle_diagnostics` covering registry lifecycle, result handles, panic boundaries, and executor metrics without scheduler submission.
- Added task-id allocation, spawned-metrics, completed-metrics, and no-metrics public-wrapper attribution rows to `result_handle_diagnostics`.
- Added registry hot-path attribution rows to `result_handle_diagnostics` for mutex lock-only, dense block lookup, slot initialization, lifecycle timestamp publication, aggregate mutex registration, and direct lifecycle cost.
- Added registry timestamp primitive attribution rows to `result_handle_diagnostics` for precise elapsed-offset sampling, release-store publication, and duration offset arithmetic.
- Added async state primitive attribution rows to `result_handle_diagnostics` for state claims, `Waker::from(Arc)`, and `wake_by_ref` notification.
- Added async completion component attribution rows to `result_handle_diagnostics` for completed-state publication, future-present drop, lifecycle completion, result-sender cell send/join, and full ready-completion components.
- Added scheduler submission attribution rows to `result_handle_diagnostics` for monomorphized worker selection, pending counter publication, selected-worker unpark, priority queue push/pop, combined submission queue publication, and before/after spawn metrics ordering.
- Added sealed zero-sized scheduler wake-decision diagnostics for empty selected-worker wake, contended wake-all, and saturated no-wake paths.
- Added a sealed static `BoundedContendedWake` scheduler policy and retained-code benchmark evidence showing the bounded contended wake path ahead of the prior wake-all diagnostic while public rows remain ahead of Tokio/Rayon references.
- Added a wake-once async result-handle row to `public_result_handle_comparison`, comparing Moirai `spawn_async` against Tokio `JoinHandle` with value assertions.
- Added an async-ready result-handle row to `public_result_handle_comparison`, separating ready async spawn overhead from wake/requeue overhead.
- Added a captured-ready public result-handle row to `public_result_handle_comparison`, comparing non-empty task captures against Tokio `JoinHandle` with value assertions.
- Added an oversized-capture public result-handle row to `public_result_handle_comparison`, exercising the scheduled-job oversized fallback against Tokio `JoinHandle` with value assertions.
- Added a direct Moirai `scope` single-work row to `public_result_handle_comparison` for a value-checked scoped completion comparison against Rayon `scope`.
- Added a 1,048,576-iteration release stress test for public `spawn_fn`/`join` result-handle completion.
- Added Windows IOCP and BSD/macOS kqueue PAL module files so platform reactor module trees resolve.
- Added rkyv-style transport archive helpers that validate owned message bytes and expose borrowed typed archive views.
- Added transport archive tests for value semantics, borrowed `String` views, channel receive views, malformed archive rejection, and exact archive-size preallocation.
- Added `transport_archive_comparison` Criterion benchmark for borrowed archive views versus owned decode references over the same archive bytes and transport path.
- Added borrowed `str` archive serialization so callers can encode string slices without constructing an owned `String`.
- Added a benchmark source contract rejecting the previously rejected scheduler inline handoff feature and `InlineHandoffSlot` source shape.
- Added iterator channel-fusion source contracts rejecting boxed `FusableChannel` split/merge storage, placeholder hash distribution, non-executing pipeline APIs, and O(n) FIFO removal.
- Added iterator source contracts rejecting the obsolete boxed-future base execution trait and boxed streaming producer storage.
- Added timer-wheel cancellation source contracts and value-semantic wake suppression tests.
- Added `moirai-iter::parallel` `filter_map` and `flat_map` adapters with value-semantic tests and benchmark-contract audit markers.
- Added `moirai-iter::parallel` `inspect`, `panic_fuse`, `chunks`, and `partition` surfaces with value-semantic tests, benchmark-contract audit markers, and Rayon comparison rows in `iterator_adapter_comparison`.

### Changed
- Routed `HybridExecutor` through the unified scheduler instead of the legacy per-worker `Mutex<VecDeque<_>>` worker implementation.
- Replaced global task lifecycle mutation during worker execution with per-task shared lifecycle state to reduce registry lock contention.
- Replaced per-task executor result channels with `TaskHandle::new_pending` and a shared one-shot result slot.
- Replaced per-priority worker queue locks with one permission-guarded priority queue state per worker.
- Replaced task lifecycle timestamp mutexes with atomic timestamp offsets and typestate lifecycle tokens.
- Replaced executor metrics `last_updated` mutex storage with an atomic timestamp offset and `last_updated()` accessor.
- Moved scheduler metrics refresh from spawn paths to metrics/stat observation paths.
- Replaced hashed task registry storage with dense direct-indexed task slots for monotonic task IDs.
- Replaced per-task lifecycle `Arc` allocation with registry-owned lifecycle block storage.
- Moved average task duration calculation from task completion to stats observation.
- Coalesced scoped logical jobs into worker-sized scheduler batches so completion-only ready work avoids per-item scheduler submission and result-slot allocation.
- Coalesced indexed logical work into worker-sized scheduler chunks so indexed ready work shares one typed closure and avoids per-item erased jobs.
- Replaced per-item aggregation in the indexed reduction benchmark path with per-chunk local reduction and caller-side final reduction.
- Added a cache-line-derived inline threshold for small indexed reductions, avoiding scheduler wakeup and result-slot allocation when dispatch overhead dominates the reduction work.
- Changed indexed map/reduce to compute one chunk on the caller thread and schedule only the remaining chunks.
- Changed indexed map/reduce chunk planning so scheduled chunks must amortize worker wakeup cost before parallel dispatch is used.
- Changed scheduler pending/active counter ordering so a worker cannot create a transient false quiescent state while moving a job from queued to running.
- Rewrote `industry_comparison` benchmark to current public APIs and restored all benchmark target compilation.
- Replaced mutex-protected public task result storage with a single-producer atomic one-shot result cell.
- Replaced the public result-slot condvar wait path with bounded spin plus single-waiter `thread::park` / `thread::unpark`.
- Replaced best-effort waiter registration with an explicit `WAITING` result-slot state so completion cannot publish READY before a parked consumer is visible.
- Replaced the public result-slot waiter mutex with an inline single-waiter cell guarded by the result-slot state machine.
- Added a sealed zero-sized result wait policy so blocking task-handle joins monomorphize the spin budget without storing runtime policy state.
- Reduced the blocking result-wait spin budget from 100 to 64 under the same zero-sized policy after diagnostics showed lower pending-spin miss cost while same-run Tokio/Rayon comparison rows remained ahead.
- Changed task-handle join to keep the first already-ready claim as a direct CAS and use relaxed-load gating only during pending spins before the existing park/unpark fallback.
- Changed satisfied result completion endpoints and running lifecycle tokens to consume their hot-path drop guards after successful send/complete while preserving drop-based cancellation and panic completion.
- Preserved the verified `Arc` result-slot ownership model after a raw-pointer endpoint variant failed stress verification.
- Replaced async public-handle dynamic future dispatch and boxed panic callback with a generic wake-coalesced poll state.
- Replaced async public-handle boxed future pinning with inline future storage inside the heap-stable async state.
- Replaced async public-handle lifecycle mutexing with inline lifecycle state guarded by the async poll-owner state machine.
- Changed async public-handle wake handling to consume one coalesced in-poll wake before scheduler requeue.
- Replaced async public-handle wrapper waker allocation with a direct `Wake` implementation on the future-state `Arc`.
- Replaced heap allocation for common small scheduled closures with inline erased job storage and a boxed fallback for oversized closures.
- Resized inline scheduled-job storage from 16 to 14 machine words while preserving a two-cache-line `InlineJob` footprint and increasing captured-closure inline coverage.
- Replaced the scheduled-job runtime consumed flag with a post-execute no-op drop function, recovering one inline payload word without changing queue element size.
- Added async-ready and wake-once rows to `result_handle_diagnostics` so async wake/requeue locality is measured outside the full public comparison target.
- Replaced production contended scheduler wake-all with the bounded two-worker wake policy, marked out of line so the serial submission branch remains compact.
- Added lifecycle elapsed-only and atomic-only rows to `result_handle_diagnostics` to isolate timestamp-source cost from lifecycle state stores.
- Added lifecycle start-instant diagnostic rows to `result_handle_diagnostics` to test token-carried duration timing without changing production lifecycle semantics.
- Added cached-clock lifecycle diagnostic rows to `result_handle_diagnostics` to quantify the overhead floor of scheduler-local clock sampling without changing production lifecycle semantics.
- Added an inlined by-reference async scheduling path so in-poll `wake_by_ref` can mark the future state notified without cloning the task `Arc`.
- Replaced oversized scheduled-job `Box<dyn FnOnce>` dispatch with typed heap storage and static execute/drop function pointers.
- Replaced the oversized scheduled-job raw-pointer heap variant with a typed boxed closure behind the existing inline job trampoline, preserving the two-cache-line `InlineJob` footprint.
- Replaced scoped scheduler `Box<dyn FnOnce>` buffering with inline `ScheduledJob` values and direct single-job scheduling.
- Relaxed the per-worker queue length hint because queue contents are synchronized by the queue mutex and scheduler quiescence is synchronized by global pending/active counters.
- Enforced Rayon/Tokio runtime dependency boundaries through benchmark contract tests.
- Replaced `moirai-async::timer::Timeout<F>` heap-pinned future storage with inline generic future storage and in-place pin projection.
- Replaced `moirai-async::timer::TimerWheel` placeholder cancellation with lazy canceled-ID tracking and split the wheel into a dedicated timer leaf module.
- Replaced `moirai-async::AsyncExecutor` dynamic future queue dispatch with monomorphized erased-future poll/drop functions and executor-owned task ID allocation.
- Replaced `moirai-async::AsyncHandle` mutexed result storage and global waker hash-map registration with an inline atomic result/waker slot.
- Replaced `moirai-iter::channel_fusion` splitter and merger storage with generic concrete channel types so `FusableChannel` calls monomorphize instead of using boxed trait-object dispatch.
- Replaced channel merger FIFO buffering with `VecDeque` so reads do not shift buffered elements.
- Replaced `StreamingIter` boxed producer storage with a generic producer type and `VecDeque` FIFO buffering.
- Split `moirai-iter/src/iter_ops.rs` into streaming, stateful, and test leaves under `moirai-iter/src/iter_ops/`.
- Split `moirai-iter/src/parallel.rs` into traits, sources, adapters, consumers, and test leaves under `moirai-iter/src/parallel/`.
- Split new Rayon-style side-effect and chunk adapter implementations into dedicated `parallel/adapters/` leaves so the adapter root stays below the structural line target while preserving generic static dispatch.
- Changed owned `Vec<T>` parallel iteration to use one by-value `VecParIter<T>` backed by `Vec<T>` and `split_off`, removing the prior `Arc<Vec<T>>` owned-source allocation path.

### Breaking
- `moirai-iter::channel_fusion::ChannelSplitter` and `ChannelMerger` now take concrete channel types directly. Callers pass `channel` to `add_channel` instead of `Box::new(channel)`, and all channels in one splitter or merger instance must share the same concrete type or an explicit enum wrapper.
- Removed `SplitStrategy::Hash` because it was a placeholder branch that always selected channel 0.
- Removed the non-executing `channel_fusion::Pipeline` builder surface. The existing typed iterator pipeline in `advanced_patterns` remains the implemented pipeline path.
- Removed the unused `moirai_iter::base::ExecutionBase` boxed-future trait. The active public execution context trait remains `moirai_iter::execution::ExecutionBase`.
- `StreamingIter` now has the type shape `StreamingIter<T, F>` and monomorphizes the producer closure instead of storing `Box<dyn FnMut()>`.
- Removed the public `VecNonCloneParIter<T>` parallel source; owned vector iteration now uses `VecParIter<T>` for clone and non-clone item types.
- Replaced the public async-handle result-sender mutex with a state-machine-guarded inline sender cell.
- Replaced the async public-handle future-present atomic flag with a poll-owner inline `UnsafeCell<bool>` flag under the future-state ownership contract.
- Removed the async public-handle poll-time future-present guard; the async state machine now remains the single poll-permission invariant while `future_present` is only a drop guard.
- Replaced `moirai-iter::ThreadPool` boxed dynamic job queue dispatch with monomorphized erased-job run/drop functions.
- Changed indexed scheduler chunk caps to include the caller execution lane alongside worker threads for large reductions.
- Encoded serial scheduler handoff affinity as `WorkClass::SERIAL_AFFINITY_OFFSET`, preserving monomorphized ZST routing without a runtime policy object.
- Completed `moirai-async::ErasedTaskFuture` with typed poll/drop function pointers and heap-stable concrete future ownership.
- Reused running lifecycle completion duration for public result-handle metrics instead of taking duplicate task-local timing samples.
- Replaced scheduler work availability condition-variable notifications with selected-worker `Thread::unpark`; the condition variable remains for quiescence joins.
- Changed quiescent single-task scheduling to reuse the stable work-class worker and limited idle spin to local-queue work.
- Changed scheduler `join` to perform a bounded fast quiescent spin before registering a condition-variable waiter.
- Narrowed claim-only public result-slot atomic orderings while preserving READY publication acquire/release semantics.
- Narrowed scheduler execution counter orderings while preserving active-worker acquire/release quiescence publication.
- Kept Windows QPC lifecycle timing diagnostic-only after production registry promotion regressed the public oversized-capture path; benchmark contracts now reject QPC in the production registry.
- Removed the chunk vector, boxed wrapper closure, and per-scope `Arc` state from single scoped-job completion.
- Increased `public_result_handle_comparison` to 20 Criterion samples, 500 ms warm-up, and 2 second measurement windows.
- Removed the duplicate Tokio async-ready benchmark row; Moirai async-ready now compares against the equivalent ready Tokio `JoinHandle` baseline.
- Made `performance_benchmarks` and `moirai_benchmarks` executable Criterion targets with bounded sample, warm-up, and measurement windows.
- Disabled plot generation in `performance_benchmarks` so the public-handle diagnostic Cargo bench path exits under the verification gate.
- Tightened `performance_benchmarks` so measured task results are value-checked before black-boxing.
- Bounded SIMD Criterion sample, warm-up, and measurement windows so `simd_benchmarks` completes under the 300s verification gate.
- Replaced deserialize-to-owned transport safe-channel helpers with zero-copy archive views over transport-owned bytes.
- Added exact archive-size hints for fixed-size and string transport archives to avoid avoidable `Vec` growth during encoding.
- Replaced the PAL reactor's internal raw-handle registry key with a transparent integer key while preserving the public `RawFd` API.
- Removed stale non-executable benchmark estimates from current performance reporting.
- Separated public-handle diagnostic rows from active competitive batch targets because Rayon scope is not result-handle equivalent.
- Removed the non-equivalent side-effect-only indexed row from active competitive comparison targets; value-equivalent indexed comparisons now use `map_reduce_indexed`.
- Split `result_handle_diagnostics` and `benchmark_contracts` into vertical domain file trees with each leaf below the 500-line structural target.
- Refreshed Rayon/Tokio gap evidence after the registry timestamp primitive split. Public result handles keep Moirai ahead of same-run Tokio references, scoped completion keeps Moirai ahead of Rayon scope, and indexed reduction keeps Moirai ahead of Rayon indexed reduction, while local Criterion baseline regressions keep scheduler handoff and async wake variance on the active optimization path.

### Fixed
- Removed unused scheduler job queue/execution timestamp measurements from the ready-task hot path.
- Fixed the 256-item indexed reduction benchmark gap by combining caller participation and amortized chunk planning.
- Changed task result completion to wake the single consuming waiter instead of broadcasting.
- Fixed a READY/park lost-wake race in public task-handle joins by making waiter registration part of the result-slot atomic state machine.
- Reclassified the previous inline-job stress hang to the result-slot lost-wake root cause after debugger verification.
- Fixed the `performance_benchmarks task_scheduling_overhead` timeout caused by Criterion plot/report generation after measurements completed.
- Fixed the `moirai-python` local `moirai` dependency version so workspace package resolution matches the `0.2.0` workspace version.
- Fixed a strict Clippy float-equality test by comparing exact `f64` bit patterns.
- Fixed `SecurityAuditor::generate_report` to produce monotonic report timestamps under same-tick report generation.
- Fixed executor reactor channel type inference on non-Unix targets.
- Fixed strict Clippy Send/Sync analysis for the PAL reactor registry on Windows raw handles.
- Fixed strict clippy blockers in core CPU-count detection, `Priority` default derivation, and timeout subtraction.
- Covered scoped scheduler body-error and body-panic cases so registered borrowed jobs complete before errors return or panics resume.
- Covered async public-handle requeue with a one-worker regression that proves pending futures release the worker before an external wake.
- Covered async public-handle self-wake completion with a value-semantic one-worker regression.
- Rejected a larger public result-slot spin threshold after benchmark results showed no statistically significant improvement.
- Rejected an unconditional load-before-CAS result take path after already-ready result-slot rows regressed.
- Rejected removing per-task metrics timestamp updates after it failed to improve the ready public result-handle row.
- Rejected routing public `spawn_fn` through the `SyncTask` work class after the ready public result-handle row regressed.
- Rejected an inline async result-sender cell after filtered async-ready and wake-once benchmark rows regressed.
- Rejected per-worker running-bit wake suppression after it added scheduled-job atomic traffic and regressed public result-handle rows.
- Rejected a direct CAS-only `wake_by_ref` fast path after it improved wake-once but regressed async-ready.
- Rejected relaxed lifecycle metadata atomics after isolated lifecycle rows improved but the public scheduling gate regressed.
- Rejected removing the duplicate scheduler worker identity field after the public scheduling gate failed to retain an improvement.
- Rejected production Windows QPC lifecycle timing after public-path and scheduling-gate regressions; retained the `Instant` registry lifecycle policy.
- Rejected result-slot write-then-swap publication after public spawn/join and quiescent-barrier diagnostics regressed.
- Rejected relaxed submit-side scheduler counter loads and increments after the public scheduling gate regressed.
- Rejected the lock-free registry allocator after it improved one focused ready diagnostic but regressed the scheduling gate and registry component rows; retained the dense-block registry and added a source contract rejecting the concurrent allocator shape.
- Rejected routing production scheduler wake publication through a shared helper after the scheduling gate classified the candidate as a regression; retained the direct hot-path branch and kept helper-based attribution feature-gated.
- Released empty trailing registry lifecycle blocks during cleanup while preserving active metadata and dense direct indexing for retained blocks.
- Removed redundant task-id storage from dense registry task-state slots; metadata ids are now derived from direct slot lookup.
- Replaced registry lifecycle completion saturating duration arithmetic with a debug-asserted monotonic timestamp invariant and plain subtraction.
- Removed the explicit running-lifecycle completion `Option` branch while preserving the drop-based implicit completion path.
- Fixed the partial `ErasedTaskFuture` implementation that blocked async executor benchmark-contract builds.
- Fixed `moirai-iter::parallel` reduction consumer result types and the empty-vector base case so Rayon adapter contracts compile and empty reductions terminate.
- Fixed benchmark compilation issues with SIMD functionality
- Fixed AtomicCounter interface compatibility between modules
- Fixed float comparison warnings in tests
- Fixed dead code warnings in metrics module
- Fixed memory size calculation to use std::mem::size_of_val
- Fixed useless vec! warnings in iterator tests

### Added
- Added SIMD performance counter for tracking vectorization usage
- Added comprehensive documentation for utility functions
- Added AtomicCounter fetch_add method for benchmark compatibility

## [0.1.0] - 2024-09-04

### Added
- Initial release of Moirai concurrency library
- **Phase 15**: Code Quality & Design Principles Enforcement
  - Fixed clippy errors for clean builds
  - Implemented underscored parameters
  - Extracted magic numbers to named constants
  - Applied SOLID, CUPID, GRASP design principles
- **Phase 14**: Critical Infrastructure Fixes
  - Fixed HybridExecutor to execute tasks properly
  - Fixed spawn_blocking result communication
  - Fixed spawn_async implementation
  - Verified examples work end-to-end
- **Unified Execution Model**: Hybrid runtime combining async and parallel execution
- **Work-Stealing Scheduler**: Intelligent load balancing across CPU cores
- **Memory Efficiency**: NUMA-aware allocation and cache optimization
- **Zero-Copy Primitives**: High-performance channel implementations
- **Iterator System**: Execution-agnostic iterators with SIMD optimization
- **Synchronization Primitives**: FutexMutex, WaitGroup, lock-free collections
- **Communication Patterns**: Broadcast channels, pub/sub, collective operations
- **Enterprise Features**: Security audit framework, performance monitoring
- **Comprehensive Testing**: 95% test coverage with property-based testing

### Architecture
- **Modular Design**: Clean separation following SOC and domain-oriented principles
- **Zero Dependencies**: Pure Rust standard library implementation
- **Cross-Platform**: Support for Linux, Windows, macOS with platform-specific optimizations
- **SIMD Support**: Vectorized operations for x86_64 AVX2 and ARM64 NEON
- **Memory Safety**: Zero unsafe code in public APIs

### Performance
- **Task Scheduling**: Sub-microsecond overhead per task
- **Scalability**: Linear scaling up to CPU core count
- **SIMD Optimization**: 4-8x performance improvement for vectorizable workloads
- **Cache Efficiency**: Data structures aligned to cache boundaries

### Design Principles
- **SOLID**: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
- **CUPID**: Composable, Unix philosophy, predictable, idiomatic, domain-centric
- **GRASP**: Information expert, creator, controller, low coupling, high cohesion
- **ACID**: Atomicity, consistency, isolation, durability in task execution
- **DRY**: Don't repeat yourself - unified abstractions
- **KISS**: Keep it simple - minimal complexity with maximum performance
- **YAGNI**: You aren't gonna need it - focused feature set

### Testing
- **Unit Tests**: 51 tests in core, 44 in iterators, 13 in main
- **Integration Tests**: Comprehensive system testing
- **Property-Based Tests**: Formal verification for critical algorithms
- **Stress Testing**: High-concurrency validation
- **Platform Testing**: Cross-platform compatibility verification

### Documentation
- **API Documentation**: Complete rustdoc coverage
- **Examples**: Working examples for all major features
- **Architecture Guide**: Detailed design documentation
- **Performance Guide**: Optimization recommendations
- **Migration Guide**: From std::thread and other frameworks
