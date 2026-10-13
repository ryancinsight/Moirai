### Changed

- Pull-request workflows now skip every Rust, Python, and book job while a pull
  request is draft, then run the unchanged path-selected workloads when it is
  opened ready or marked ready for review. Default-branch, scheduled, and
  manual executions are unchanged.
- Bound Chase-Lev storage-generation and resize-owner waits to 64 processor
  hints before cooperatively yielding, without allocation or sleeping. Executor
  steal retries use Moirai's established 1,000-hint pre-yield ceiling while
  retaining the victim priority selected by the scheduler. Queue arithmetic,
  task ordering, and public APIs are unchanged. A rebuilt paired comparison
  satisfies the precommitted raw-median non-regression rule across two, four,
  and eight thieves; no throughput or latency improvement is claimed.
- Dropping the final external `ThreadScheduler` handle now drains and releases
  its worker pool. Worker-owned scheduler state no longer makes the automatic
  shutdown condition unreachable. Only non-worker callers enter the join
  election; scheduler workers close the blocking lane and return before joining
  any peer, while external callers remain synchronous. A worker-owned final
  handle releases scheduler state as the drained worker loops return.
  Completion is published under the waiter mutex so a concurrent external
  caller cannot miss the condition-variable notification.
  Compute admission now publishes pending work before its single shutdown
  observation, preventing workers from exiting between validation and queue
  publication. Cloned-handle and successful scheduling behavior remain
  unchanged.
- Re-publish each worker's idle bit before every park attempt. If a producer
  consumed the previous bit but another worker drained that task first, the
  re-parking worker remains visible to the next wake lottery instead of relying
  on the blind wake fallback.
- Bound cross-task inline wake polling under scheduler saturation to one nested
  level per thread. A deeper rejected wake now completes with typed
  `TaskError::ResourceExhausted`, matching the existing self-wake saturation
  contract instead of growing the waking thread's stack.
- Gate `result_handle_diagnostics` on its `registry-diagnostics` feature so
  default all-target verification does not select a benchmark whose diagnostic
  API is disabled. Feature-enabled benchmark behavior and workload are
  unchanged.
- Route the public CPU-bound indexed fan-out and reduction facade through the
  compute-worker pool instead of the dedicated blocking lane. Indexed-only
  runtimes no longer lazily construct blocking workers; retained four-worker
  measurements reduce `map_reduce_indexed` sample medians by 18.2-50.5% from
  4,096 through 65,536 elements without changing values, task counts, or the
  generic executor work-class seam.
- Cancelling the timer that determines the driver's current wait now wakes the
  driver immediately. Non-head cancellation retains the no-wakeup fast path,
  and heap compaction also wakes the driver when it may change the next
  deadline.
- Honor `ExecutorConfig::max_global_queue_size` at scheduler construction.
  The executor-wide external-admission bound is partitioned into power-of-two
  per-worker injectors without exceeding the configured total; configurations
  smaller than two slots per worker now return `InvalidConfiguration`, matching
  the queue sequence protocol's minimum valid ring size.
- Reduce the default resizable local-queue initial capacity from 256 to 128.
  Four 128-byte priority planes per worker now retain 1,572,864 direct bytes
  at 24 workers instead of 3,145,728, while the controlled warmed queue-kernel
  confidence interval overlaps the former policy. A cold 257-item burst pays
  one additional owner-only growth step; admission and exactly-once scheduling
  semantics are unchanged.
- **Breaking.** Replace the ineffective `max_local_queue_size` configuration
  field and builder methods with `local_queue_initial_capacity`. The value now
  reaches all four resizable priority-local Chase-Lev deques on every worker,
  normalizes to a supported power of two, and returns
  `InvalidLocalQueueInitialCapacity` before worker startup when normalization
  or the concrete slot layout is unrepresentable. `ThreadScheduler` callers
  replace `new_with_config` with `new_with_local_queue_initial_capacity`;
  direct deque callers construct `DequeCapacity::<T>` and pass the validated
  value to `ChaseLevDeque::new`. The scheduler's first const parameter now
  controls only bounded blocking-lane admission.
- Keep indexed completion state on the caller's stack. Warmed
  `ThreadScheduler::for_each_indexed` calls now allocate nothing, while
  `map_reduce_indexed` allocates only its result-slot buffer instead of adding
  reference-counted scope and slot ownership around it. Scoped scheduling and
  identity-clone unwinds now drain already-admitted borrowing jobs before their
  stack state is released.
- Distribute unhinted multi-batch scopes from one preselected base worker
  across the worker set. This prevents an earlier admission from changing
  selection state and routing every physical batch to one occupied lane during
  saturated nested execution. Explicit locality hints and single-job scope
  admission retain their existing behavior.
- Make the workspace packageable from a standalone checkout: internal
  benchmark and integration-test dependencies carry explicit version
  requirements, runtime examples live under the facade crate, and the PyO3
  binding and test-harness manifests carry complete package metadata. Update
  the routed-execution contract to track the versioned transport dependency.
- Replace facade-test sleeps and conditional result checks with runtime
  quiescence joins and value-semantic assertions.
- Thread the `numa_aware` facade builder setting through `ExecutorConfig` and
  the work-stealing scheduler. The `numa` feature now enables the core and
  executor seams together; the default remains topology-aware and an explicit
  `false` value skips NUMA assignment construction. This controls scheduler
  locality only and does not claim topology-directed memory placement.
- Harden the Chase-Lev deque's inline storage protocol for arbitrary `Send`
  values: slot generation claims now serialize owner reuse with thief reads,
  resize quiescence preserves those claims across buffer replacement, and
  batch steals use the same exactly-once arbitration as single steals without
  per-item heap nodes. Steal arbitration uses strong CAS operations so a single
  attempt reports only real contention rather than a spurious retry.
- **Breaking.** `Moirai::channel()` now returns a channel bounded at
  `moirai_core::channel::DEFAULT_CHANNEL_CAPACITY` (1024) instead of an
  unbounded one. A producer that outruns its consumer now blocks (or gets
  `ChannelError::Full` from `try_send`) rather than growing the queue until
  allocation fails. `Moirai::bounded_channel(capacity)` is unchanged.
- **Breaking.** `unbounded` is no longer re-exported from `moirai_core` or
  `moirai_core::prelude`. It remains available as
  `moirai_core::channel::unbounded`, documented with the backpressure it does
  not apply.
- **Breaking.** `moirai_core::channel::config::DEFAULT_RING_BUFFER_CAPACITY` is
  renamed `DEFAULT_CHANNEL_CAPACITY` and re-exported from `moirai_core`.
- `CacheAligned` now separates its payload by `DESTRUCTIVE_INTERFERENCE_SIZE`
  (128 bytes on x86-64/aarch64) rather than 64. This widens the MPMC ring's
  enqueue/dequeue positions, the SPSC ring's head/tail, `LockFreeQueue`'s
  head/tail, and the scheduler's counters. `TaskResultSlot`, `SpinLock` and
  `WorkerState` now obtain their separation from `CacheAligned`/`CachePad`
  instead of a hardcoded `#[repr(align(64))]`; `TaskResultSlot`'s hand-rolled
  `_pad: [u8; 63]` is gone.
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

[Version 0.6.1 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
