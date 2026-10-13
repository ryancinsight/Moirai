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

[Version 0.3.0 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
