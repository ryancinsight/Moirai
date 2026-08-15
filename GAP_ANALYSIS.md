# Moirai vs. Leading Concurrency Libraries: Comprehensive Gap Analysis

## 2026-08-14 Interleaved execution tests use event synchronization

### Fixed

`tests/src/interleaved_execution_tests.rs` used `std::thread::sleep`,
wall-clock polling, and completion-time assertions in the rapid-switching,
cascade, burst-load, and error-handling cases. The error simulation predicate
was also unreachable inside its branch (`i % 10 == 0` under `i % 4 == 3`), so
the intended handled-error result was never produced.

The tests now retain task handles, join every task, and use bounded channel
receives only where an inter-stage completion event is part of the contract.
The burst case compares returned work against an independently computed exact
sum, and the error case asserts 15 successes plus 5 handled errors. Evidence:
`cargo fmt --all -- --check` and configured Nextest
`-p moirai-tests interleaved_execution_tests::` pass 6/6 on Windows.

## 2026-07-27 Unreproduced exactly-once failure in the Chase-Lev deque

### Open — watchpoint, not a confirmed defect

`moirai-scheduler::deque_concurrency chase_lev_exactly_once_high_thief_contention`
failed once, in a full `cargo nextest run --workspace` (770 tests, nextest run
ID `f20686ea-05f7-40bc-96f4-c4880e23dc8c`) on a host also running ~11 concurrent
compiles. FAIL at 0.538s against a 0.086s isolated baseline. The assertion text
was not captured.

Recorded because the test has no wall-clock or timing dependence — it asserts
exactly-once over 8 rounds of 50,000 items with 8 thieves — so a failure means a
genuine lost, duplicated, or out-of-range consume, not a slow machine.

Not reproduced by:

- ~75 executions of the `chase_lev_exactly_once` family at 5-way process
  concurrency;
- a full 770-test workspace run under the same conditions on merged `main`
  (`770 passed`).

Hypotheses examined and **rejected** by reading the implementation, so they are
not re-chased:

1. *Bounded ring overflowing at 1024 with 50,000 pushes.* No — `push` calls
   `resize`, which doubles.
2. *The resize path being unexercised on a quiet host and racing under load.* No
   — `chase_lev_exactly_once_small_capacity_forces_resize` covers it directly.
3. *Preemption widening the fence-free `pop` window on x86.* The optimization
   skips the `SeqCst` fence when `bottom - top >= MAX_BATCH_STEAL`. On TSO a
   plain load of `top` observes every completed steal, and a batch steal takes
   at most half the visible length, so no single steal reaches the popped slot.
   The residual window is the few instructions where the owner's `bottom` store
   is still in its store buffer — which a context switch *closes*, so preemption
   makes this path safer, not riskier.

Next method if it recurs: capture the assertion text (it distinguishes lost from
duplicated, which separates a transfer bug from a double-consume), and re-run
`tests/loom_chase_lev.rs` with the x86 fence-free branch forced on, since
`--cfg loom` on an x86 host already compiles that branch and would be the place
an unsound skip shows up.


## 2026-06-24 Lock-free / reactor concurrency re-audit

Paranoid re-audit of `moirai-sync` primitives, the `moirai-scheduler` work-stealing
deques, and the `moirai-async` reactor/waker via independent adversarial passes.

### Fixed
- `moirai-sync::FutexMutex` (Linux) lost-wakeup deadlock — slow-path acquired via
  `CAS 0 -> 1`, erasing the waiters marker; now `swap(2)` per Drepper/std. Real
  defect, Linux-only. Evidence tier: differential against the Rust-std `futex`
  mutex reference algorithm + traced interleaving; Windows fallback path
  value-verified; Linux build/run verification is CI-side (cannot cross-compile
  the workspace from the Windows dev host).
- `moirai-async::Notify` — `notify_waiters` destroyed a `notify_one` permit;
  fixed with a red->green regression test (`Mutex`-serialized, fully verified).

### Examined and ruled NOT a defect (paranoid false-positive control)
- `moirai-scheduler::chase_lev::pop` x86 fence-free fast path (`b - t >= 16`
  returns without the SeqCst fence): this is the **published Morrison-Afek
  bounded-TSO fence-free pop**; the 16-element margin is exactly what makes the
  relaxed `top` load safe against concurrent thieves, not a missing fence. Sound
  as written. (Minor consideration, not a defect: the `MAX_BATCH_STEAL = 16`
  margin must exceed the count of simultaneously-stealing workers; on a machine
  with >16 active thieves the fence-free bound would need revisiting.)
- `moirai-sync::{spin_lock, concurrent_hash_map, resource_pool, atomic_counter,
  wait_group}` and the non-Linux `FutexMutex` fallback: orderings and the
  SeqCst-fence store-buffer guards are correct; no real defect.
- `moirai-async::{handle, result_slot, rwlock, broadcast, waker}`: register-then-
  recheck and waker re-registration are correct on every `Pending` path; result
  publish/consume pairing (Release store / Acquire CAS) is sound.

## 2026-06-12 Unified scheduler architecture alignment audit

### Closed alignment findings
- `moirai-executor::schedule::route` is the current routing SSOT for thread,
  process, server, accelerator metadata, and per-process async-lane placement.
  Its implementation is split into vertical `policy`, `ids`, `topology`,
  `decision`, `summary`, and `router` leaves. Evidence tier: type-level/source
  audit, backed by value-checked `process_server_scheduler_routing` benchmark
  contracts.
- `moirai-transport` consumes route metadata through the `scheduler-routes`
  feature and binds server/process routes to bounded fixed-format remote task
  execution. Evidence tier: value-semantic tests and Criterion benchmarks for
  `process_server_routed_execution`.
- Mnemosyne integration is an allocator and owned-byte handoff boundary, not a
  cross-process or cross-device pointer-sharing claim. Evidence tier: source
  audit of `TransportPayload<R>` region markers, value-semantic payload tests,
  and the upstream Git dependency lock.
- `moirai-gpu::occupancy` owns the current accelerator-adjacent planning slice:
  topology-aware launch-shape and resident-block planning. Evidence tier:
  type-level API plus value-semantic tests over themis topology and Mnemosyne
  kernel resource budgets. The planner is available without `wgpu-backend`, so
  external Atlas GPU backends can consume launch planning without inheriting
  Moirai's concrete WGPU runtime dependency.
- `moirai-executor::schedule::route` now includes accelerator route metadata:
  `AcceleratorRoutePolicy`, `AcceleratorCounts`, `AcceleratorKind`, and
  `SchedulerRoute::Accelerator` cover CPU/GPU/TPU/NPU placement metadata through
  sealed ZST policy dispatch. The route module has been split into cohesive
  leaf files to keep policy, ID/topology, route-decision, summary, and router
  concerns separate. Evidence tier: type-level/source audit, value-semantic
  route tests, benchmark contracts, and value-checked
  `scheduler_route_accelerator_metadata_summary` Criterion rows.
- `moirai-transport::payload` now includes a sealed `DevicePayloadRegion` and
  `PayloadBoundary::Device`; process, server, and device regions reject pointer
  transfer while zero-copy `TransportPayload<R>::handoff` moves the same owned
  archive buffer. Evidence tier: type-level/source audit, value-semantic
  payload and route tests, benchmark contracts, and value-checked
  `device_region_owned_handoff` Criterion row.
- The top-level `Moirai` facade now exposes fixed-capability routed server and
  process execution through `execute_routed_server_task` and
  `execute_routed_process_task`. The public API admits only sealed
  `RemoteCapabilityToken<C>` plus matching `IntoRemoteOperation<C>` payloads.
  Evidence tier: type-level/source audit, value-semantic facade tests,
  benchmark contracts, and value-checked public routed execution benchmark rows.
- `moirai_core::communication::zero_copy` now uses vertical error, ring,
  channel, adaptive, and router leaves below the 500-line structural target
  while preserving the public re-export surface. Evidence tier: source audit,
  core crate tests, and benchmark-contract coverage.
- Executor idle-maintenance TLS and PAL active-reactor TLS now share the
  Melinoe `thread_cached!` abstraction instead of maintaining duplicated
  nightly/stable thread-local branches. Evidence tier: source audit plus
  feature-path clippy coverage for `moirai-executor` and `moirai-pal`.
- `moirai-iter::async_iter` now uses vertical traits, sources, adapters,
  consumers, and bounded-parallel leaves. Dead source cursor fields and the
  inherited module-wide dead-code suppression are removed. Evidence tier:
  source audit, structural size tests, benchmark-contract coverage, iterator
  nextest coverage, clippy, and refreshed async iterator Criterion rows.
- `moirai-iter::base` adapter wrappers no longer hide fields behind
  `#[allow(dead_code)]`; public accessors and consuming `into_parts` methods
  make each field part of the typed API, and base tests now live in a vertical
  `base/tests.rs` leaf. Evidence tier: source audit, value-semantic unit tests,
  benchmark-contract coverage, clippy, and refreshed iterator map/reduce
  Criterion rows.
- The root workspace no longer patches GitHub Mnemosyne dependencies to the
  sibling `../mnemosyne` checkout. `Cargo.lock` resolves the Mnemosyne crate set
  from `git+https://github.com/ryancinsight/Mnemosyne.git#8a428c4ce72786ff4a28a94342d8e724a36319a3`.
  Evidence tier: remote head check, lockfile source audit, focused build/tests,
  benchmark contracts, and same-run iterator plus routed execution benchmarks.
- `moirai-metrics` no longer contains placeholder storage or empty snapshot /
  export paths. The crate is split into collector, counter, gauge, histogram,
  snapshot, and exporter leaves; snapshots and Prometheus output are derived
  from real registered metric handles. Evidence tier: source audit,
  value-semantic unit tests, benchmark-contract coverage, clippy, and
  value-checked `metrics_collector_comparison` Criterion rows.
- `moirai-pal::timer::Timer` no longer completes before its deadline. It stores
  real completion/waker state, returns `Pending` before the deadline, wakes the
  registered task from a single sleeper thread, and completes immediately only
  for elapsed timers. Evidence tier: source audit, value-semantic unit tests,
  and benchmark-contract coverage.
- The retired `moirai-iter::distributed::DistributedIterator::execution_stats`
  path previously replaced a fixed 10 second completion estimate with
  input-sensitive telemetry. The current branch no longer exposes that
  iterator module; remaining lifecycle/stat accounting now lives in
  `result_handle_diagnostics`, where lifecycle-backed rows use
  registry-owned task IDs and contracts reject the removed external-ID
  helpers. Evidence tier: source audit and benchmark-contract coverage.

### Open alignment findings
- [x] README architecture drift: the public README still framed Moirai mostly as
  a Tokio/Rayon synthesis and contained old illustrative timing/GPU claims not
  tied to current executable evidence. Resolved in this documentation pass by
  describing Moirai as a unified scheduler/router and by limiting performance
  statements to current benchmark surfaces.
- [x] [arch] Accelerator route metadata: `SchedulerRoute` now has typed
  CPU/GPU/TPU/NPU placement metadata. This is not backend execution; it is the
  route-decision layer required before GPU/TPU/NPU co-scheduling.
- [x] [arch] Device-memory ownership gap: `TransportPayload<R>` now
  distinguishes thread/process/server/device payload regions. Device handoff
  moves owned archive bytes without cloning and rejects pointer transfer by
  type-level region constant; this is not accelerator backend execution.
- [ ] [arch] Accelerator backend consumption gap: no GPU/TPU/NPU backend consumes
  `SchedulerRoute::Accelerator` yet. The existing GPU slice plans occupancy only;
  co-scheduling requires a backend adapter with value and benchmark evidence.
- [x] [minor] Public facade gap: process/server route execution is now
  available through the top-level `Moirai` facade for fixed-format sealed
  capability tasks. Arbitrary remote closure execution and node discovery
  placeholders remain absent by construction.

## 2026-06-12 Stack architecture audit (cross-repo)

### Closed
- `moirai-iter` NUMA detection delegates to `themis::current_numa_node` (topology SSOT) instead of a local libc walk that returned 0 off-Linux.
- `moirai-iter::cache::CACHE_LINE_SIZE` re-exports the `moirai-core::constants` SSOT instead of a second copy of the value.
- themis 0.7.0 / mnemosyne 43a02f3 dependency bumps verified across the workspace (623 tests).

### Open findings — resolved 2026-06-12
- [x] `ipc.rs` dead-code audit: deleted `RdmaConnection`/`DistributedComm`/`CommBackend` (placeholder scaffolds returning `Unsupported`) and `GpuIpc`/`GpuMemHandle` (`create_handle` fabricated a zeroed handle as `Ok` — a mock; GPU IPC is hephaestus domain). `SharedMemory`/`SharedQueue` (real, tested) remain; the one surviving allow is the documented RAII keep-alive field.
- [x] IPC completion (follow-up): `SharedMemory` gained the real Windows half (`CreateFileMappingW`/`MapViewOfFile`/`UnmapViewOfFile`, raw extern decls — no windows-sys dep) and the module is gated `any(unix, windows)`, so the shared-memory queue runs on the primary dev platform; the round-trip tests execute on both OS families plus negative paths (missing segment, zero size). Module doc rewritten to the actual same-machine scope; the unconstructible `Unsupported` error variant is gone. Residual: the unix branch is compile-verified only in CI (Windows dev host).
- [x] Feature-flag finding was a false positive: `numa` gates executor builder/config code, `coroutine` gates lib modules, `result-diagnostics` is forwarded by benchmarks. No change.
- [x] `set_current_task`: the entire `CURRENT_TASK` TLS block had zero callers — `current_task_id` was a public API that could only return `None`. Deleted outright; worker identity lives in the scheduler runtime (`current_worker_id` via melinoe `thread_cached!`).

## 2026-05-22 Scheduler Audit Update

### Closed Gaps
- `HybridExecutor` no longer uses the legacy per-worker `Mutex<VecDeque<Box<dyn FnOnce()>>>` worker module.
- Sync, blocking, and async-ready tasks route through a single scheduler hierarchy with ZST work-class markers.
- Windows and BSD/macOS PAL reactor module trees now have concrete module files.
- Benchmark targets compile with `cargo bench -p moirai-benchmarks --no-run`.
- Public executor task handles no longer allocate a per-task `std::sync::mpsc` result channel.
- Worker queues use one permission-guarded priority queue state per worker instead of one mutex per priority.
- Task lifecycle mutation uses typestate tokens and atomic timestamp offsets instead of per-task timestamp mutexes.
- Scheduler job execution no longer records unused queue/execution timing.
- Scheduler metrics refresh moved from spawn to metrics/stat observation points.
- Task registry storage uses dense direct-indexed task slots instead of hashing monotonic task IDs.
- Average task duration is computed at stats observation instead of per task completion.
- Task result completion wakes one consuming waiter instead of broadcasting.
- Public task handles now use a single-producer atomic one-shot result cell instead of mutex-protected `Option<Result<_, _>>` storage.
- Public task handle waits now use a monomorphized zero-sized wait policy, a bounded load-gated pending spin, an explicit `WAITING` result state, and an inline single-waiter `thread::park` / `thread::unpark` cell, removing the prior condvar and waiter-mutex paths from result completion and preventing READY/park lost wakes.
- The public `spawn_fn`/`join` path passed delayed completion tests and 1,048,576-iteration debug and release stress tests with value assertions.
- Small scheduled closures now use 14-word inline erased storage while `InlineJob` remains two cache lines; oversized jobs allocate one typed `Box<F>` behind the same inline job trampoline instead of `Box<dyn FnOnce>` or a separate raw-pointer heap job variant.
- Scoped scheduler jobs now buffer inline `ScheduledJob` values instead of boxed `dyn FnOnce` closures. Single scoped jobs schedule directly, and scoped chunks execute buffered scheduled jobs without reintroducing a scoped dynamic-dispatch buffer.
- `moirai-async::timer::Timeout<F>` stores the wrapped future inline instead of heap-pinning `Pin<Box<F>>`, preserving monomorphized timeout composition for concrete future types.
- `moirai-async::timer::TimerWheel` now uses lazy canceled-ID tracking instead of placeholder cancellation, and tests prove canceled timers do not wake.
- `moirai-async::AsyncExecutor` queues `ErasedTaskFuture` values with monomorphized poll/drop function pointers instead of `Pin<Box<dyn Future<Output = ()>>>`; task IDs now come from executor-owned monotonic state instead of a per-spawn local atomic.
- `moirai-async::AsyncHandle` now uses an inline atomic `AsyncResultSlot<T>` with one result cell and one waker cell instead of `Arc<Mutex<Option<T>>>` plus a global `HashMap<TaskId, Waker>`.
- `moirai-iter` owns no thread pool: indexed fan-out goes through the unified scheduler and falls back to the caller's own thread, so one work-stealing runtime serves the whole stack (ADR-023).
- `moirai-iter::parallel` reduction consumers use `Reduction<T, F>` carriers for associative reduction state, find consumers return `Option<T>`, empty vector inputs terminate before recursive chunk splitting, and the touched parallel iterator implementation is split into vertical leaves.
- `moirai-iter::parallel` now covers `enumerate`, `zip`, `filter_map`, `flat_map`, `take`, `skip`, `chain`, `rev`, `inspect`, `panic_fuse`, `chunks`, and `partition` with value-semantic tests, and `ParallelSliceMut` covers the sorting slice-extension boundary with Rayon benchmark rows.
- `moirai-iter::channel_fusion` splitters and mergers store concrete `FusableChannel` implementations in `Vec<C>` instead of boxed channel trait objects. The placeholder hash branch and non-executing pipeline builder are removed.
- `moirai-iter::base` no longer exposes the unused boxed-future `ExecutionBase` trait, and `StreamingIter<T, F>` stores its producer as a concrete generic with `VecDeque` FIFO buffering.
- `moirai-utils::simd` now exposes generic sealed `SimdScalar`/`SimdReal` operations instead of public type-suffixed vector kernels. The SIMD source is split into a vertical module tree, benchmark call sites consume the generic API, and `simd_benchmarks -- vector_addition` measures generic addition overlapping the native-checked row while staying below scalar loops.
- PAL reactor `TaskHandle` futures now complete through per-task atomic completion state instead of returning `Pending` unconditionally.
- PAL reactor platform dispatch now uses compile-target `PlatformReactor` instead of `Box<dyn Reactor>`.
- PAL reactor queued futures now use bounded inline storage plus monomorphized poll/drop function pointers instead of `Pin<Box<dyn Future<Output = ()>>>`; oversized futures use typed boxed fallback.
- Public `moirai-core` and `moirai-scheduler` task scheduling surfaces now use `ScheduledTask` inline storage with monomorphized execute/drop/context functions instead of `Box<dyn BoxedTask>`, `dyn Scheduler`, or `TaskSlot`.
- Standalone `moirai-scheduler::ChaseLevDeque` now stores queued items in contiguous `UnsafeCell<MaybeUninit<T>>` ring slots instead of allocating one boxed node per pushed task. Retired-array reclamation is policy-parametric: default `QuiescentReclaim` has zero-sized state and guard, while opt-in `SharedEpochReclaim` carries one `AtomicUsize` active-access counter.
- `standalone_deque_reclaim_policy` now measures the forced-resize reclaim path and verifies the drained sum before timing. The latest run measures `QuiescentReclaim` at 2.5038-2.5309 us and `SharedEpochReclaim` at 6.8529-6.8897 us, keeping the shared epoch policy explicit instead of production-default.
- Async file and UDP Tokio comparison benchmarks now drive Moirai futures through `Moirai::block_on` rather than `futures::executor::block_on`.
- `moirai_async::fs::write` now delegates to `moirai_pal::fs::write`, which calls platform `std::fs::write` over `C: AsRef<[u8]>` and removes the convenience write path's facade handle construction, stats mutation, manual write loop, and unconditional `sync_all`.
- `moirai_async::fs::append` now delegates to `moirai_pal::fs::append`, which opens with platform append options and writes `C: AsRef<[u8]>` without constructing the facade handle, mutating facade stats, looping through facade writes, or unconditionally syncing.
- The public-handle Criterion diagnostic target disables plot generation and exits under the Cargo benchmark path. Post-QPC-revert `task_scheduling_overhead` measured 528.88-535.17 ns with the asserted result value.
- `public_result_handle_comparison` now measures real public result-handle paths with 20 samples, 500 ms warm-up, and 2 second measurement windows. The latest rerun measured Moirai ready, captured, oversized, async-ready, and wake-once rows at 520.30-524.09 ns, 522.57-527.70 ns, 725.88-731.07 ns, 753.72-772.60 ns, and 739.89-762.97 ns. Equivalent Tokio rows measured 1.5627-1.6873 us, 1.5896-1.6059 us, 1.6363-1.6575 us, and 1.9103-2.2233 us for ready, captured, oversized, and wake-once. Its scoped completion row measured Moirai `scope` at 523.35-532.37 ns versus Rayon `scope` at 631.72-647.93 ns.
- Async public tasks store futures inline in the heap-stable async state, consume one coalesced in-poll wake before scheduler requeue, build the waker directly from the future-state `Arc`, and use an inlined by-reference scheduler path for in-poll `wake_by_ref` notifications.
- Scheduler execution counters use release publication for active/pending handoff, relaxed completed/failed metric increments, and retain acquire/release synchronization on the active-worker decrement that can publish quiescence.
- Task lifecycle state now uses registry-owned block storage instead of a per-task lifecycle `Arc`.
- `Moirai::scope` now exposes borrowed completion-only fan-out through the unified scheduler.
- Scoped logical jobs are coalesced into worker-sized scheduler batches, reducing scheduler submission overhead from one physical job per work item to one physical job per worker batch.
- `Moirai::for_each_indexed` now exposes typed indexed fan-out through worker-sized chunks, reducing scheduler submission overhead from one physical job per logical item to at most one physical job per worker and sharing the typed closure once across chunks.
- `Moirai::map_reduce_indexed` now exposes typed indexed map/reduce through worker-sized chunks, reducing aggregation overhead to one initialized result slot per physical chunk followed by caller-side reduction.
- Indexed map/reduce now uses a cache-line-derived inline threshold for small reductions. For 4 workers and `usize` results, 64 ready items run on the caller thread instead of paying scheduler wakeup and per-chunk result-slot cost.
- Indexed map/reduce now participates on the caller thread for one chunk and uses an amortized chunk planner so scheduled chunks are used only when the work volume justifies worker wakeup.
- Indexed chunk caps now include the caller execution lane, so large reductions can use `worker_count` scheduled chunks plus one caller chunk without adding runtime policy state.
- `Moirai::join` now exposes a non-destructive quiescence barrier over the unified scheduler. It processes all queued and active work before returning while keeping worker threads alive for later submissions.
- Scheduler quiescence detection now treats queued and active counters as one invariant: a worker increments active work before removing the job from pending work, and completion notifies join waiters only when both counters are zero.
- `thread_schedule_comparison` now includes value assertions for all ready-work sums before results are black-boxed.
- `thread_schedule_comparison` now includes `ready_task_schedule/moirai_scope`, which measured 13.997-14.190 us for 256 ready work units versus Rayon scope at 77.572-79.220 us and Tokio ready spawn at 77.915-80.571 us in the latest rerun.
- `thread_schedule_comparison` now includes `indexed_reduce_schedule/moirai_indexed_reduce`, which measured 666.94-678.71 ns for 256 ready work units versus Rayon indexed at 3.3286-6.0594 us in the latest rerun.
- `thread_schedule_comparison` now includes `mixed_unified_schedule`, a value-checked mixed workload combining sync scoped completion, async result handles, and indexed reduction. The latest run measures Moirai's single runtime at 40.510-41.370 us versus Tokio plus Rayon at 50.147-56.014 us.
- `moirai-async::RwLock` release handoff now has value-semantic tests for final-reader-to-writer and writer-to-multiple-reader grant paths, keeping the mixed scheduler benchmark gate compiling under strict diagnostics.
- `indexed_reduce_scaling` shows `moirai_indexed_reduce` ahead of Rayon indexed at 64, 256, and 1024 items while preserving the same value assertion.
- `scoped_ready_scaling` shows Moirai scope ahead at 64, 256, and 1024 ready work units while preserving the same value assertion.
- `industry_comparison` now includes a scoped unified-scheduler ready-task row and checksum assertions. Moirai scope measured 13.641-14.512 µs for 100 ready work units, 62.649-63.627 µs for 1,000, and 487.03-519.19 µs for 10,000, ahead of the Tokio and Rayon rows in the same run.
- `industry_comparison` now includes an official Rayon-pattern map/reduce group using `into_par_iter().map(...).sum()`. Moirai indexed reduction measured 3.9433-4.0053 µs for 4,096 work items, 12.244-12.461 µs for 32,768, and 20.315-20.855 µs for 65,536, ahead of the Rayon row in the same run.
- `benchmark_contracts` verifies that active competitive batch comparison sources exclude non-equivalent public-handle diagnostic rows, retain value assertions, declare executable bounded Criterion targets, and compute the same ready-task, spawn-smoke, SIMD, and map/reduce values as their references.
- Current performance reporting removes stale non-executable estimates and reports only executable Criterion benchmark evidence.
- Criterion benchmark targets now set explicit sample, warm-up, and measurement windows and compile as executable `harness = false` benches.
- The PAL reactor registry now stores an internal transparent integer handle key, removing the strict Clippy `Arc` Send/Sync blocker without changing the public `RawFd` API.
- Transport safe-channel payloads now use rkyv-style archive bytes and borrowed typed views. `String` receive validates length and UTF-8, then returns `&str` over the received message buffer instead of allocating an owned `String`.
- `transport_archive_comparison` benchmarks the borrowed archive receive path against an owned-decode reference over the same bytes. Borrowed archive view measured 15.913-16.095 ns versus owned decode at 32.097-32.415 ns, and the full transport round trip measured 233.63-237.09 ns versus 259.54-261.53 ns for raw transport plus owned decode.
- `docs/rayon_tokio_gap_audit.md` now records the active scheduler/result-handle/indexed-reduction comparison scope and maps each accepted Rayon/Tokio reference path to executable benchmark targets and `benchmark_contracts` checks.
- `benchmark_contracts` now verifies sealed ZST work-class routing, monomorphized scheduler calls, `WorkClass::SERIAL_AFFINITY_OFFSET` serial handoff selection, inline scoped job buffering, and the absence of Rayon/Tokio from runtime `[dependencies]` sections.
- `benchmark_contracts` now rejects heap-pinned timeout future storage in the generic timeout combinator.
- `benchmark_contracts` now rejects dynamic async-executor future queue dispatch and the local-atomic task ID allocation pattern.
- `benchmark_contracts` now rejects boxed dynamic iterator thread-pool job queues.
- `result_handle_diagnostics` and `benchmark_contracts` now use vertical domain file trees; each benchmark and contract leaf remains below the 500-line structural target while preserving the same executable targets.
- `result_handle_diagnostics` now separates registry lock acquisition, dense block lookup, slot initialization, lifecycle timestamp publication, aggregate mutex registration, and full direct lifecycle rows before any further registry rewrite.
- `result_handle_diagnostics` now separates scheduler worker selection, pending counter publication, selected-worker unpark, priority queue push/pop, combined submission queue publication, and before/after spawn metrics ordering with value assertions before any further scheduler publication change.
- Production contended scheduler submissions now use the sealed `BoundedContendedWake` ZST policy instead of wake-all. The retained path wakes the selected queue owner and one deterministic peer without allocation, dynamic dispatch, or new submission-side atomics. Focused diagnostics measured 162.41-180.11 ns versus the prior 404.11-409.07 ns wake-all branch, while same-run public rows kept Moirai ahead of Tokio/Rayon references.
- `AsyncFutureState::future_present` now uses a poll-owner `UnsafeCell<bool>` drop guard instead of atomic synchronization, and `result_handle_diagnostics` separates async completed-state store, future-present drop, lifecycle completion, sender-cell send/join, and full ready-completion component cost. `AsyncFutureState::poll` no longer checks that flag because the state machine is the authoritative poll-permission guard.
- `moirai-python` now provides PyO3 wrappers over `moirai::Moirai`; the package removed standalone scheduler, planner, backend logic, workload kernels, Python comparison scripts, optional joblib dependency, generated CSV results, and empty/deprecated package trees.
- `moirai-python` exposes only the native runtime lifecycle facade. Benchmark-specific Python functions remain excluded unless they correspond to comparable joblib or Tokio runtime primitives.
- Formally designed and accepted [ADR-006](file:///d:/Moirai/docs/adr.md#adr-006-async-io-compatibility-and-tokio-trait-integration) (Async I/O Compatibility and Tokio Trait Integration) and [ADR-007](file:///d:/Moirai/docs/adr.md#adr-007-webassembly-browser-event-loop-integration) (WebAssembly Browser Event-Loop Integration), closing the deferred design gaps.
- `moirai-iter::MoiraiIterator` now lives in a vertical `facade` leaf, carries `ExecutionContext` directly across transforms instead of reconstructing contexts through string matching, and rejects hidden empty-output fallbacks on execution failure.
- `parallel_iterator_regression` adds focused multi-size Moirai/Rayon rows for parallel iterator throughput regression checks independent of the broader adapter suite, including borrowed copied reductions, chunked map/reduce, indexed step/interleave, partition/unzip, and position/find terminals.
- `moirai-executor::schedule::route` now defines concrete thread/process/server/async-lane route values, sealed zero-sized `RoutePolicy` markers, and `HybridRouter<P>` for monomorphized route decisions. `process_server_scheduler_routing` benchmarks value-checked route summaries for sync, async, and blocking work classes without fabricating OS process or server execution.
- `moirai-transport/scheduler-routes` now consumes `SchedulerRoute` values through `RouteAddressBook`, `RoutedArchivedSender<P>`, and `RoutedArchivedReceiver<P>`, binding route metadata to transport-owned archive bytes for local roundtrips and to remote endpoint metadata for known server routes. `TransportPayload<R>` now tags those archive bytes with sealed thread/process/server/device ownership regions.
- `NetworkTransport` now transfers remote payload bytes over a bounded length-prefixed TCP frame and `TransportManager` routes `Address::Remote` through that real byte path. Remote task envelopes/results use server-region owned payload frames.
- `RemoteTaskEnvelope` and `RemoteTaskResult` now provide fixed-format request/response archives over the remote byte transport for explicit `EchoBytes` and `SumU64` operations. `RemoteCapabilityToken<C>` adds a sealed zero-sized capability boundary for building only admitted fixed-format operations, and `TransportPayload<R>` defines the allocator ownership handoff.
- `RoutedRemoteTaskClient<P>` now binds selected `SchedulerRoute::Server` values to fixed-format remote task execution through `RouteAddressBook` and `RemoteTaskClient`. Arbitrary closure remoting is rejected by the fixed-format capability boundary; process/server/device payload pointer transfer is rejected by region constants.
- `moirai-transport::process` now provides real OS process lifecycle primitives: `ProcessSupervisor` spawns child processes from `ProcessSpec`, observes blocking and bounded wait status, terminates live children, and applies explicit drop cleanup policy.
- `RoutedProcessTaskClient<P>` now binds selected `SchedulerRoute::Process` values to supervised child process execution through registered `ProcessEndpoint` entries and fixed-format remote task request/response. This is explicit built-in task execution only; arbitrary closure remoting is outside the admitted capability set.
- `BoundedRemoteTaskServer` now owns one listener lifecycle and admits fixed-format request frames through bounded `sync_channel` capacity and a bounded worker set. This closes bounded server execution for explicit built-in remote tasks only; arbitrary closure remoting is rejected by sealed capability construction.
- `process_server_routed_execution` now benchmarks selected server-route and process-route fixed-format `SumU64` execution end to end. The server row uses a real `RemoteTaskServer` thread and TCP request/result frames. The process row launches the benchmark binary in child-server mode through `ProcessSupervisor`, sends the fixed-format task through the selected process route, waits for child completion, and asserts the returned value and process status.
- Moirai now resolves Mnemosyne through the upstream Git dependency instead of
  the local patch override. `Cargo.lock` pins `mnemosyne`, `mnemosyne-core`,
  `mnemosyne-arena`, `mnemosyne-backend`, `mnemosyne-local`,
  `mnemosyne-decay`, `mnemosyne-hardened`, and `mnemosyne-prof` to
  `git+https://github.com/ryancinsight/Mnemosyne#8a428c4ce72786ff4a28a94342d8e724a36319a3`.
  The 2026-06-14 rerun compiled Mnemosyne-consuming crates against the GitHub
  source, passed focused route/payload/iterator tests and benchmark contracts,
  kept `iter_ops_parallel_comparison` map/reduce rows ahead of same-run Rayon,
  and measured real routed server/process execution through the fixed-format
  route benchmark.

### 2026-05-23 Active Scope Closure
- No active comparison gap remains in the scheduler/result-handle/indexed-reduction scope.
- Latest Criterion reruns keep the active comparison gap closed: Moirai result handles remain ahead of equivalent Tokio rows, Moirai scoped ready work remains ahead of Tokio and Rayon rows, and Moirai indexed reduction remains ahead of Rayon indexed reduction under the bounded benchmark filters.
- Accepted Tokio comparisons are public ready, captured-ready, oversized-captured, async-ready, wake-once async, and batch quiescence paths.
- Accepted Rayon comparisons are scoped completion-only fan-out, indexed map/reduce, and indexed worker chunking.
- Accepted mixed-engine comparison is sync scoped completion plus async result handles plus indexed reduction through one Moirai runtime versus Tokio plus Rayon over the same closed-form result.
- Full drop-in compatibility with every Tokio I/O type and every Rayon `ParallelIterator` adapter is an ecosystem extension goal, not an active scheduler comparison gap. The current audit covers Moirai-owned file and directory facade benchmarks against Tokio plus Moirai-owned TCP/UDP network facade value tests.
- Scoped dynamic-dispatch buffering is closed in the active scheduler scope. `thread_schedule_comparison -- scoped_ready_scaling` measures Moirai ahead of Rayon and Tokio at 64, 256, and 1024 ready work units after the change.
- Timeout future heap-pinning is closed in `moirai-async`. The public API remains unchanged.
- Async-executor dynamic future dispatch is closed in `moirai-async`. The queue remains a heterogeneous boundary, but dispatch uses monomorphized poll/drop functions rather than a `dyn Future` vtable.
- Iterator thread-pool dynamic job dispatch is closed in `moirai-iter`. The queue remains heterogeneous, but run/drop dispatch uses monomorphized function pointers instead of a `Box<dyn FnOnce>` vtable.
- Iterator channel-fusion dynamic channel dispatch is closed in `moirai-iter`. Split/merge routing now monomorphizes over one concrete channel type per instance, and source contracts reject boxed channel storage plus the removed placeholder pipeline/hash surface.
- Iterator streaming dynamic producer dispatch is closed in `moirai-iter`, and the obsolete boxed-future iterator base trait is removed. Source contracts reject both regressions.

### Remaining Gap
- Public result-bearing ready-task spawn/join is no longer behind Tokio's equivalent `JoinHandle` paths. Rayon remains non-equivalent for per-task result handles, so the benchmark also reports a direct Moirai `scope` versus Rayon `scope` completion-only row.
- Active competitive batch comparison targets keep public-handle rows separate from scoped and indexed batch rows so API semantics remain traceable.
- Root cause update: result-slot ownership is no longer the leading public spawn/join candidate. `result_handle_diagnostics` measured direct ready result-slot completion at 38.548-39.209 ns, direct same-thread send/join at 48.293-49.115 ns, direct scheduler submit/join at 336.87-348.66 ns, scheduled result-slot completion at 380.06-402.10 ns, registry lifecycle at 87.811-90.472 ns after post-QPC cleanup, mutex-only registry registration at 43.140-49.856 ns, and full Moirai public spawn/join at 552.31-560.74 ns. The latest public comparison target measured the same public API at 520.30-524.09 ns versus Tokio at 1.5627-1.6873 us and remains the authoritative Tokio/Rayon comparison surface. The latest public async rows measure Moirai async-ready at 753.72-772.60 ns and wake-once at 739.89-762.97 ns versus Tokio wake-once at 1.9103-2.2233 us. Lifecycle timestamp-source diagnostics show elapsed-time reads and atomic lifecycle stores are both measurable, while mutexed duration-only timing, token-carried start-instant timing, coarse cached-clock timing, and production QPC timing are rejected as production replacements. Atomic-only and cached-clock lifecycle timing are faster but remove or weaken duration observability; QPC remains diagnostic-only after production promotion regressed the public oversized-capture path. Remaining work should profile scheduler result-handoff variance and capture-storage shape without replacing lifecycle timing or adding locks. Oversized scheduled jobs still allocate, but no longer use `Box<dyn FnOnce>` dispatch.
- Quiescent-barrier update: forcing `Moirai::join` or `ThreadScheduler::join` after each result-handle join is rejected as a public hot-path strategy. The 2026-05-23 diagnostic measured direct scheduler result-slot completion at 380.06-402.10 ns without a barrier and 272.61-286.91 ns with one after fast scheduler-join spinning, while public `spawn_fn`/`join` measured 552.31-560.74 ns without a barrier and 667.67-681.32 ns with one. Process joining remains a batch-level API after producers finish submitting work.
- Public-wrapper update: `direct_public_wrapper_components` measured real public registry lifecycle, result handle, panic boundary, result publication, handle join, and metrics work at 201.56-205.32 ns without scheduler submission. The same diagnostic measured direct scheduler result-slot completion at 380.06-402.10 ns and public `spawn_fn`/`join` at 552.31-560.74 ns. Remaining variance is scheduler handoff plus capture storage.
- Public-wrapper attribution refresh: task-id allocation measures 6.1355-6.2125 ns, spawned metrics 28.634-29.053 ns, completed metrics 32.521-32.850 ns, wrapper without metrics 133.18-135.09 ns, full wrapper components 196.58-198.85 ns, registry lifecycle 86.249-87.135 ns, and mutex registry registration 44.510-45.247 ns. The retained public comparison measures Moirai ready result handles at 529.27-556.48 ns versus Tokio at 1.9803-2.1555 us and Moirai single scope at 525.82-538.29 ns versus Rayon at 697.25-714.03 ns. Result-slot swap publication and relaxed submit-side scheduler counters are rejected after public-path or scheduler-gate regressions.
- Lock-free registry allocator update: the tested concurrent block allocator is rejected. It improved the focused `moirai_spawn_join_ready` diagnostic to 459.61-487.90 ns, but `task_scheduling_overhead` regressed to 558.97-595.53 ns and registry component rows regressed or failed to improve. The production source retains dense `Vec<TaskStateBlock>` storage, boxed optional slots, and executor `Arc<Mutex<TaskRegistry>>` access; benchmark contracts reject `ConcurrentTaskRegistry` and the tested unique-registration allocator APIs.
- Registry hot-path split update: the focused component rows measure registry lock-only at 26.297-31.281 ns, dense block lookup at 40.774-52.762 ns, slot initialization at 108.28-133.47 ns, lifecycle timestamp publication at 161.60-177.75 ns, mutex registration at 91.007-103.90 ns, and direct lifecycle at 207.47-332.21 ns. Timestamp publication and slot initialization are the next registry targets; replacing the registry mutex alone is not supported by the measured cost split.
- Registry memory update: `cleanup_completed` now releases empty trailing lifecycle blocks after clearing completed slots, so an idle registry can return dense block storage without changing active task metadata or direct indexing for retained blocks. The retained scheduling gate measured 531.56-541.96 ns within the noise threshold.
- Dense registry state update: `TaskState` no longer stores a redundant task id because direct-indexed lookup already determines the task id. `TaskMetadata.id` is derived from the lookup id, preserving external metadata while reducing per-slot state. The current public comparison keeps Moirai ahead of Tokio on ready, captured, oversized, async-ready, and wake-once result handles, and ahead of Rayon on scoped completion and indexed reduction.
- Registry timestamp split update: feature-gated diagnostics measure precise `Instant` elapsed offset sampling at 24.645-24.783 ns, start release publication at 940.34-945.05 ps, completion release publication at 563.93-566.76 ps, and duration offset math at 449.67-453.51 ps. Existing-slot start and completion publication measure 25.159-25.406 ns and 27.402-27.507 ns respectively, so precise clock sampling is the next registry timing target rather than atomic stores or duration subtraction.
- Rayon/Tokio gap refresh update: the active comparison scope remains closed after the registry timestamp primitive split. Public result handles measure Moirai ready/captured/oversized/wake-once rows at 506.20-516.98 ns, 516.68-523.19 ns, 700.12-723.74 ns, and 756.79-761.38 ns versus equivalent Tokio rows at 1.6938-1.8250 us, 1.6755-1.7911 us, 1.6593-1.6871 us, and 1.7899-1.9801 us. Single scoped completion measures Moirai at 495.48-506.85 ns versus Rayon at 656.84-668.62 ns. Scheduler scope measures Moirai at 19.044-19.341 us versus Tokio at 89.273-90.520 us and Rayon at 80.283-81.728 us; indexed reduction measures Moirai at 714.22-729.27 ns versus Rayon at 7.7215-8.1235 us. Several Moirai rows regressed against prior local baselines, so scheduler handoff and async wake variance remain active optimization risks.
- Async sender-cell update: `AsyncFutureState` no longer stores its result sender behind a mutex; a state-machine-guarded `UnsafeCell<Option<TaskResultSender<_>>>` owns the one-shot completion endpoint. Async state primitive diagnostics measure state transitions and `wake_by_ref` in the 5.5297-6.3783 ns range and `Waker::from(Arc)` at 7.4358-8.3286 ns, so those primitives are not the dominant async public-handle cost. The post-change public comparison keeps Moirai ahead of Tokio/Rayon on the active gap scope, while async-ready still reports local baseline variance at 656.81-720.42 ns.
- Async future-present update: `future_present: AtomicBool` is removed from the async public-handle state and replaced by `UnsafeCell<bool>` under the poll-owner/drop-exclusivity contract. Corrected diagnostics measure the future-present drop flag at 191.60-194.35 ps, full ready-completion components at 150.12-151.23 ns, async-ready at 711.65-739.10 ns, and wake-once at 540.30-577.27 ns. Latest same-run public comparison remains ahead of Tokio on ready/captured/oversized/wake-once rows and ahead of Rayon on the scoped row, but default scheduling and scope rows show high local variance; scheduler handoff and async completion composition remain the next performance risks.
- Async poll guard update: `AsyncFutureState::poll` no longer reads the future-present drop guard before polling. Source contracts reject the removed `future_is_present` helper and poll guard. Focused diagnostics measure full ready-completion components at 148.04-148.58 ns, async-ready at 652.71-665.92 ns, wake-once at 551.11-579.84 ns, and the default scheduling gate at 540.37-550.84 ns. Same-run public comparison keeps Moirai ahead on equivalent Tokio and Rayon rows, while transient Criterion variance keeps scheduler handoff attribution as the next risk.
- Rejected candidate update: using `pending_tasks.fetch_add` as the worker-selection input removed one atomic load but regressed direct scheduler submit/join and public oversized, async-ready, wake-once, and scoped rows. Moving the freshly created async state `Arc` into initial scheduling removed one refcount pair but regressed wake-once. Both candidates are reverted, and benchmark contracts now reject restoring the old `scheduler-inline-handoff` feature and `InlineHandoffSlot` source shape.
- Scheduler submission diagnostic update: worker selection measures 1.1736-1.1792 ns, pending counter pair 9.6017-9.9314 ns, selected-worker unpark 27.731-28.763 ns, priority queue push/pop 59.064-59.332 ns, and combined submission queue publication 67.131-67.829 ns. Metrics-before submission did not beat the retained metrics-after ordering, which avoids overcounting failed submissions. The latest scheduling gate improved to 387.46-416.14 ns and public comparison remains ahead of Tokio/Rayon on all accepted equivalent rows.
- Scheduler wake decision update: feature-gated diagnostics now split empty selected-worker wake, contended wake-all, and saturated no-wake paths through sealed zero-sized markers. Empty wake measures 23.393-25.197 ns, contended wake-all measures 404.11-409.07 ns, and saturated no-wake measures 374.20-376.44 ps. A shared production wake helper is rejected after the first scheduling gate classified it as a regression; production keeps the direct wake branch. The next candidate should reduce contended wake-all frequency without adding serial-path atomics or moving pending publication before queue publication.
- Result wait spin-budget update: the sealed zero-sized `BlockingResultWait` policy now uses a 64-spin const budget instead of 100 before the single-waiter park fallback. The pending spin-miss diagnostic measures 626.15-640.32 ns versus the prior documented 100-spin miss at 1.1886-1.4520 us, and `task_scheduling_overhead` remains statistically unchanged at 533.78-555.30 ns. Same-run public rows remain ahead of Tokio/Rayon references, while captured, wake-once, oversized, and scope local Criterion regressions keep scheduler/result-publication variance as an active follow-up.
- Registry completion update: lifecycle completion now encodes the monotonic timestamp invariant with a debug assertion and plain subtraction instead of saturating arithmetic. The scheduling gate remains stable at 533.17-546.20 ns, and source contracts reject the saturating completion-duration path.
- Running lifecycle completion update: explicit completion now consumes the token and publishes completion directly instead of routing through the drop-path `Option` branch. The scheduling gate measured 534.64-549.65 ns with no regression, while the warm public comparison improved Moirai ready handles to 502.43-514.85 ns versus Tokio at 1.5021-1.5354 μs and improved Moirai scope to 479.32-493.46 ns versus Rayon at 661.60-671.01 ns.
- Scheduler queue update: per-worker queue length is now a relaxed advisory lock-skip counter; queue contents remain synchronized by the queue mutex and scheduler quiescence remains synchronized by global pending/active counters. The focused direct scheduler result-slot row improved to 328.03-335.01 ns, the scheduling gate improved to 538.01-545.54 ns, and the isolated public comparison kept Moirai ahead of Tokio ready handles and Rayon scoped completion.
- Oversized fallback update: replacing the separate typed heap job variant with a boxed inline trampoline measured `direct_scheduler_oversized_captured_result_slot` at 383.99-452.70 ns, down from the prior 853.63-957.61 ns diagnostic row. The same filtered run measured public oversized captured `spawn_fn`/`join` at 494.10-548.80 ns and direct `HybridExecutor::spawn_blocking` oversized captured at 543.14-579.37 ns. Remaining work should focus on public scheduler handoff variance.
- Rejected candidate: an inline async result-sender cell removed one mutex but regressed the filtered async-ready and wake-once rows, so it is not retained. A larger result-spin threshold produced no statistically significant improvement and is not retained. An unconditional load-before-CAS result take path regressed already-ready result slots and was replaced with a direct first CAS plus load-gated pending spins. Metrics-before-result publication regressed `result_handle_diagnostics/moirai_spawn_join_ready` to 581.34-586.56 ns and was reverted. Registry-owned task ID allocation regressed the same row to 628.34-641.23 ns; fresh-slot registry insertion regressed it to 683.31-768.95 ns. Both registry variants were reverted. A per-worker running-bit wake suppression variant improved some oversized diagnostics but added atomic traffic to every scheduled job and regressed public result-handle rows, so it is rejected.
- A raw-pointer two-endpoint result slot is rejected in the current architecture. Earlier variants reproduced a join hang; the latest endpoint variant passed targeted correctness checks but regressed `task_scheduling_overhead` to 633.01-640.02 ns, estimate 636.61 ns.
- The inline erased-job storage hang was reclassified after debugger evidence proved the parked join was a result-slot lost wake; inline storage is retained after the `WAITING` state fix passes stress and benchmark verification.
- The scoped completion-only path is not a substitute for public result-bearing task handles; it is the correct API when the caller needs a scoped reduction or side-effect barrier rather than per-task results.
- `Moirai::join` is a quiescence barrier, not a submission close. Work submitted after `join` observes quiescence belongs to a later batch and requires a later join or task-handle join.
- Latest Rayon-pattern example gap is closed: `example_pattern_comparison -- example_rayon_patterns` measured Moirai indexed reduction at 330.64-351.94 µs versus fixed-pool Rayon at 380.51-403.21 µs after indexed chunk caps included the caller execution lane. The refreshed official `industry_comparison -- official_rayon_map_reduce` row remains ahead for Moirai at 4,096, 32,768, and 65,536 work items.
- Iterator channel-fusion update: `ChannelSplitter<T, I, C>` and `ChannelMerger<T, C>` now use concrete channel storage and FIFO `VecDeque` buffering. `benchmark_contracts` rejects boxed channel storage, placeholder hash distribution, and non-executing pipeline APIs. The bounded channel matrix is now represented in the Tokio comparison matrix and contract coverage. The latest p1/c1 rerun measures Moirai MPMC at 1.4157-1.4504 ms versus Tokio MPSC at 2.5089-2.6101 ms for the same 8,192-item checksum workload.
- Async TCP facade update: `async_tcp_comparison` now covers same-payload TCP loopback accept/echo, persistent stream echo, and write shutdown against Tokio with `Moirai::block_on` for the Moirai rows; `async_tcp_backpressure_comparison` covers direct write-readiness observation against Tokio over bounded socket buffers; `async_tcp_readiness_comparison` covers direct read-readiness observation against Tokio before peer data is released; `async_tcp_cancel_safety_comparison` covers pending-read future cancellation safety against Tokio. The benchmarks assert the 24-byte request/echo, the 19-byte shutdown payload plus peer EOF, positive progress until write-side `Poll::Pending`, read-side `Poll::Pending` followed by the exact 5-byte payload, or dropped pending-read futures with unchanged caller buffers before timing. The latest accept/echo run measures Moirai at 294.02-354.85 µs versus Tokio at 323.75-365.72 µs. The persistent stream row isolates established-stream read/write with TCP_NODELAY and measures Moirai at 23.946-26.092 µs versus Tokio at 42.768-45.817 µs. The write-shutdown row measures Moirai at 26.185-34.695 ms versus Tokio at 21.158-27.122 ms. The write-backpressure row measures Moirai at 20.171-61.392 ms versus Tokio at 16.257-43.003 ms. The read-readiness row measures Moirai at 564.43-903.33 µs versus Tokio at 474.64-739.83 µs. The pending-read cancellation row measures Moirai at 299.08-340.01 µs versus Tokio at 339.36-368.55 µs.
- Async file facade update: `moirai_async::fs::write`, `moirai_async::fs::append`, `moirai_async::fs::metadata`, `moirai_async::fs::rename`, `moirai_async::fs::remove_file`, and `moirai_async::fs::copy` now delegate to PAL platform operations instead of constructing the higher-level facade for write/append/metadata/rename/remove or allocating a 64 KiB user-space transfer buffer for copy. `async_fs_comparison` now covers `async_fs_read_to_end`, `async_fs_write_file`, `async_fs_append_file`, `async_fs_metadata_file`, `async_fs_rename_file`, `async_fs_remove_file`, and `async_fs_copy_file` against Tokio with byte, length, file-type, destination-preservation, and path-removal assertions. The latest read row measures Moirai at 39.127-45.710 µs versus Tokio at 96.964-100.34 µs. The latest write row measures Moirai at 2.8650-3.4698 ms versus Tokio at 2.5939-3.2074 ms. The latest append row measures Moirai at 272.59-291.93 µs versus Tokio at 190.29-320.18 µs. The latest metadata row measures Moirai at 25.187-28.833 µs versus Tokio at 85.097-87.725 µs. The latest rename row measures Moirai at 603.37 µs-2.0949 ms versus Tokio at 3.5253-7.3040 ms. The latest remove row measures Moirai at 168.50-193.31 µs versus Tokio at 189.80-211.05 µs. The latest copy row measures Moirai at 536.26-604.18 µs versus Tokio at 541.41-716.30 µs.
- Async directory facade update: `moirai_async::fs::{create_dir, create_dir_all, remove_dir, remove_dir_all}` now delegate to PAL platform directory operations instead of calling `std::fs` directly from the async facade. `async_fs_dir_comparison` covers Tokio directory facade create/remove and recursive create/remove rows with directory-state and marker-file assertions. The latest single directory create/remove row measures Moirai at 251.56-276.02 µs versus Tokio at 443.78-503.63 µs. The latest recursive directory create/remove row measures Moirai at 2.9657-3.5428 ms versus Tokio at 4.7569-5.4030 ms.
- Async iterator adapter update: `AsyncIterator::enumerate` and `AsyncIterator::zip` now provide logical-position and shortest-input pair-stream semantics over the authoritative materialized stream. `async_iterator_enumerate_zip_pipeline` covers this path against a Tokio `JoinSet` reference over both inputs and measures Moirai at 672.68-734.62 µs versus Tokio at 48.260-49.144 ms after checksum equality.
- Native async I/O extension update: `AsyncReadExt::read_exact` and `AsyncWriteExt::shutdown` are now production extension futures over borrowed readers, writers, and caller buffers. Feature-gated `TokioCompat<T>` and `MoiraiCompat<T>` wrappers are transparent newtypes with `From<T>` constructors and value tests proving native Moirai readers/writers operate through Tokio traits and Tokio duplex streams operate through Moirai traits. The TCP persistent stream benchmark uses production `MoiraiAsyncReadExt::read_exact` and `MoiraiAsyncWriteExt::write_all` instead of local helper loops. The `async_io_compat_comparison` benchmark measures 4 KiB read compatibility at 2.5060-2.6553 µs for native Moirai versus 2.4962-2.6191 µs through `TokioCompat`, and write/shutdown at 179.85-191.55 ns for native Moirai versus 186.41-195.91 ns through `TokioCompat`.
- Iterator base/streaming update: the unused boxed-future `base::ExecutionBase` trait is removed, and `StreamingIter<T, F>` replaces boxed producer dispatch with a generic producer plus `VecDeque` FIFO buffering. The touched iterator operations tree is split into streaming, stateful, and test leaves under the 500-line target. The refreshed mixed scheduler benchmark still keeps Moirai ahead of Tokio plus Rayon at 44.023-44.699 µs versus 57.095-58.571 µs.
- Timer-wheel update: `TimerWheel::cancel` now returns value-sensitive cancellation results and suppresses canceled waker wakeups through lazy `HashSet<u64>` tracking. `timer.rs` is split below the 500-line structural target, and `example_pattern_comparison -- example_tokio_fanout` measured overlapping timer fanout intervals: Tokio at 15.356-15.597 ms and Moirai at 15.518-15.636 ms.
- Rayon adapter reduction update: reduce and reduce-with now return `Reduction<T, F>` internally so split halves combine with the supplied reduce function, find returns `Option<T>`, empty vector reductions terminate at the sequential base case instead of recursing, and the parallel adapter tree is split into traits, sources, adapters, consumers, and tests leaves below the structural target.
- Rayon adapter transform update: `enumerate`, `zip`, `filter_map`, `flat_map`, `take`, `skip`, `chain`, `rev`, `inspect`, `panic_fuse`, `chunks`, and `partition` are now value-tested covered subset adapters. The latest `iterator_adapter_comparison` run keeps Moirai ahead of same-run Rayon on every adapter row, including inspect/chunks at 31.061-31.810 µs versus 36.916-38.040 µs and partition at 29.242-30.103 µs versus 658.16-693.21 µs. The remaining unsupported Rayon comparison boundary is sorting, which belongs to a slice-extension trait rather than the current non-indexed `ParallelIterator` module.

## Executive Summary

This document provides a detailed gap analysis comparing Moirai's current implementation against leading concurrency libraries (Rayon, Tokio) to ensure feature completeness and competitive advantage in the unified concurrency architecture space.

## Current Moirai Strengths

### ✅ **Unified Architecture Excellence**
- **Hybrid Executor**: Seamless integration of async and parallel execution models
- **Work-Stealing Scheduler**: Chase-Lev deque implementation with multiple stealing strategies
- **GPU Compute Integration**: wgpu-rs based GPU acceleration with zero-copy principles
- **Unified Channel System**: Multiple channel types (SPSC, MPMC, unified channels)
- **Type-Safe Task System**: Comprehensive task abstractions with priority support

### ✅ **Performance & Safety**
- **Zero-Copy Principles**: Memory-efficient data transfers
- **Lock-Free Data Structures**: Chase-Lev deques, lock-free stacks
- **NUMA Awareness**: Architecture-aware scheduling and memory allocation
- **Memory Safety**: Rust ownership model prevents data races
- **Metrics & Monitoring**: Comprehensive performance tracking

## Ecosystem Extension Backlog

The sections below describe optional ecosystem expansion beyond the active scheduler comparison scope closed above. They are not unresolved gaps for the current unified-scheduler sprint.

## 1. **Parallel Iterator System (Rayon Ecosystem Extension)**

### Extension Candidates:
- **Parallel Iterator Trait** (`par_iter()` equivalent)
- **Collection Parallel Methods** (parallel map, reduce, filter, fold)
- **Range Parallel Processing** (`(0..n).into_par_iter()`)
- **Parallel Sorting** (`par_sort()`, `par_sort_by()`)
- **Join/Fork Patterns** (`join()`, `scope()` equivalents)
- **Custom Thread Pools** (isolated parallel execution contexts)

### Impact: HIGH
Rayon's parallel iterators are the gold standard for data parallelism in Rust.

### Recommended Implementation:
```rust
// Target API Design
use moirai::prelude::*;

vec![1, 2, 3, 4]
    .par_iter()
    .map(|x| x * 2)
    .filter(|&x| x > 4)
    .collect::<Vec<_>>();

// Custom thread pool
let pool = ThreadPoolBuilder::new()
    .num_threads(8)
    .build()?;

pool.install(|| {
    // Parallel work isolated to this pool
});
```

## 2. **Async I/O Ecosystem (Tokio Ecosystem Extension)**

### Extension Candidates:
- **TCP/UDP Networking** (async listeners, streams)
- **File System I/O** (async file operations)
- **Timer/Timeout System** (comprehensive timer wheel)
- **Signal Handling** (UNIX signal integration)
- **Process Spawning** (async process management)
- **Runtime Configuration** (current-thread vs multi-thread)

### Current Status:
- ✅ Basic async executor
- ✅ Basic timer system
- ✅ Moirai-owned file facade with value-semantic operation tests and a Tokio `fs::read` benchmark row
- ✅ Moirai-owned TCP/UDP network facade with loopback payload and counter tests
- ✅ PAL async file facade value tests, PAL TCP/UDP no-active-reactor self-wake progress tests, and Linux epoll eventfd wake path
- ✅ PAL reactor task handles publish completion for spawned ready tasks
- ✅ PAL reactor task queues use bounded inline future storage, monomorphized future dispatch, and static platform reactor dispatch
- ✅ Moirai async file and UDP comparison rows use the Moirai runtime surface
- ✅ Production reactor-native Tokio I/O compatibility designed and accepted under [ADR-006](file:///d:/Moirai/docs/adr.md#adr-006-async-io-compatibility-and-tokio-trait-integration).

### Impact: HIGH
Essential for async applications requiring I/O operations.

### Recommended Implementation:
```rust
// Enhanced async I/O API
use moirai::net::TcpListener;
use moirai::fs;
use moirai::time::{timeout, sleep};

// Robust networking
let listener = TcpListener::bind("127.0.0.1:8080").await?;
while let Ok((stream, addr)) = listener.accept().await {
    tokio::spawn(handle_connection(stream));
}

// Enhanced timer system
let result = timeout(Duration::from_secs(5), slow_operation()).await?;
```

## 3. **Advanced Synchronization Primitives**

### Extension Candidates:
- **Broadcast Channels** (one-to-many communication)
- **Watch Channels** (state watching with notifications)
- **Semaphore** (resource limiting primitive)
- **Async RwLock** (async-aware read-write locks)
- **Async Condvar** (async condition variables)
- **Notify** (efficient task waking mechanism)

### Current Status:
- ✅ SPSC/MPMC channels
- ✅ Unified channels
- ✅ Sync primitives (WaitGroup, SpinLock, FutexMutex)
- Async-aware synchronization expansion remains outside the active scheduler comparison scope.

### Impact: MEDIUM
Important for complex async coordination patterns.

## 4. **Tracing and Observability**

### Extension Candidates:
- **Structured Logging Integration** (tracing crate compatibility)
- **Distributed Tracing** (OpenTelemetry support)
- **Performance Profiling** (flame graph generation)
- **Runtime Introspection** (task monitoring, deadlock detection)
- **Metrics Export** (Prometheus/StatsD integration)

### Current Status:
- ✅ Basic metrics collection
- ✅ Performance counters
- Structured tracing and external monitoring remain outside the active scheduler comparison scope.

### Impact: MEDIUM
Critical for production monitoring and debugging.

## 5. **Ecosystem Integration**

### Extension Candidates:
- **Serde Integration** (serialization support)
- **HTTP Client/Server** (reqwest/hyper equivalents)
- **Database Drivers** (async database connectivity)
- **Message Queue Integration** (Kafka, RabbitMQ connectors)
- **Cloud Services** (AWS SDK integration)

### Impact: LOW
Can be implemented as separate crates or community contributions.

## 6. **WebAssembly Support**

### Extension Candidates:
- **WASM Runtime** (comprehensive WebAssembly support)
- **Browser Integration** (web worker compatibility)
- **JavaScript Interop** (efficient JS bridge)

### Current Status:
- ✅ Basic WASM executor stub
- ✅ Production WebAssembly browser event-loop integration designed and accepted under [ADR-007](file:///d:/Moirai/docs/adr.md#adr-007-webassembly-browser-event-loop-integration).

### Impact: MEDIUM
Important for web deployment scenarios.

## 7. **Testing and Development Tools**

### Extension Candidates:
- **Async Test Runtime** (test framework integration)
- **Deterministic Testing** (controlled scheduling for tests)
- **Fuzzing Support** (property-based testing integration)
- **Benchmark Suite** (comprehensive performance benchmarks)

### Current Status:
- ✅ Basic benchmarks
- ✅ Unit tests
- Advanced testing tools remain outside the active scheduler comparison scope.

### Impact: LOW
Development quality-of-life improvements.

## Implementation Priority Matrix

### **Priority 1: Ecosystem Extensions**
1. **Parallel Iterator System** - Essential for data parallelism
2. **Production Async I/O** - Essential for async applications
3. **Advanced Timers** - Critical for timeouts and scheduling

### **Priority 2: Developer Experience**
4. **Tracing Integration** - Important for debugging
5. **Advanced Sync Primitives** - Important for complex patterns
6. **Testing Tools** - Important for reliability

### **Priority 3: Ecosystem Expansion**
7. **WASM Support** - Good for web deployment
8. **External Integrations** - Community-driven features

## Unique Moirai Advantages

### **Differentiators Over Competition:**

1. **True Unified Architecture**
   - Single runtime for CPU + GPU + Async
   - Work-stealing across heterogeneous compute units
   - Zero-copy memory management

2. **GPU-First Design**
   - Native GPU compute integration
   - Automatic CPU-GPU load balancing
   - Memory-efficient GPU buffer management

3. **Performance Focus**
   - NUMA-aware scheduling
   - Cache-aligned data structures
   - Minimal abstraction overhead

4. **Rust-Native Design**
   - Zero-cost abstractions
   - Memory safety without GC
   - Compile-time optimization

## Recommended Implementation Strategy

### **Phase 1: Parallel Iterator Parity (4-6 weeks)**
- Implement core `ParallelIterator` trait
- Add collection parallel methods
- Integrate with existing work-stealing scheduler

### **Phase 2: Async I/O Foundation (6-8 weeks)**
- Robust TCP/UDP networking
- Production file system operations
- Enhanced timer system

### **Phase 3: Advanced Features (8-10 weeks)**
- Broadcast/watch channels
- Async synchronization primitives
- Tracing integration

### **Phase 4: Ecosystem Integration (Ongoing)**
- Community-driven feature additions
- External library integrations
- Performance optimizations

## Success Metrics

### **Technical Metrics:**
- Active scheduler comparison scope has executable Rayon/Tokio benchmark coverage.
- Competitive benchmark paths assert values before timing.
- Criterion targets use bounded sample, warm-up, and measurement windows.
- Drop-in Rayon iterator and Tokio reactor-native I/O API compatibility remains an ecosystem extension metric, not a 0.1.0 scheduler release gate.

### **Ecosystem Metrics:**
- Community crate adoption.
- Production deployments.
- Public repository adoption.
- Active contributor community.

## Conclusion

Moirai's unified architecture provides unique advantages in the concurrency landscape, particularly for heterogeneous compute workloads. The identified gaps are primarily in ecosystem completeness rather than fundamental architecture limitations. By addressing the Priority 1 gaps, Moirai can achieve feature parity with leading libraries while maintaining its architectural advantages.

The recommended implementation strategy focuses on high-impact, user-facing features that will drive adoption while leveraging Moirai's existing strengths in work-stealing and GPU integration.
