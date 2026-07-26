# Architecture Decision Record (ADR)

## ADR-001: Moirai as Complete Alternative to Tokio/Rayon

**Date**: 2024-12-19  
**Status**: Accepted  
**Context**: Moirai Concurrency Library Architecture  

### Decision

Moirai shall be implemented as a complete, standalone alternative to existing concurrency libraries (tokio, rayon, openmp, tbb) with native WebAssembly support, without runtime dependencies on the libraries it aims to replace.

### Context

The Rust ecosystem currently requires developers to combine multiple libraries for comprehensive concurrency:
- **Tokio**: Async runtime for I/O-bound operations
- **Rayon**: Data parallelism for CPU-bound operations  
- **OpenMP/TBB**: Traditional parallel computing patterns
- **Separate WASM solutions**: Limited cross-platform async support

This fragmentation leads to:
- Complex integration patterns
- Performance overhead from library boundaries
- Inconsistent APIs across paradigms
- Limited WebAssembly compatibility

### Decision Rationale

**Core Principle**: Moirai provides unified concurrency primitives that eliminate the need for external runtime dependencies, particularly tokio and rayon.

**Implementation Strategy**:
1. **Native Async Runtime**: Custom executor without tokio dependencies
2. **Unified API**: Single interface for async, parallel, and hybrid execution
3. **Zero-Cost Abstractions**: Compile-time optimizations that match or exceed alternatives
4. **WebAssembly First**: Native WASM support without platform-specific limitations

### Implementation Details

#### Allowed Tokio/Rayon Usage
- **Benchmarks Only**: Performance comparison testing (`benchmarks/` directory)
- **Examples Only**: Comparison demonstrations (`examples/moirai_vs_tokio_rayon_comparison.rs`)
- **Development Dependencies**: Testing infrastructure only

#### Prohibited Tokio/Rayon Usage
- **Runtime Dependencies**: No tokio/rayon in core library Cargo.toml dependencies
- **Implementation Dependencies**: Core modules must not import tokio/rayon for functionality
- **API Exposure**: Public APIs must not expose tokio/rayon types

#### Alternative Implementations Required
- **File I/O**: Native async file operations using platform syscalls
- **Network I/O**: Direct socket programming with epoll/kqueue/iocp integration  
- **Timer Systems**: Custom timer wheels and deadline management
- **Task Scheduling**: Work-stealing schedulers with custom executors

### Consequences

**Positive**:
- **Zero External Runtime Dependencies**: Simplified deployment and reduced attack surface
- **Unified Programming Model**: Single API for all concurrency patterns
- **WebAssembly Compatibility**: Full async support in WASM environments
- **Performance Control**: Direct optimization without library boundary overhead
- **Predictable Behavior**: No hidden runtime complexity or thread pool conflicts

**Negative**:
- **Development Complexity**: Requires implementing low-level async primitives
- **Platform Abstraction**: Must handle OS-specific async I/O mechanisms
- **Maintenance Burden**: Responsible for async runtime quality and performance
- **Ecosystem Integration**: May require adapters for tokio-based libraries

### Compliance Verification

**Build-Time Checks**:
```bash
# Verify no tokio runtime dependencies
cargo tree | grep -v "benchmarks\|examples" | grep tokio && echo "VIOLATION"

# Verify core modules clean
find moirai-* -name "Cargo.toml" -exec grep -l "tokio" {} \; | grep -v benchmarks
```

**Code Review Requirements**:
- All `use tokio::` imports must be in benchmarks or examples
- All `#[tokio::test]` must be replaced with native test harness
- All async function implementations must use Moirai primitives

### Related Decisions

- **ADR-002**: WASM-First Async Architecture
- **ADR-003**: Zero-Copy Communication Primitives  
- **ADR-004**: Hybrid Execution Model Design

### Implementation Status

- [x] **Phase 1**: Remove tokio dependencies from `moirai-async`
- [x] **Phase 2**: Implement native file I/O operations
- [x] **Phase 3**: Implement native network I/O operations
- [x] **Phase 4**: Replace tokio test infrastructure
- [ ] **Phase 5**: Validate performance parity with benchmarks

---

## ADR-002: WASM-First Async Architecture

**Date**: 2024-12-19  
**Status**: Accepted  
**Context**: WebAssembly Support Strategy

### Decision

Moirai's async architecture shall be designed with WebAssembly as a first-class target, ensuring full functionality in WASM environments without platform-specific dependencies.

### Context

Current async runtimes have limited or no WebAssembly support:
- **Tokio**: Minimal WASM compatibility, lacks I/O reactor
- **Rayon**: No WASM support for parallel execution
- **Platform Dependencies**: Most runtimes require OS-specific thread management

### Implementation Strategy

**WASM Compatibility Requirements**:
- No dependency on OS threads (use web workers or cooperative scheduling)
- Platform-agnostic I/O abstraction layer
- JavaScript interop for browser environments
- Node.js compatibility for server-side WASM

**Architecture Patterns**:
- Pluggable executor backends (native threads vs web workers)
- Async I/O via platform abstraction layer (PAL)
- Timer implementation using platform-appropriate mechanisms

### Consequences

**Positive**: Universal deployment, browser compatibility, server-side WASM support
**Negative**: Additional abstraction complexity, platform-specific testing requirements

---

## ADR-003: Zero-Copy Communication Primitives

**Date**: 2024-12-19  
**Status**: Accepted

### Decision

All inter-task communication shall prioritize zero-copy operations through shared memory, memory-mapped regions, and ownership transfer rather than serialization.

### Implementation

- Lock-free queues with ownership transfer
- Memory-mapped channels for large data
- Copy-on-write semantics for shared state
- NUMA-aware memory allocation

---

## ADR-004: Hybrid Execution Model

**Date**: 2024-12-19  
**Status**: Accepted

### Decision

Moirai provides a unified API that automatically selects optimal execution strategy (async, parallel, or hybrid) based on workload characteristics and system resources.

### Implementation

- Adaptive task scheduler with workload detection
- Automatic async/parallel task routing
- Resource-aware execution planning
- Single API surface for all execution patterns

---

## ADR-005: Unified Thread Scheduler for Executor Work Classes

**Date**: 2026-05-22
**Status**: Accepted
**Context**: HybridExecutor scheduler replacement

### Decision

`HybridExecutor` uses one scheduler facade for synchronous, blocking, and async-ready work. Sync and async-ready jobs use the compute worker set; `BlockingTask` uses the bounded lane defined by ADR-021. Workload shape is encoded with zero-sized marker types (`SyncTask`, `AsyncTask`, `BlockingTask`) implementing a sealed `WorkClass` trait. The scheduler stores heterogeneous jobs only at the executor queue boundary because closure output types are not homogeneous at runtime.

### Rationale

- One worker set removes the prior multi-engine split between worker queues and ad hoc async polling.
- ZST work-class markers keep dispatch monomorphized before heterogeneous queue storage.
- Per-worker priority queues preserve priority ordering without cloning scheduler algorithms per task class.
- Per-task lifecycle records remove global registry lock contention from task execution start/completion paths.
- Task handles use a shared one-shot result slot instead of per-task `std::sync::mpsc` result channels.
- The public task result slot uses a single-producer atomic state machine over one initialized result cell, so completed joins avoid mutex-protected result storage while preserving cancellation and panic values.
- Public result handles wait with bounded spin followed by an explicit `WAITING` result-state and one inline registered parked thread. Completion unparks the registered waiter only, preserving single-consumer semantics without condvar wait overhead, waiter-mutex overhead, or READY/park lost wakes.
- Small scheduled closures use inline erased storage sized to 14 machine words while keeping `InlineJob` at two cache lines. The consumed state is encoded by swapping the stored drop function to a no-op after execution, recovering one payload word without increasing the queue element footprint. Oversized closures allocate one typed `Box<F>` behind the same inline job trampoline instead of using `Box<dyn FnOnce>` or a separate raw-pointer heap job variant. The scheduler still has one heterogeneous erased boundary, but common small jobs avoid a heap allocation at that boundary without making every queue element carry a 16-word slot.
- Queue mutation is mediated by a `QueueAccess` permission guard over one priority queue state per worker, reducing lock count and making mutable access explicit.
- Lifecycle mutation is mediated by `TaskLifecycleToken` and `RunningTaskToken`; the registry owns observation, and the scheduled job owns mutation authority.
- Lifecycle and metrics timestamps use atomic offsets from their creation instant instead of mutex-protected `Instant` fields on hot paths.
- Lifecycle timestamp diagnostics keep task duration observability as a required contract; timing-policy changes must reduce timestamp-source cost without removing lifecycle completion or execution-duration metrics.
- Token-carried start-instant lifecycle timing is rejected for the current production policy because it regresses ready result availability and does not produce a workload-stable oversized improvement.
- Coarse cached-clock lifecycle timing is rejected for the current production policy because it weakens start/completion timestamp precision to a background update cadence, even though it establishes a lower overhead floor for future precise monotonic clock work.
- Windows QPC lifecycle timing remains diagnostic-only. It is precise and lock-free, but production promotion regressed the public oversized-capture path and an earlier scheduling gate; the production registry keeps the `Instant` policy and source contracts reject QPC imports, fields, and calls there.
- Task registry storage uses dense direct-indexed slots because executor task IDs are monotonic, eliminating per-task hash computation from registration and lookup.
- Task lifecycle state uses registry-owned fixed-size blocks, removing per-task lifecycle `Arc` allocation while keeping task-state addresses stable for running lifecycle tokens.
- Running lifecycle tokens carry their start offset and return execution duration on completion, so public result-handle metrics reuse lifecycle timing instead of sampling duplicate task-local clocks. Explicit completion consumes the token and publishes completion directly; the `Option` branch is retained only for drop-based implicit completion.
- Worker queue length is an advisory counter only. `WorkerQueues::state` synchronizes queued job storage, while global pending/active counters synchronize scheduler quiescence; therefore the queue length hint uses relaxed atomics and is not a memory-publication boundary.
- Completion metrics accumulate totals on the hot path and compute average duration only when stats are observed.
- Public-wrapper diagnostics now isolate task-id allocation, spawned metrics, completed metrics, registry/result handoff without metrics, and full public-wrapper components before any further hot-path registry rewrite. The tested lock-free registry allocator is rejected after a scheduling-gate regression, so the next optimization target is finer registry cost attribution and metrics tail cost rather than allocator replacement or result-slot publication.
- Scheduler workers use selected-worker `Thread::unpark` notifications for work availability. The global condition variable remains only for quiescence waits, avoiding submit-side condition-variable lock traffic on the spawn/join hot path.
- Quiescent single-task submissions route to a stable work-class worker through `WorkClass::SERIAL_AFFINITY_OFFSET`, and idle spin checks only the local queue so non-selected idle workers cannot steal freshly submitted serial work.
- Scheduler execution counters use release publication for active/pending handoff and relaxed completed/failed metric increments. The active-worker decrement keeps acquire/release ordering because it can publish quiescence to join waiters.
- Async public tasks store futures and lifecycle state inline in the heap-stable async state. The async poll state consumes one coalesced in-poll wake before falling back to scheduler requeue, and the task waker is built directly from the future-state `Arc` instead of allocating a wrapper waker. The by-reference wake path uses an inlined scheduler state transition so in-poll `wake_by_ref` can mark `ASYNC_NOTIFIED` without cloning the task `Arc`. The poll path uses the async state machine as the authoritative future-storage permission; the `future_present` flag is only a drop guard.
- The standalone `moirai-async::AsyncExecutor` queue owns `ErasedTaskFuture` values with monomorphized poll/drop function pointers instead of `dyn Future` dispatch. The concrete future allocation remains heap-stable so queued `!Unpin` futures can be polled safely after queue movement. `AsyncHandle` completion uses an inline atomic result/waker slot instead of mutexed result storage or a global waker hash map.
- `moirai-async::timer::Timeout<F>` stores `F` inline and projects it in place while pinned, avoiding heap-pinned generic future storage for timeout composition.
- `moirai-iter::ThreadPool` queues `ErasedThreadJob` values with monomorphized run/drop functions instead of boxed `dyn FnOnce` queue items. The queue remains heterogeneous at the ownership boundary, but it does not use a closure vtable to execute or drop jobs.
- The parking waker in `block_on_current_thread` replaces sleep-loop future polling for ready or properly waking futures.
- Borrowed completion-only fan-out uses `ThreadScheduler::scope`, exposed through `HybridExecutor::scope` and `Moirai::scope`.
- Scoped logical jobs are buffered as inline `ScheduledJob` values during the scope body and coalesced into worker-sized physical scheduler batches. This preserves the scope lifetime invariant while avoiding boxed `dyn FnOnce` scoped buffers, per-item result slots, and per-item scheduler submission when the caller only needs a completion barrier.
- Single scoped jobs use stack-owned scope state, direct scheduler closures, and no chunk vector or wrapper closure allocation.
- Typed indexed fan-out uses `ThreadScheduler::for_each_indexed`, exposed through `HybridExecutor::for_each_indexed` and `Moirai::for_each_indexed`.
- Indexed logical work is split into worker-sized chunks, sharing one typed closure across chunks and avoiding per-item erased scheduler jobs for bounded index domains.
- Typed indexed map/reduce uses `ThreadScheduler::map_reduce_indexed`, exposed through `HybridExecutor::map_reduce_indexed` and `Moirai::map_reduce_indexed`.
- Indexed map/reduce computes one local reduction per physical chunk, writes one initialized result slot per chunk, and performs final reduction on the caller thread after the scoped completion barrier.
- Indexed chunk caps include the caller execution lane: large indexed fan-out and reduction may create `worker_count + 1` chunks because the caller computes one chunk while worker threads execute scheduled chunks.
- Indexed map/reduce uses a cache-line-derived inline threshold for small reductions. Work at or below two result-value cache lines per worker executes synchronously on the caller thread, avoiding scheduler wakeup and result-slot allocation where dispatch overhead dominates the reduction.
- Indexed map/reduce computes one scheduled workload chunk on the caller and schedules only the remaining chunks. The reduction chunk planner requires enough work per scheduled chunk to amortize worker wakeup overhead.
- Non-destructive quiescence join uses `ThreadScheduler::join`, exposed through `HybridExecutor::join` and `Moirai::join`, to drain queued and active work without stopping worker threads.
- Quiescence is defined as `pending_tasks == 0 && active_workers == 0`. Workers increment active work before decrementing pending work, preventing a false empty state while a job moves from queued to running.
- Scheduler benchmarks assert value correctness for every ready-work path before reporting timing, preventing performance-only regressions that skip or corrupt work.
- Industry comparison benchmarks include the scoped unified-scheduler path directly, so the performance claim is not inferred from one isolated benchmark target.
- Industry comparison benchmarks include Rayon's documented parallel-iterator map/reduce pattern, `into_par_iter().map(...).sum()`, as a direct executable comparison.
- Mixed scheduler benchmarks combine sync scoped completion, async result handles, and indexed reduction, comparing one Moirai runtime against a Tokio plus Rayon reference with the same closed-form value assertion.
- The PAL reactor tracks platform handles with an internal transparent integer key, keeping public `RawFd` unchanged while satisfying Send/Sync analysis for shared reactor state.
- Transport safe-channel payloads use rkyv-style archive bytes plus borrowed typed views. The transport owns the byte buffer, and `String` receive returns `&str` over that buffer after length and UTF-8 validation, avoiding deserialize-to-owned allocation on the receive path.
- Transport archive benchmarking compares borrowed views and owned decode references over identical archive bytes and through the same `TransportManager` path. Send-side string archive encoding accepts borrowed `str` so callers do not need to construct an owned `String` before encoding.

---

## ADR-008: Scheduler Route Consumption and Transport Ownership Boundary

**Date**: 2026-06-02
**Status**: Accepted
**Context**: Process/server scheduler routing

### Decision

Scheduler route selection and transport execution are separate bounded contexts. `moirai-executor` owns `HybridRouter<P>` and concrete `SchedulerRoute` values. `moirai-transport` consumes those values behind the `scheduler-routes` feature, resolves them to concrete `Address` values, and sends archived payload bytes through the existing transport ownership boundary.

Route values are metadata until a transport backend consumes them. A route benchmark may measure route decisions and address resolution, but it must not claim OS process execution or server execution unless the corresponding process or network backend performs real work and returns value-checked results.

### Rationale

- The scheduler must remain independent of transport crates, process lifecycle APIs, and network backends.
- The transport crate already owns archive bytes and borrowed `ArchiveView` validation, making it the correct boundary for route-address consumption.
- Static `RoutePolicy` parameters keep route consumption monomorphized; no `dyn RoutePolicy` is introduced.
- Server route resolution can produce `RemoteAddress` metadata before a server transport exists, but sending over that route remains a transport backend responsibility.
- Mnemosyne allocator handoff is an owned-byte transfer contract, not cross-process or cross-device pointer sharing. Region markers specify whether pointer transfer is valid before a payload crosses a process, server, or device route.

### Implementation

- `moirai-transport` feature `scheduler-routes` optionally depends on `moirai-executor`.
- `RouteAddressBook` maps `SchedulerRoute::Thread` and `SchedulerRoute::Process` to local transport addresses and known `SchedulerRoute::Server` targets to remote addresses.
- `RoutedArchivedSender<P>` archives payloads through `ArchiveSerialize`, sends transport-owned bytes to the selected route address, and returns the selected route or address for value checks.
- `RoutedArchivedReceiver<P>` receives bytes from a route address and returns `ArchivedMessage<T>` so callers borrow validated views from the owned message buffer.
- Tests cover local archived route roundtrip, async process route async-lane address resolution, and server route remote endpoint resolution without sending.
- `NetworkTransport` sends and receives remote payload bytes through a blocking TCP length-prefixed frame with a fixed maximum message size.
- Remote byte transport is not remote task execution. Task envelopes, result envelopes, scheduler integration, and failure propagation remain separate contracts.
- Remote task envelopes/results are fixed-format archive contracts. Only explicit built-in operations are executable: `EchoBytes` returns the request payload and `SumU64` computes a wrapping sum without materializing the borrowed `u64` archive view.
- `RoutedRemoteTaskClient<P>` binds `SchedulerRoute::Server` selection to fixed-format remote task execution by resolving the selected route through `RouteAddressBook` and executing `RemoteTaskClient`.
- OS process lifecycle primitives use `ProcessSupervisor`, `ProcessSpec`, explicit `ProcessDropPolicy`, bounded wait polling, typed `ProcessOutcome`, and `ManagedProcess` drop cleanup around real `std::process::Child` handles.
- Process lifecycle is real OS process management. It is not process-routed task execution until a scheduler route, request envelope, child process protocol, and result return path are bound together.
- `RoutedProcessTaskClient<P>` binds selected `SchedulerRoute::Process` values to registered `ProcessEndpoint` entries, launches the configured child process, executes a fixed-format `RemoteTaskEnvelope` through that child's task server, waits under a bounded `ProcessWaitPolicy`, terminates non-exiting children, and returns the `RemoteTaskResult` plus typed `ProcessStatus`.
- `BoundedRemoteTaskServer` owns one `TcpListener` lifecycle for a bounded run, reads length-prefixed request frames, admits requests through a bounded `sync_channel`, executes them on a bounded worker set, and reports accepted/completed counts. This closes fixed-format server backpressure only; it does not make arbitrary Rust closures remotable.
- Arbitrary Rust closure remoting remains unsupported by design. `RemoteCapabilityToken<C>` is a sealed zero-sized capability boundary that admits only built-in fixed-format operation payloads and rejects closure or dynamic-task transport at the type surface.
- The top-level `Moirai` public facade exposes routed execution only through `FixedRemoteTask<C, P>`, `RoutedServerTarget`, and `RoutedProcessTarget`. `Moirai::execute_routed_server_task` and `Moirai::execute_routed_process_task` accept sealed `RemoteCapabilityToken<C>` values and matching `IntoRemoteOperation<C>` payloads, then delegate to the existing transport clients. No public API accepts arbitrary remote closures, dynamic remote task traits, or node discovery placeholders.
- `TransportPayload<R>` tags archive bytes with sealed thread, process, server, and device payload regions. `RoutedArchivedSender<P>` archives in the thread region, consumes the owned buffer into the process, server, or device region when the selected route crosses that boundary, and sends only owned bytes. `RemoteTaskClient` and `BoundedRemoteTaskServer` decode server-region frames into archive views owned by the receiver buffer. Process, server, and device regions set `POINTER_TRANSFER_ALLOWED` to `false`; the top-level `moirai` crate forwards Mnemosyne provider integration but never installs a process-global allocator from a library. Final binaries own allocator selection.

### Deferred Work

No deferred ADR-008 implementation work remains.

### Verification

- `cargo test -p moirai-core --all-features`: 61 passed.
- `cargo test -p moirai-executor --all-features`: 32 passed.
- `cargo test -p moirai --lib --all-features`: 18 passed.
- `cargo test -p moirai --lib test_repeated_public_spawn_join_completes -- --nocapture`: 1,048,576 public `spawn_fn`/`join` iterations passed with value assertions in 1.45s.
- `cargo test -p moirai --lib --release test_repeated_public_spawn_join_completes -- --nocapture`: 1,048,576 public `spawn_fn`/`join` iterations passed with value assertions in 1.39s.
- `cargo test -p moirai-pal --all-features`: 5 passed.
- `cargo clippy -p moirai-core --all-features -- -D warnings`: passed.
- `cargo clippy -p moirai-executor --all-features --tests -- -D warnings`: passed.
- `cargo clippy -p moirai --lib --all-features -- -D warnings`: passed.
- `cargo clippy -p moirai-pal --all-features -- -D warnings`: passed.
- `cargo clippy -p moirai-benchmarks --tests --benches -- -D warnings`: passed.
- `cargo test -p moirai-benchmarks --test benchmark_contracts`: 25 passed.
- `cargo test -p moirai-transport --all-features safe_channel -- --nocapture`: 4 passed.
- `cargo clippy -p moirai-transport --all-features -- -D warnings`: passed.
- `cargo bench -p moirai-benchmarks --bench transport_archive_comparison --verbose`: borrowed archive view 15.913-16.095 ns, owned decode reference 32.097-32.415 ns, archived transport round trip 233.63-237.09 ns, raw transport plus owned decode 259.54-261.53 ns.
- `cargo bench -p moirai-benchmarks --bench result_handle_diagnostics --verbose`: direct ready result slot 38.548-39.209 ns, direct same-thread send/join result slot 48.293-49.115 ns, direct scheduler submit/join 336.87-348.66 ns, scheduled result-slot completion 380.06-402.10 ns, direct public wrapper components 201.56-205.32 ns, direct registry lifecycle 87.811-90.472 ns after post-QPC cleanup, mutex-only registry registration 43.140-49.856 ns, Moirai public spawn/join 552.31-560.74 ns.
- `cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_task_id_allocate|direct_metrics_record_task_spawned|direct_metrics_record_task_completed|direct_public_wrapper_without_metrics|direct_public_wrapper_components|direct_registry_lifecycle|mutex_registry_register)"`: task-id allocation 6.1355-6.2125 ns, spawned metrics 28.634-29.053 ns, completed metrics 32.521-32.850 ns, public wrapper without metrics 133.18-135.09 ns, full public wrapper components 196.58-198.85 ns, registry lifecycle 86.249-87.135 ns, mutex registry registration 44.510-45.247 ns.
- Lock-free registry allocator A/B: candidate `moirai_spawn_join_ready` measured 459.61-487.90 ns, but `task_scheduling_overhead` measured 558.97-595.53 ns with Criterion reporting a regression; wrapper without metrics measured 154.49-159.21 ns, registry lifecycle measured 106.94-110.11 ns, and mutex registry registration measured 60.959-62.140 ns. The production source retains the dense-block registry and benchmark contracts reject the concurrent allocator shape.
- Registry hot-path attribution: `result_handle_diagnostics` now splits the retained dense registry path into lock-only 26.297-31.281 ns, block lookup 40.774-52.762 ns, slot initialization 108.28-133.47 ns, lifecycle timestamp publication 161.60-177.75 ns, mutex registration 91.007-103.90 ns, and direct lifecycle 207.47-332.21 ns. The next registry candidate targets timestamp publication and slot initialization before another lock replacement.
- Registry cleanup now releases empty trailing lifecycle blocks after completed slots are removed. Verification covers two completed blocks reclaimed to an empty registry, while the sequential scheduling gate remains within the noise threshold at 531.56-541.96 ns.
- Dense registry state layout: `TaskState` no longer stores a redundant task id because direct-indexed lookup already supplies the id for `TaskMetadata`. Focused verification preserves metadata ids and rejects reintroducing the field; the retained scheduling gate measured 612.29-627.91 ns with no statistically significant change.
- Registry completion duration uses a debug-asserted monotonic timestamp invariant and plain subtraction rather than saturating arithmetic. This keeps impossible clock-order violations visible in debug builds and removes defensive arithmetic from the release hot path; the retained scheduling gate measured 533.17-546.20 ns.
- Timestamp primitive attribution: precise elapsed-offset sampling measured 24.645-24.783 ns, while start release publication measured 940.34-945.05 ps, completion release publication measured 563.93-566.76 ps, and duration offset math measured 449.67-453.51 ps. The next timing candidate must reduce precise monotonic clock sampling without weakening timestamp precision.
- Rayon/Tokio gap refresh after timestamp split: same-run public result handles keep Moirai ahead of Tokio on ready, captured, oversized, and wake-once rows; scoped completion keeps Moirai ahead of Rayon; scheduler ready scope and indexed reduction keep Moirai ahead of Tokio/Rayon references. Local Criterion baselines still report Moirai regressions on several rows, so the next target remains scheduler handoff and async wake variance rather than broadening the active audit to non-equivalent APIs.
- Async sender-cell decision: `AsyncFutureState` result publication is owned by the state-machine-selected poll owner, so `result_sender` now uses `UnsafeCell<Option<TaskResultSender<_>>>` instead of `Mutex<Option<_>>`. Async primitive diagnostics show state transitions and `wake_by_ref` at 5.5297-6.3783 ns and `Waker::from(Arc)` at 7.4358-8.3286 ns; the next async target is lifecycle/result-publication composition, not dynamic dispatch or state-CAS cost.
- Async future-present decision: initialized-future ownership is also poll-owner scoped, and final `Drop` has exclusive access after the last `Arc` release, so `future_present` now uses `UnsafeCell<bool>` instead of an atomic flag. Corrected diagnostics measure the flag at 191.60-194.35 ps and full ready-completion components at 150.12-151.23 ns. Same-run public Tokio/Rayon comparison remains closed, while high local benchmark variance keeps scheduler handoff and async completion composition as active risks.
- Async poll-guard decision: `future_present` is not read on the poll hot path. The state transition from queued to polling grants exclusive poll permission and completion stores `ASYNC_COMPLETED`, so a separate future-present check duplicates the state invariant. Source contracts reject reintroducing the removed helper and guard; focused diagnostics measure async-ready at 652.71-665.92 ns and the default scheduling gate at 540.37-550.84 ns after removal.
- Rejected candidate decision: scheduler worker selection must occur before pending-count publication in the retained path. A fetch-first variant removed one load but regressed public rows, likely by widening the interval where pending work is globally visible before queue publication. The initial async-state `Arc` owner is also retained through spawn metrics recording because moving it into scheduling regressed wake-once. Benchmark contracts reject restoring the old inline handoff slot feature.
- Scheduler submission diagnostic decision: submission attribution remains diagnostic-only and feature-gated where it exposes scheduler internals. `ThreadScheduler::diagnostic_submission_queue_publication<C>` uses the same monomorphized `WorkClass` selection and real queue push/pop primitives with local atomics, so it measures publication shape without racing live workers or adding runtime policy state. Metrics stay recorded after successful scheduler submission because the before-submission row did not improve and would overcount failed submissions.
- Scheduler wake diagnostic decision: wake-path attribution remains diagnostic-only and uses sealed zero-sized marker types for empty, contended, and saturated paths. The production scheduler keeps the direct wake branch because routing it through a shared helper did not pass the scheduling gate. The expensive measured path is contended wake-all, not empty serial selected-worker wake.
- Bounded contended wake decision: production contended submissions use a sealed `BoundedContendedWake` ZST policy. The policy wakes the selected queue owner and one deterministic peer from `previous_pending`, preserving queue visibility without adding submission atomics, allocation, dynamic dispatch, or pending publication before queue publication. The helper is `#[inline(never)]` to keep the serial branch compact while retaining static dispatch.
- Prior `cargo bench -p moirai-benchmarks --bench performance_benchmarks task_scheduling_overhead`: public `spawn_fn`/`join` with asserted value measured 4.7953-5.7837 µs.
- `cargo bench -p moirai-benchmarks --bench performance_benchmarks task_scheduling_overhead`: exits under the Cargo benchmark path after disabling plot generation; public `spawn_fn`/`join` with asserted value measured 528.88-535.17 ns after rejecting production QPC lifecycle timing.
- `cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead`: retained scheduler path measured 533.08-540.29 ns after rejecting relaxed submit-side scheduler counters.
- `cargo bench -p moirai-benchmarks --features scheduler-diagnostics --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_scheduler_(select_worker_serial|pending_counter_pair|worker_unpark|priority_queue_push_pop|submission_queue_publication)|direct_spawn_metrics_(before|after)_scheduler_submission)"`: worker selection 1.1736-1.1792 ns, pending counter pair 9.6017-9.9314 ns, worker unpark 27.731-28.763 ns, priority queue push/pop 59.064-59.332 ns, submission queue publication 67.131-67.829 ns, metrics-before submission 241.22-255.10 ns, and metrics-after submission 225.53-254.91 ns.
- `cargo bench -p moirai-benchmarks --features scheduler-diagnostics --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_scheduler_(empty_wake_decision|contended_wake_decision|saturated_wake_decision|worker_unpark|submission_queue_publication))"`: selected-worker unpark 23.614-25.729 ns, submission queue publication 66.705-67.185 ns, empty wake decision 23.393-25.197 ns, contended wake-all decision 404.11-409.07 ns, and saturated no-wake decision 374.20-376.44 ps.
- Retained bounded wake rerun: `direct_scheduler_contended_wake_decision` measured 162.41-180.11 ns versus the prior 404.11-409.07 ns wake-all diagnostic. `task_scheduling_overhead` measured 546.64-561.03 ns within noise. Retained-code public comparison kept Moirai ahead: ready 563.74-579.31 ns versus Tokio 1.2717-1.3821 us, captured 473.92-493.81 ns versus Tokio 1.2943-1.5040 us, wake-once 553.83-578.44 ns versus Tokio 1.4885-1.5539 us, oversized 706.14-759.37 ns versus Tokio 1.3046-1.3845 us, and scope 403.98-502.30 ns versus Rayon 637.15-664.14 ns.
- `cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead`: public `spawn_fn`/`join` measured 387.46-416.14 ns with Criterion improvement after adding diagnostics and aligning the workspace manifest.
- Shared production wake helper candidate: `task_scheduling_overhead` first gate measured 540.36-584.30 ns with Criterion regression, so production retains the direct wake branch. Retained branch rerun measured 547.63-564.18 ns with no statistically significant change.
- `cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- public_result_handle_ready`: Moirai `spawn_fn` + `TaskHandle::join` measured 515.51-525.52 ns; Tokio `tokio::spawn` + `JoinHandle::await` measured 1.9694-2.2197 us; Moirai captured-ready `spawn_fn` + `TaskHandle::join` measured 552.23-562.69 ns; Tokio captured-ready `JoinHandle` measured 1.8724-2.0308 us; Moirai oversized-captured `spawn_fn` + `TaskHandle::join` measured 740.32-756.19 ns; Tokio oversized-captured `JoinHandle` measured 2.0403-2.1709 us; filtered Moirai async-ready `spawn_async` + `TaskHandle::join` measured 761.89-779.07 ns; filtered Moirai wake-once `spawn_async` + `TaskHandle::join` measured 782.06-792.38 ns; Tokio wake-once `JoinHandle` measured 2.9087-3.1672 us; Moirai scoped completion measured 506.22-515.42 ns; Rayon `scope` measured 679.76-697.40 ns.
- `cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- public_result_handle_ready`: Moirai ready/captured/oversized/async-ready/wake-once/scope rows measured 477.68-493.23 ns, 344.89-357.08 ns, 525.24-583.31 ns, 463.95-474.29 ns, 480.29-490.62 ns, and 275.67-285.21 ns respectively; equivalent Tokio/Rayon references measured 1.1178-1.2865 us, 986.24 ns-1.0404 us, 1.1105-1.1795 us, 1.1903-1.3200 us, and 591.62-614.02 ns.
- `cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_spawn_join_ready|tokio_spawn_join_ready|moirai_scope_single_ready|rayon_scope_single_ready)"`: Moirai ready result handles measured 529.27-556.48 ns versus Tokio at 1.9803-2.1555 us, and Moirai single scope measured 525.82-538.29 ns versus Rayon at 697.25-714.03 ns.
- `cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- "result_handle_diagnostics/(moirai_spawn_async_(ready|wake_once)|direct_scheduler_max_inline|direct_scheduler_oversized_(captured|capture_read_one)_result_slot)"`: Moirai async-ready measured 731.44-755.33 ns; wake-once measured 772.48-796.90 ns; max-inline captured result slot measured 498.22-520.61 ns; oversized captured result slot measured 608.32-649.76 ns after rerun; oversized read-one result slot measured 503.79-516.69 ns.
- `cargo bench -p moirai-benchmarks --bench thread_schedule_comparison`: `moirai_scope` 26.816-27.033 µs, `rayon_scope` 63.130-77.987 µs, `tokio_spawn_ready` 85.535-87.446 µs, `moirai_indexed_reduce` 1.5913-1.6066 µs, `rayon_indexed` 6.8983-7.3793 µs.
- `cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- mixed_unified_schedule`: Moirai unified mixed workload measured 42.000-42.856 us versus Tokio plus Rayon at 53.337-55.645 us.
- `cargo bench -p moirai-benchmarks --bench thread_schedule_comparison`: `scoped_ready_scaling` kept Moirai scope ahead at 64, 256, and 1024 work units; `indexed_reduce_scaling` kept Moirai indexed reduction ahead of Rayon indexed at 64, 256, and 1024 work units.
- `cargo bench -p moirai-benchmarks --bench industry_comparison`: `moirai_scope` stayed ahead of Tokio and Rayon at 100, 1,000, and 10,000 ready work units.
- `cargo bench -p moirai-benchmarks --bench industry_comparison`: Moirai indexed reduction stayed ahead of Rayon `into_par_iter().map(...).sum()` at 4,096, 32,768, and 65,536 work items.
- `cargo bench -p moirai-benchmarks --bench simd_benchmarks`: completed under the 300s verification bound.
- `cargo bench -p moirai-benchmarks --no-run`: compiled every benchmark executable.

---

## ADR-005: PyO3 Python Comparison Harness Boundary

**Date**: 2026-05-23
**Status**: Accepted
**Context**: Python runtime bindings for native Moirai execution

### Decision

`moirai-python` is a PyO3 binding crate over `moirai::Moirai`. It is a workspace crate and Python package, but it is not a dependency of the Rust runtime crates. The binding crate does not own scheduler, planner, backend logic, workload kernels, or comparison harnesses; it forwards to the Rust `moirai` crate. Python code provides the runtime facade and lifecycle tests.

### Rationale

- Preserves Rust runtime dependency boundaries while exposing `moirai::Moirai` to Python.
- Keeps one authoritative execution path through the Rust `moirai` crate instead of a Python or binding-crate stand-in.
- Separates FFI registration, facade code, and lifecycle tests.
- Prevents benchmark-specific Python functions from becoming public wrappers unless they correspond to comparable joblib or Tokio runtime primitives.

### Verification

- `cargo test -p moirai-python`
- `cargo clippy -p moirai-python -- -D warnings`
- `py -3.13 -m unittest discover moirai-python\tests`

### Residual Risk

Scoped completion-only ready work now exceeds the Tokio/Rayon scope and spawn baselines in scheduler-focused, industry-style, and single scoped-job targets. Indexed map/reduce exceeds Rayon indexed at 64, 256, and 1024 ready items. `Moirai::join` drains all work visible before quiescence without shutting down worker threads; work submitted after quiescence is a later batch. Public result-bearing `spawn_fn` and `spawn_async` no longer use mutexed result storage, condvar work notifications, condvar completion wakes, waiter-mutex registration, READY/park-racy waiter registration, per-task lifecycle `Arc` allocation, duplicate task-local timing for metrics, dynamic future dispatch in the async result path, boxed future pinning in the async result path, lifecycle mutexes in async polling, wrapper waker allocation, heap allocation for common small scheduled jobs, `Box<dyn FnOnce>` dispatch for oversized scheduled jobs, or a separate raw-pointer heap job variant for oversized scheduled jobs. The standalone async executor queue uses monomorphized poll/drop function pointers instead of `dyn Future` queue dispatch, and its handles use inline atomic result/waker slots instead of mutexed result storage and global waker hash maps. They remain a separate diagnostic category because each logical task still owns a result slot and Rayon `scope` is not result-handle equivalent. The public result-handle diagnostic includes real Tokio `JoinHandle` rows and measures Moirai ahead on the equivalent ready, captured-ready, oversized-captured, async-ready, and wake-once result-handle paths; it also includes a direct Moirai `scope` row ahead of Rayon's scoped completion row. The public-handle Criterion timeout was isolated to plot/report generation and closed by disabling plots in that target. A raw-pointer two-endpoint result slot was rejected after earlier stress variants reproduced a join hang and the latest targeted variant regressed `task_scheduling_overhead` to 633.01-640.02 ns; relaxed lifecycle metadata atomics were rejected after `task_scheduling_overhead` regressed to 608.31-641.98 ns; duplicate worker identity removal was rejected after the scheduling gate failed to retain an improvement; production QPC lifecycle timing was rejected after the public oversized-capture path and an earlier scheduling gate regressed; a larger spin threshold was rejected after no statistically significant improvement; an unconditional load-before-CAS result take path, per-task metrics timestamp removal, public `spawn_fn` routing through `SyncTask`, and per-worker running-bit wake suppression were rejected after benchmark regressions or no improvement. The retained result wait path keeps the already-ready claim as one direct CAS and uses a monomorphized zero-sized policy for load-gated pending spins. Direct result-slot diagnostics now show same-thread slot completion below 50 ns, so the remaining public result-handle work moves to scheduler wake/result-handoff variance, async wake/requeue locality, and registry lifecycle bookkeeping rather than result-slot pooling. Transport safe-channel receives now avoid owned deserialization for archived `String` payloads while preserving malformed-input rejection. Active competitive batch targets keep public-handle rows separate from scoped and indexed batch rows so value semantics remain explicit.

This ADR establishes the foundational architectural principles that guide all implementation decisions in the Moirai concurrency library, ensuring consistency with the project's vision of being a complete alternative to existing fragmented concurrency solutions.

---

## ADR-006: Typed Iterator Channel Fusion Boundary

**Date**: 2026-05-24
**Status**: Accepted
**Context**: `moirai-iter::channel_fusion` had boxed `FusableChannel` split/merge endpoints, a placeholder hash strategy, and a pipeline builder that returned success without executing stages.

### Decision

`ChannelSplitter<T, I, C>` and `ChannelMerger<T, C>` store concrete channel values in `Vec<C>` and dispatch through `C: FusableChannel<T>`. The incomplete `SplitStrategy::Hash` and `Pipeline` surface are removed instead of preserved through compatibility wrappers.

### Rationale

- Preserves one monomorphized channel type per split/merge instance and removes vtable dispatch from iterator channel routing.
- Keeps heterogeneous channel graphs explicit through caller-defined enum channel types rather than implicit boxed trait objects.
- Removes a placeholder hash branch that violated value-sensitive distribution semantics.
- Removes a non-executing pipeline API that reported success without performing work.

### Verification

- `cargo test -p moirai-iter -- --nocapture`
- `cargo clippy -p moirai-iter -- -D warnings`
- `cargo test -p moirai-benchmarks --test benchmark_contracts -- --nocapture`
- `cargo clippy -p moirai-benchmarks --test benchmark_contracts -- -D warnings`
- `cargo bench -p moirai-benchmarks --bench channel_matrix -- tokio_mpsc/p1_c1`
- `cargo bench -p moirai-benchmarks --bench channel_matrix -- moirai_mpmc/p1_c1`

### Residual Risk

`channel_matrix` keeps Moirai ahead of the same-run Tokio p1/c1 channel row, but Criterion reports a local baseline regression on the Moirai row. The next channel increment should isolate bounded-channel transport variance before changing the core MPMC implementation.

---

## ADR-007: Iterator Base And Streaming Monomorphization

**Date**: 2026-05-24
**Status**: Accepted
**Context**: `moirai-iter::base` exposed an unused boxed-future execution trait while `StreamingIter` boxed its producer and shifted buffered items.

### Decision

Remove the unused `base::ExecutionBase` trait and keep `execution::ExecutionBase` as the active context trait. Change `StreamingIter<T, F>` to store a concrete `F: FnMut() -> Option<T>` producer and a `VecDeque<T>` FIFO buffer. Split the touched iterator operations tree into streaming, stateful, and test leaves.

### Rationale

- Removes `Pin<Box<dyn Future<...>>>` from the iterator base surface.
- Preserves static dispatch for streaming producer calls through monomorphization.
- Replaces O(n) front removal with O(1) FIFO buffer operations.
- Keeps `iter_ops.rs` below the 500-line structural target without changing public adapter names.
- Avoids compatibility wrappers because this is a pre-1.0 breaking cleanup.

### Verification

- `cargo test -p moirai-iter -- --nocapture`
- `cargo clippy -p moirai-iter -- -D warnings`
- `cargo test -p moirai-benchmarks --test benchmark_contracts -- --nocapture`
- `cargo clippy -p moirai-benchmarks --test benchmark_contracts -- -D warnings`
- `cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- mixed_unified_schedule`

---

## ADR-008: Timer Wheel Lazy Cancellation

**Date**: 2026-05-24
**Status**: Accepted
**Context**: `TimerWheel::cancel` ignored the timer id and always returned `false`, so the timer-wheel cancellation API did not perform the requested state transition.

### Decision

Move `TimerWheel` into `moirai-async/src/timer/wheel.rs` and implement cancellation with a lazy `HashSet<u64>` of canceled timer ids. Keep scheduled timer entries in a `BinaryHeap`, skip canceled entries during expiration polling, and expose `TimerWheel` through the existing `timer` module boundary.

### Rationale

- Preserves heap-based deadline ordering without arbitrary heap removal on the scheduling path.
- Makes cancellation value-sensitive: first cancel succeeds, duplicate or absent cancels fail.
- Prevents canceled timer wakers from firing when their heap entry expires.
- Keeps `timer.rs` below the 500-line structural target through a cohesive timer-wheel leaf module.

### Verification

- `cargo test -p moirai-async timer_wheel -- --nocapture`
- `cargo test -p moirai-benchmarks --test benchmark_contracts -- --nocapture`
- `cargo bench -p moirai-benchmarks --bench example_pattern_comparison -- example_tokio_fanout`
- `cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- mixed_unified_schedule`

### Residual Risk

Lazy cancellation retains canceled entries until their deadline reaches the heap root. This preserves low scheduling-path cost but means long-deadline canceled timers can occupy heap storage until expiration; a future compaction policy should be benchmarked before adoption.

---

## ADR-009: Parallel Iterator Vertical Split

**Date**: 2026-05-24
**Status**: Accepted
**Context**: `moirai-iter::parallel` mixed trait surfaces, sources, adapters, consumers, and tests in one file while reduction consumers had inconsistent result carriers.

### Decision

Split the parallel iterator implementation into `traits`, `sources`, `adapters`, `consumers`, and `tests` leaves under `moirai-iter/src/parallel/`. Keep `moirai-iter/src/parallel.rs` as the public module root and re-export the same public items.

### Rationale

- Keeps each touched leaf below the 500-line structural target.
- Separates Rayon-style public traits from source iterators and consumer machinery.
- Keeps reduction state in `Reduction<T, F>` so split halves combine through the caller-provided associative function.
- Adds an empty-vector base case before chunk splitting so empty reductions terminate.

### Verification

- `cargo test -p moirai-iter parallel -- --nocapture`
- `cargo test -p moirai-iter -- --nocapture`
- `cargo test -p moirai-benchmarks --test benchmark_contracts -- --nocapture`
- `cargo clippy -p moirai-iter -- -D warnings`

---

## ADR-010: Rayon-Style Transform Adapter Expansion

**Date**: 2026-05-24
**Status**: Accepted
**Context**: The Rayon adapter audit listed `filter_map` and `flat_map` as unsupported after `enumerate` and `zip` were already present with value tests.

### Decision

Add `ParallelIterator::filter_map` through `FilterMap<I, F>` and `ParallelIterator::flat_map` through `FlatMap<I, F>`. The adapters store concrete closure types and monomorphize through the existing `ParallelIterator` trait, preserving the non-indexed adapter boundary.

### Rationale

- Closes the next Rayon-style transform adapter gap without claiming full Rayon parity.
- Keeps adapter variation in generic types instead of dynamic callbacks.
- Uses value-semantic tests for optional retention and flattened output order.
- Leaves indexed execution on `Moirai::for_each_indexed` and `Moirai::map_reduce_indexed`.

### Verification

- `cargo test -p moirai-iter parallel -- --nocapture`
- `cargo test -p moirai-benchmarks --test benchmark_contracts -- --nocapture`
- `cargo clippy -p moirai-iter -- -D warnings`

---

## ADR-011: Rayon-Style Utility Adapter Expansion

**Date**: 2026-05-25
**Status**: Accepted
**Context**: The adapter audit still classified `inspect`, `panic_fuse`, `chunks`, and `partition` as unsupported in the non-indexed `moirai-iter::parallel` surface.

### Decision

Add typed `Inspect<I, F>`, `PanicFuse<I>`, and `Chunks<I>` adapters plus a `ParallelIterator::partition` collector. Keep sorting out of this module because Rayon sorting is a slice-extension boundary, not a `ParallelIterator` adapter.

### Rationale

- Stores closures and policy state in concrete generic types with no `dyn Trait` dispatch.
- Uses a zero-sized `PanicFusePolicy` marker so panic-fuse routing stores no runtime strategy state.
- Uses a transparent `ChunkSize` newtype so zero chunk size is rejected at construction before iteration.
- Keeps side-effect and chunk implementations in adapter leaves to preserve the vertical file hierarchy and line-count target.
- Adds direct Rayon comparison rows only where Rayon exposes equivalent public paths.

### Verification

- `cargo test -p moirai-iter parallel -- --nocapture`
- `cargo test -p moirai-iter -- --nocapture`
- `cargo test -p moirai-benchmarks --test benchmark_contracts -- --nocapture`
- `cargo clippy -p moirai-iter -- -D warnings`
- `cargo clippy -p moirai-benchmarks --test benchmark_contracts -- -D warnings`
- `cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- --quiet`

### Residual Risk

`PanicFuse` preserves value and panic propagation in the current non-indexed adapter layer. If this adapter layer later executes sibling branches concurrently, panic-fuse must gain a shared cancellation flag in the consumer path before claiming Rayon-equivalent early-stop behavior.

---

## ADR-012: Parallel Slice Sorting Boundary

**Date**: 2026-05-25
**Status**: Accepted
**Context**: Rayon exposes sorting through `ParallelSliceMut`, not through `ParallelIterator`. The adapter audit therefore needed a separate slice-extension boundary instead of another non-indexed iterator adapter.

### Decision

Add `moirai_iter::parallel::ParallelSliceMut` for `[T]` with stable and unstable sort entry points: `par_sort`, `par_sort_by`, `par_sort_by_key`, `par_sort_unstable`, `par_sort_unstable_by`, and `par_sort_unstable_by_key`.

### Rationale

- Keeps sorting in the slice domain where mutation and in-place ordering are explicit.
- Preserves static dispatch through a generic extension trait instead of dynamic sorting strategies.
- Keeps stable and unstable algorithms behind one trait surface rather than type-specific API names.
- Uses repository-local value tests, stability tests, panic-safety coverage, and direct Rayon `ParallelSliceMut` benchmark rows.

### Verification

- `cargo test -p moirai-iter sorting -- --nocapture`
- `cargo test -p moirai-benchmarks --test benchmark_contracts sorting_slice_extension_is_value_semantic_and_benchmarked -- --nocapture`
- `cargo clippy -p moirai-iter -- -D warnings`
- `cargo clippy -p moirai-benchmarks --test benchmark_contracts -- -D warnings`
- `cargo bench -p moirai-benchmarks --bench sorting_comparison -- --quiet`

### Residual Risk

Stable sorting uses a temporary left-half buffer during merge. This preserves stable ordering and keeps the public API in-place, but the implementation must keep panic-safety tests active because comparator panics can occur during merge.

---

## ADR-013: Async I/O Facade Audit Boundary

**Date**: 2026-05-25
**Status**: Accepted
**Context**: The Tokio gap audit needed to separate covered Moirai-owned file/network facade behavior from unsupported Tokio reactor-native I/O drop-in compatibility.

### Decision

Treat `moirai-async::fs` as a Moirai-owned file facade with value-semantic tests and a `tokio::fs::read` benchmark row. Treat `moirai-async::net` as a Moirai-owned socket facade with TCP and UDP loopback value tests. PAL TCP types may register wakers with an active `IoReactor`; without an active reactor they must self-wake before returning `Pending` so local executors do not deadlock on delayed readiness. PAL reactor-spawned tasks must publish completion to their `TaskHandle` through per-task state. PAL reactor platform dispatch must use the compile-target `PlatformReactor`, and queued reactor futures must use bounded inline storage plus monomorphized poll/drop dispatch instead of `dyn Future`. Moirai comparison benchmark rows must use `Moirai::block_on`, not an external futures executor. Do not claim Tokio reactor-native file or network drop-in compatibility until PAL file readiness, Tokio trait compatibility, cancellation, and backpressure contracts are specified.

### Rationale

- Keeps the comparison evidence tied to implemented APIs instead of broad ecosystem claims.
- Removes obsolete async-file placeholder future machinery that provided no authoritative execution path.
- Adds network payload and counter assertions without measuring std-socket immediate-poll facades against Tokio's reactor as if they were equivalent.
- Adds a PAL TCP/UDP no-active-reactor wake contract so cooperative socket facades remain progress-safe under `block_on`.
- Adds PAL reactor task-handle completion state so spawned ready tasks cannot leave awaiting handles pending forever.
- Removes PAL reactor dynamic dispatch at the platform and queued-future boundaries where the concrete implementation is known by compile target or monomorphized at spawn, and stores fitting futures inline under a static size/alignment contract.
- Keeps Moirai benchmark rows on the Moirai runtime surface instead of measuring through `futures::executor::block_on`.
- Replaces Linux epoll's no-op wake placeholder with an internal `eventfd` wake path.
- Preserves the production dependency boundary: Tokio remains a benchmark/reference dependency only.

### Verification

- `cargo test -p moirai-async net -- --nocapture`
- `cargo test -p moirai-async fs -- --nocapture`
- `cargo test -p moirai-pal -- --nocapture`
- `cargo test -p moirai-benchmarks --test benchmark_contracts -- --nocapture`
- `cargo bench -p moirai-benchmarks --bench async_fs_comparison -- --quiet`
- `cargo bench -p moirai-benchmarks --bench async_udp_comparison -- --quiet`

### Residual Risk

Tokio reactor-native I/O drop-in compatibility remains deferred. The next I/O increment must define PAL file readiness ownership, Tokio trait compatibility, cancellation semantics, and bounded resource behavior before adding Tokio network/file compatibility benchmarks.

---

## ADR-010: Blocking Result Wait Spin Budget

**Date**: 2026-05-24
**Status**: Accepted
**Context**: `TaskHandle::join` uses the sealed zero-sized `BlockingResultWait` policy to probe a pending result slot before entering the single-waiter park fallback. Caller-side attribution measured the previous 100-spin pending miss at 1.1886-1.4520 us.

### Decision

Set `MAX_SPIN_ATTEMPTS` to 64 for the blocking result-wait policy. Preserve the direct first READY-to-TAKEN CAS for already-ready handles, the relaxed-load gated pending probes, and the existing `WAITING` plus `thread::park` fallback.

### Rationale

- Keeps wait-policy dispatch static through `TaskResultSlot::wait::<BlockingResultWait>`.
- Preserves a zero-sized policy type and associated-const budget with no runtime storage.
- Reduces pending CPU spin work before the blocking fallback.
- Avoids result-slot layout changes, allocation, dynamic dispatch, result-slot pooling, and scheduler-side atomics.

### Verification

- `cargo test -p moirai-core --features result-diagnostics task:: -- --nocapture`
- `cargo test -p moirai-benchmarks --features scheduler-diagnostics,result-diagnostics --test benchmark_contracts -- --nocapture`
- `cargo clippy -p moirai-core --features result-diagnostics -- -D warnings`
- `cargo bench -p moirai-benchmarks --features scheduler-diagnostics,result-diagnostics --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_result_slot_(ready_take|spin_miss|register_waiter|complete_waiting)|direct_scheduler_join_fast_spin_(quiescent|pending)|moirai_spawn_join_ready|direct_scheduler_result_slot|direct_scheduler_submit_join)"`
- `cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead`
- `cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_spawn_join_ready|tokio_spawn_join_ready|moirai_spawn_join_captured_ready|tokio_spawn_join_captured_ready|moirai_spawn_join_oversized_captured_ready|tokio_spawn_join_oversized_captured_ready|moirai_spawn_async_wake_once|tokio_spawn_async_wake_once|moirai_scope_single_ready|rayon_scope_single_ready)"`

### Residual Risk

The 64-spin budget keeps Moirai ahead of same-run Tokio/Rayon public rows and leaves `task_scheduling_overhead` statistically unchanged, but captured, wake-once, oversized, and scope rows still show local Criterion baseline regressions. Further work should split scheduler result-publication variance before changing the wait budget again.

---

## ADR-014: Reactor-Bound Async I/O and Readiness Integration

**Date**: 2026-05-25
**Status**: Accepted
**Context**: We needed to complete the transition from a cooperative/blocking async I/O simulation to a true event-driven, reactor-backed asynchronous I/O and execution architecture. The busy-polling loop in the async executor consumed excessive CPU, and file/socket operations lacked real readiness integration.

### Decision

1. **Reactor-Bound Event Loop**: Integrate a thread-safe `IoReactor` that manages OS-level handles (using `epoll` on Linux, `kqueue` on macOS, and readiness structures on Windows). Establish thread-local `ACTIVE_REACTOR` bindings.
2. **Readiness-Driven Sockets**: Implement non-blocking `AsyncTcpStream` and `AsyncTcpListener` in `moirai-pal::net` that register wakers with the `IoReactor` on `WouldBlock` errors and self-wake when no active reactor is present.
3. **Cooperative File Operations**: Build a clean `AsyncFile` abstraction in `moirai-pal::fs` that executes non-blocking read, write, seek, and flush operations, relying on a cooperative waker-yielding mechanism for safety.
4. **Executor Run-Queue Scheduling**: Replace the task-queue busy-polling loop in `moirai-async::executor::AsyncExecutor` with a thread-safe run-queue and block-on notification powered by a platform-specific `ExecutorWaker`.
5. **Clean Modular Delegation**: Decouple `moirai-async::net` and `moirai-async::fs` facades by delegating entirely to their `moirai-pal` counterparts, adhering to the 500-line structural limit.

### Rationale

- **High-Performance Event Dispatch**: Eliminates unnecessary polling loops, reducing CPU utilization of idle executors to zero.
- **Zero-Copy Readiness Integration**: Avoids buffer allocations and copies by delegating handle registration and waker updates directly to the platform reactor.
- **Progress Guarantee**: The fallback waker yield ensures that execution progresses even when an I/O reactor is absent or when operations are synchronous.
- **Strict Domain Boundaries**: Keeps platform-specific socket/file descriptors confined to `moirai-pal`, exposing clean traits and facades to `moirai-async`.

### Verification

- `cargo test -p moirai-pal --all-targets`
- `cargo test -p moirai-async --all-targets`
- `cargo test --workspace`
- `cargo bench -p moirai-benchmarks --test benchmark_contracts`

### Residual Risk

Platform-specific async file I/O (e.g., via io_uring or Windows IOCP) remains deferred in favor of cooperative standard-file abstractions. Future work must define thread-pool scheduling for file blocking operations if true non-blocking disk access is required under high load.

---

## ADR-006: Async I/O Compatibility and Tokio Trait Integration

**Date**: 2026-05-25
**Status**: Accepted
**Context**: To provide a complete, low-overhead alternative to Tokio, Moirai needs a unified compatibility strategy for asynchronous I/O operations. This involves supporting or matching `AsyncRead`, `AsyncWrite`, and `AsyncBufRead` semantics, implementing a robust file readiness strategy, and ensuring strict cancellation safety and backpressure guarantees.

### Decision

1. **Trait Equivalence and Interoperability**:
   - Moirai defines `moirai_async::io::{AsyncRead, AsyncWrite, AsyncBufRead}` traits.
   - For ecosystem integration, Moirai provides feature-gated conversion shim layers (e.g., `into_tokio()` / `from_tokio()`) mapping Moirai's native I/O structures to `tokio::io` traits and vice-versa, avoiding any compile-time or runtime dependencies in the default build configuration.
2. **File Readiness and Blocking I/O Strategy**:
   - Since standard disk files do not support traditional poll-based readiness (e.g., via epoll/kqueue) on typical Unix platforms, Moirai implements a dual-path file readiness strategy:
     - **Cooperative Worker Offloading**: Standard disk file operations that would otherwise block are dispatched to the `BlockingTask` scheduler pool using `spawn_blocking` wrappers, ensuring that asynchronous worker threads remain free.
     - **Platform Native AIO/IOCP**: On platforms supporting true non-blocking file systems (such as Windows IOCP or Linux io_uring when enabled), Moirai registers the file handle directly with the `IoReactor` to receive completion notifications.
3. **Cancellation Safety Contracts**:
   - All async I/O futures (e.g., `Read`, `Write`, `Flush`) must be fully cancellation-safe. If an I/O future is dropped before completion:
     - The internal handle state must cleanly cancel the pending I/O operation (e.g., via `CancelIoEx` on Windows or cancellation queues in io_uring) to prevent dangling references to stack-allocated or heap-allocated user buffers.
     - Shared buffer ownership is structured using zero-copy primitives or Rust's ownership model so that no buffer is leaked or left in an undefined state upon early drop.
4. **Backpressure and Resource Limits**:
   - Write streams must enforce backpressure by returning `Poll::Pending` when reactor write queues are saturated.
   - Flow control is mediated by a cooperative waker-registration scheme where writers are notified to wake only when the underlying socket or descriptor buffer has drained below a configured water-mark threshold.

### Rationale

- **Ecosystem Coexistence**: Allowing clean shims for Tokio traits allows Moirai to serve as a drop-in replacement or coexist in mixed-library environments without polluting the core dependency tree.
- **Worker Isolation**: Keeping blocking file I/O separate from async task scheduling prevents CPU-bound tasks and async event loops from starving, matching Moirai's hybrid execution model goals.
- **Safety and Correctness**: Explicit cancellation semantics and buffer lifetime guarantees prevent memory corruption and resource leaks during future cancellation (e.g., under timeouts).

### Verification

- Comprehensive unit testing of read, write, seek, and cancel operations under simulated slow connections.
- Benchmark validation mapping throughput and latency against equivalent Tokio streams.
- Clippy and cargo checks verified on target files.

### Residual Risk

- OS-specific differences in disk caching and non-blocking I/O support may result in varying file I/O latency profiles between platforms. Continuous empirical validation is required.

---

## ADR-007: WebAssembly Browser Event-Loop Integration

**Date**: 2026-05-25
**Status**: Accepted
**Context**: Moirai's WASM target architecture must run reliably in standard web browsers where OS threads are unavailable, requiring integration with the browser's JavaScript event loop and cooperative task scheduling.

### Decision

1. **Cooperative Web Worker Event Loop**:
   - In WASM environments lacking native threading, Moirai's `WasmExecutor` cooperative mode runs directly inside the JavaScript thread or schedules work across Web Workers.
   - Tasks queue directly in Moirai's light internal queue. Tick dispatching cooperates with JS via `requestAnimationFrame`, `setTimeout`, or JS Microtasks (e.g., `Promise.resolve().then(...)`) to prevent blocking the browser's rendering thread.
2. **Browser Callback Ownership and Lifetime Management**:
   - WASM-JS boundaries use clear ownership patterns for closures and event listeners:
     - Event listener callbacks (e.g., for fetch, websockets, or timer events) are wrapped in Rust-managed types that automatically deallocate and unregister listeners when dropped.
     - Rust futures wait on JS Promises via thread-safe channels or local polling loops mapped to JS events.
3. **Event Queue Mutation**:
   - The event loop in WASM uses a lock-free or single-threaded cooperative queue. Interrupts and events from JS (such as I/O readiness, timers, or worker messages) write directly to the event queue and wake the Moirai reactor.
4. **Static and Dynamic WASM Verification**:
   - The workspace enforces WASM target compilation check via:
     `cargo check --target wasm32-unknown-unknown --all-features`
   - CI runs headless WASM tests using `wasm-pack test` or `cargo-diners` equivalents to guarantee event-loop correctness, callback deallocation, and task execution progress.

### Rationale

- **Browser Responsive Execution**: Cooperating with JavaScript's execution cycles ensures that Moirai applications do not cause page freezes or browser warnings.
- **Leak Prevention**: Explicit callback lifecycle tracking prevents memory accumulation at the boundary of WASM and JS runtimes.
- **Universal Portability**: Enables Moirai to compile cleanly for browsers, Node.js, and WASI hosts.

### Verification

- `cargo check --target wasm32-unknown-unknown --workspace`
- Automated test runs in simulated browser environments.

### Residual Risk

- Dynamic browser memory management and JS engine garbage collection cycles can introduce non-deterministic latency. Performance testing under browser load is necessary to isolate engine-specific variances.

---

## ADR-015: Native HTTP/S3 Transport Stack (Tokio-Free Object Storage)

**Date**: 2026-06-02
**Status**: Proposed (requires sign-off before P1 implementation)
**Change-class**: [arch]
**Context**: consus's `s3` feature is the last hard Tokio coupling across the atlas repos. `S3Reader` (consus-io `io/async_io/s3.rs`) drives `rusoto_s3` + `reqwest`, both wired to Tokio's concrete reactor/types and not runtime-swappable. consus's async *format* layer (`AsyncReadAt`-generic HDF5 parsers) is already runtime-agnostic and Moirai-drivable via `Moirai::block_on` (tokio there is a dev-dependency test harness only). Only the network transport remains coupled.

### Foundation already in place (do NOT rebuild)
Per ADR-014 / ADR-006(async) / ADR-013, Moirai already provides, reactor-backed and value-tested:
- `moirai-pal::net`: async `AsyncTcpStream`/`AsyncTcpListener` over epoll/kqueue/IOCP with waker registration and no-active-reactor self-wake.
- `moirai-async::{net,io,fs}`: `AsyncRead`/`AsyncWrite`/`AsyncBufRead` traits, `spawn_blocking`, cancellation + backpressure contracts, `Moirai::block_on`.
The transport gap is therefore ONLY the three layers above the socket: TLS, HTTP/1.1, and the S3 surface.

### Decision
Split the work by domain along the project's `communication = moirai` / `datatype-and-store = consus` boundary. Moirai ships **store-agnostic** transport only and never learns what S3/AWS is; the S3 *protocol* (a vendor-specific storage-addressing concern) lives in consus on top of Moirai's HTTP. Reuse audited sans-I/O libraries for all cryptography and parsing; build only the I/O orchestration glue each side must own.

**In Moirai (generic communication, two new crates over the existing `moirai-net` sockets):**
1. `moirai-tls` — TLS 1.2/1.3 client sessions by driving `rustls` (sans-I/O `ClientConnection` state machine) over a `moirai-async` `AsyncTcpStream`. Moirai owns only the read/write pump between socket and rustls plaintext/ciphertext buffers, plus the handshake-completion future. No hand-rolled cryptography. Cert verification via `rustls-platform-verifier`/`webpki-roots`.
2. `moirai-http` — HTTP/1.1 client over `moirai-tls`/`moirai-net`. Reuse the `http` crate (Request/Response/header types) and `httparse` (sans-I/O head parser). Moirai owns: request serialization, response-body framing (Content-Length + chunked transfer decoding), bounded-capacity keep-alive connection pooling, redirect handling, and per-request deadline via the Moirai timer. HTTP/2 explicitly out of scope (S3 runs over HTTP/1.1). **This is Moirai's S3-facing boundary — it knows HTTP, not S3.**

**In consus (storage backend, NOT in Moirai):**
3. consus S3 client — rebuild consus's existing S3 backend (`consus-io` `io/async_io/s3.rs`, `consus-zarr` `S3Store`; a `consus-s3` crate or `consus-io` module) on `moirai-http` instead of `rusoto_s3` + `reqwest`. Owns the vendor-specific parts: SigV4 signing (`aws-sigv4` or a direct HMAC-SHA256 canonical-request impl), `GetObject(Range)` + `HeadObject`, bucket/key addressing, credential resolution (env + `~/.aws/credentials`), S3 error-XML decoding (`quick-xml`). Surfaces an `AsyncReadAt` implementor in place of the rusoto `S3Reader`. **Rationale:** "where my datasets live and how to address them" is a storage concern; keeping SigV4/GetObject out of Moirai preserves Moirai as a pure, AWS-agnostic communication library (datatype/store = consus, communication = moirai).

### Reuse-vs-build matrix
- Reuse, in **Moirai** (sans-I/O, runtime-agnostic, none Tokio-coupled): `rustls` + roots, `http`, `httparse`, `socket2`.
- Reuse, in **consus** (S3 protocol, sans-I/O): `aws-sigv4`, `quick-xml`. These AWS/XML deps do NOT enter Moirai's tree.
- Build in **Moirai**: TLS↔socket pump; HTTP connection lifecycle/pool/chunked codec/timeouts.
- Build in **consus**: S3 request assembly/signing over `moirai-http` + the `AsyncReadAt` adapter.
The security-critical and spec-heavy parts (TLS, HTTP grammar, SigV4) are reused; each side builds only its readiness/lifecycle glue. Hand-rolling TLS is prohibited.

### Execution-model alignment
All three layers are async-domain → `moirai-async` (AsyncPolicy), never `moirai-parallel`; preserves the parallel≠concurrent split. The hybrid scheduler lets a parallel chunk-decompress await an S3 range fetch on the same pool (unified-runtime invariant). Pure consus format logic stays synchronous (async-contagion prohibition); only the byte-source boundary is async.

### Variant / abstraction strategy
- Transport selection (`rustls` vs future native-tls; HTTP/1.1 vs future HTTP/2) behind a sealed `Transport`/`HttpVersion` strategy trait — static dispatch, no `dyn` on the hot path.
- consus exposes the backend as a feature axis: `s3-moirai` (new) vs `s3-tokio` (legacy rusoto, retained until parity proven), both implementing the same `AsyncReadAt`. No public API change to consus's storage surface.
- Region/endpoint/credentials as validating newtypes (primitive-obsession prohibition).

### Alternatives considered
1. Reimplement TLS/crypto from scratch — REJECTED: security-critical, no value over rustls.
2. Fork reqwest/hyper onto Moirai I/O — REJECTED: hyper is deeply Tokio-coupled; fork maintenance unbounded.
3. Embed a current-thread Tokio runtime on a Moirai worker to host reqwest (`moirai-async` has a `tokio-compat` feature) — REJECTED as the goal (still ships Tokio) but RETAINED as the documented fallback if an upper layer stalls, so consus is never blocked.
4. Layered sans-I/O-glue stack over the existing reactor — SELECTED.

### Expected failure modes / risks
- Windows IOCP async-TCP maturity (dev platform; IOCP is completion- not readiness-based). Mitigation: P1 gate runs moirai-net loopback + TLS-handshake suites on Windows specifically before proceeding.
- TLS correctness/security — mitigated by reusing rustls + adversarial cert tests (expired / wrong-host / untrusted-root must fail closed).
- HTTP/1.1 edge cases (chunked trailers, 100-continue, server-initiated close mid-pool, slow-loris) — covered by differential tests vs reqwest against a local server.
- Connection-pool resource bounds — mandatory bounded capacity (no unbounded queues); idle-eviction timer.
- Dependency policy: new crates vetted via `cargo deny` and pinned; MSRV checked.
- Scope creep (HTTP/2, presigned URLs, multipart upload) — out of scope; consus needs only ranged GET + HEAD.

### Verification plan (evidence tiers per layer)
- `moirai-tls`: loopback handshake against a rustls server; plaintext-roundtrip differential vs `tokio-rustls`; adversarial cert-validation fail-closed tests.
- `moirai-http`: local HTTP/1.1 test server; property tests on chunked/Content-Length framing and pool reuse; differential — identical GET via `reqwest` vs `moirai-http` yields byte-identical status/headers/body.
- consus S3 client (on `moirai-http`): SigV4 known-answer tests from AWS's published canonical-request/string-to-sign/signature vectors; differential vs `rusoto_s3` against local MinIO/`s3mock` — byte-identical `GetObject(Range)`/`HeadObject`.
- consus integration: existing consus S3 property/integration tests run on both `s3-moirai` and `s3-tokio` against MinIO → byte-identical dataset reads.
- Comparative benchmarks (criterion; explicit deliverable): ranged-GET throughput + p50/p99 latency + CPU-time/req + allocations, moirai-s3 vs rusoto_s3, against (a) localhost MinIO (RTT≈0, exposes per-request CPU/syscall overhead) and (b) a latency-injected proxy (`toxiproxy`) at realistic RTT (where both converge on network bound). These are the go/no-go evidence for flipping the default; an intermediate regression triggers profiling/optimization (pool warm-up, buffer reuse, vectored writes), not reversion (optimization farsight).

### Phasing (vertical slices; each leaves the tree green)
- P0 [arch]: this ADR + spike proving moirai-net loopback echo and a rustls handshake over loopback on Linux **and** Windows. Exit: green TLS-roundtrip integration test.
- P1 [minor, moirai]: `moirai-tls` (rustls glue, cert verification, cancellation-safe).
- P2 [minor, moirai]: `moirai-http` (GET/HEAD, chunked, keep-alive pool, redirect, timeout). Moirai's deliverable ends here — it ships generic, AWS-agnostic transport.
- P3 [minor, consus]: consus S3 client on `moirai-http` (SigV4, GetObject Range, HeadObject, creds, error XML) behind an `s3-moirai` feature alongside legacy `s3-tokio`; both green on MinIO; consus-hdf5 async tests move to `Moirai::block_on`.
- P4 [minor, consus]: comparative bench recorded (consus-s3-on-moirai-http vs `rusoto_s3`).
- P5 [major, consus]: flip consus default to `s3-moirai`; demote rusoto/reqwest to legacy/optional; remove Tokio from consus's production tree (Tokio remains benchmark/reference only, per ADR-001/013).

### Classification & sign-off
[arch] — new canonical crates + a new transport boundary. Requires ADR sign-off (this document) before P1 opens. Implementation tracked in `docs/adr-015-checklist.md`.

## ADR-016: One Ring-Buffer Core and One Channel Family in moirai-core

- Status: Proposed (awaiting sign-off; implements the 2026-07-02 structural
  audit's S1/S2 findings)
- Change class: [arch]

### Context

moirai-core ships five sibling implementations of the same ring-buffer
algorithm family: `communication::RingBuffer` (lock-free SPSC, CachePadded
sequences, MaybeUninit slots), `channel::spsc::SpscChannel` (a line-for-line
clone of that Lamport ring plus a `closed` flag and spin-blocking),
`memory::UnifiedRingBuffer` (the same ring, mutex-locked — its "lock-free
zero-copy" doc is false), `communication::zero_copy::MemoryMappedRing` (the
same ring behind CAS spin-locks; not memory-mapped despite the name), and
`channel::mpmc::BoundedMpmcQueue` (Vyukov — the one genuinely distinct
algorithm). Above them sit four channel bounded-contexts (`channel/`,
`unified_channel/`, `communication::zero_copy/`, plus the bare
`communication::RingBuffer`) with three duplicated error enums
(`ChannelError`, `UnifiedChannelError`, `ZeroCopyError`) all repeating
Full/Empty/Closed/WouldBlock. Only `MpmcChannel` is consumed by the live
runtime (`moirai/src/runtime.rs`, moirai-transport); `unified_channel` is
consumed solely by `moirai-iter::advanced_patterns` (itself a prune candidate,
ADR-017); `HybridChannel` and `zero_copy` are consumed only by benchmarks and
contract tests. `ipc::SharedQueue` is a justified separate ring (cross-process
Pod contract) and stays.

### Decision (proposed)

The variation dimensions across the five rings are exactly producer/consumer
cardinality and blocking policy — a bounded set expressible without cloning:

1. Keep TWO algorithm cores: the SPSC Lamport ring (canonical home:
   `communication::RingBuffer`) and the Vyukov MPMC (`BoundedMpmcQueue`).
2. Express blocking policy as a ZST strategy over those cores (the crate
   already has this exact pattern in `task::handle::ResultWaitPolicy`):
   `NonBlocking` / `SpinThenPark`, monomorphized so the non-blocking path
   compiles to the bare ring.
3. `SpscChannel` becomes a thin `RingBuffer + closed-flag + policy`
   composition (the shape `HybridChannel` already proves); delete
   `UnifiedRingBuffer` and `MemoryMappedRing`, retargeting `unified_channel`
   (or deleting it with moirai-iter's advanced_patterns per ADR-017) and
   `zero_copy` consumers onto the canonical cores.
4. ONE channel error enum in `channel::error`; the other two enums' extra
   variants (InvalidConfig, the zero-copy set) become variants or per-call
   typed errors. Every call site updated in the same change; no aliases.

### Consequences

Deletes roughly 1.5-2k lines of parallel implementations while keeping every
live capability; the 18-round-audited MPMC/hybrid protocols are preserved
as-is (this ADR relocates and dedups shells, it does not restructure the
verified CAS protocols). Consumers to update: moirai (runtime), transport,
benchmarks/contract tests, and moirai-iter's advanced_patterns (interlocks
with ADR-017 — implement after that decision).

## ADR-017: moirai-iter Disposition (prune vs continue)

- Status: Proposed — BLOCKED ON OWNER ADJUDICATION (conflicting active intent)
- Change class: [arch]

### Context

The 2026-07-02 audit found moirai-iter (~14.6k lines) delivers a sequential
`ParallelIterator` (drive() runs both split halves inline on the caller),
fake SIMD (loads an unused `_mm256` intrinsic then scalar-loops), hardcoded
0.5/0.3 multi-system utilization, "execute locally — for now" distributed
placeholders, per-element `block_on` async terminals, and has ZERO production
consumers across the atlas checkout, while moirai-parallel is the real
data-parallel SSOT with 79+ consumer call sites. The audit recommends: extract
the four real pieces (`par_sort*` → moirai-parallel rebased onto the
SyncTask executor, `stream::concurrent_map` → moirai-async, numa/prefetch
primitives if a consumer materializes), delete the rest, and drop `iter` from
the umbrella's default features as the first increment.

HOWEVER: a concurrent session has been actively investing in moirai-iter
(commit 101d72c "refactor(iter): dedup ReduceWithConsumer into ReduceConsumer"
and related property-test commits landed on main this week). Pruning a
subsystem another session is actively improving is a design-intent conflict
that must not be resolved unilaterally by either session.

### Decision required from the owner

(a) PRUNE per the audit (the 14.6k lines are mostly non-functional and
unconsumed; the concurrent session's dedup effort would be redirected to
moirai-parallel), or (b) CONTINUE moirai-iter as a maintained surface — in
which case its `ParallelIterator` must be made actually parallel (route
drive() through the SyncTask executor), the fake SIMD/utilization/distributed
placeholders deleted regardless (HARD integrity rule), and a consumer story
documented. Option (b)'s integrity subset (delete fakes, fix the sequential
drive) is required under either outcome; the difference is whether the crate
survives. Until adjudicated, no session should expand moirai-iter's surface.

## ADR-017 update (2026-07-03): RESOLVED — continue-and-make-real

Owner adjudication (the maintainer) chose to KEEP moirai-iter and make its
surfaces real rather than prune. Executed on branch
`refactor/remove-dead-subsystems`:
- **Parallel:** `ParallelIterator::drive` now forks the recursive `Consumer`
  split through the unified scheduler (`moirai_parallel::join_with::<Parallel>`
  above `ADAPTIVE_PARALLEL_THRESHOLD`) — genuine work-stealing, one fork-join
  SSOT shared with moirai-parallel; a proof test asserts execution across >1
  worker thread. (commit `perf(iter): …fork-join`)
  **REVERTED (2026-07-03, commit `revert(moirai-iter): …sequential`):** this
  fork-join drive was unsound under *nested* iteration — the scheduler scope
  deadlocked (single worker) and corrupted the heap under concurrent nesting.
  `drive` returned to sequential-by-contract; the root cause is fixed in
  ADR-019, after which a parallel drive can be reintroduced (ISSUE-208 (c)).
- **Async:** the terminal futures (`AsyncForEach/Fold/Reduce`) are cooperative
  (no `block_on` in any `poll`); a `PendingOnce` harness proves cooperative
  progress. (commit `fix(iter): …cooperative`)
- **Fakes deleted:** `distributed/`, `multi_system/`, and the fake-SIMD path
  (mocks/placeholders — HARD integrity), ~2353 lines, all zero-consumer.
  `execution/`/`facade/` kept (consumer-proven live), fake tie-ins severed.
  (commit `refactor(iter): Delete fake …`)

REMAINING (own follow-up, [arch]): `AsyncIterator` is `into_vec()`-based, so
`AsyncMap`/`AsyncFilter`/`ParAsyncMap`/`ParAsyncFilter` still `block_on` inside
the synchronous `into_vec()`. Eliminating those requires redesigning
`AsyncIterator` to a streaming `poll_next`/`async fn next` surface — a breaking
public-trait change needing coordinated caller updates. Filed as ADR-018.

## ADR-018: Streaming AsyncIterator (poll_next) to remove into_vec block_on

- Status: Proposed [arch]
- `AsyncIterator`'s `into_vec()` materialize-then-process shape forces
  `AsyncMap`/`AsyncFilter` (and their parallel variants) to `block_on` the
  per-item async closure inside the synchronous `into_vec()`. The fix is a
  streaming trait (`fn poll_next(self: Pin<&mut Self>, cx) -> Poll<Option<Item>>`
  or `async fn next(&mut self)`), so adapters await natively and the terminals
  (already cooperative) drive a real stream. Breaking change: every
  `AsyncIterator` impl and caller updates in the same coordinated unit;
  the already-landed cooperative terminals are forward-compatible with it.

## ADR-019 (2026-07-03): Help-while-waiting scheduler scope (nested-scope soundness)

- Status: Accepted [arch] · Refs: ISSUE-208, concurrency_audit.md Round 20

**Context.** `ThreadScheduler::scope` fans borrowing jobs onto the unified
scheduler and blocks in `SchedulerScopeState::wait` until every scoped job
completes, keeping the stack-owned scope state alive for the jobs'
`NonNull<SchedulerScopeState>` completion tokens. `wait` spun then *parked* on a
condvar without running scheduler work. That is unsound the moment a scope is
entered from inside a running scheduled job (nested fork-join, e.g. a recursive
`moirai_iter` `drive`):

- **Deadlock (structural).** A worker that parks inside `scope` removes itself
  from the pool while its own nested scoped jobs sit unrun. With one worker this
  is an unconditional deadlock (the sole runner is the parked waiter); with `n`
  workers it deadlocks whenever every worker is simultaneously parked waiting on
  a nested scope. Reproduced deterministically: a nested `scope` on a
  one-worker pool times out at 30 s.
- **Heap corruption (empirical).** Under concurrent nested scopes the parked
  design aborted with `STATUS_HEAP_CORRUPTION` (0xC0000374) — the scope's
  stack-owned state aliased across workers while the owner made no progress.

**Decision.** Make the scope waiter *work-conserving*. `scope` calls
`drain_scope(&state)` instead of `state.wait()`:

- If the caller **is a scheduler worker** (`get_current_worker_id().is_some()`),
  it runs jobs via its own `next_job(worker_id)` (pop own deque + steal into it)
  and `execute_job` until `state.pending_tasks == 0`, spinning briefly then
  timed-parking on the scope condvar only when nothing is runnable (its
  remaining jobs are mid-flight on peers; `complete_task` wakes it). The worker
  never parks while holding runnable pending work, so the pool always has a
  runner — deadlock-free by construction — and the scope frame stays live and
  *progressing* until every borrowing job completes, closing the aliasing race.
- If the caller **is not a worker**, it parks as before: the worker pool drains
  its scoped jobs, so a blocking non-worker starves nothing.

`next_job(worker_id)` only touches the *owner's* single-owner Chase–Lev deque
(plus multi-consumer steals into it), so the help path introduces no new
cross-thread aliasing on the deques.

Indexed fan-out and indexed map/reduce create the same synchronous nested-wait
shape. They therefore use `drain_scope` as well; parking directly through
`SchedulerScopeState::wait` would bypass this decision and can deadlock a
saturated outer parallel region whose workers submit inner indexed chunks.
Their chunk count is bounded only by logical work and worker-plus-caller lanes.
Execution policy already owns the profitability decision: `Adaptive` applies
its documented threshold before reaching the executor, while explicit
`Parallel` must not be silently overridden by an index-count grain heuristic
that cannot know each index's computational cost.

**Alternatives rejected.** (b) Route `moirai_iter`'s non-indexed terminals
through the flat `for_each_indexed` fan-out — avoids nesting but leaves `scope`
itself a deadlock trap for every other nested caller; the scheduler primitive
should be sound, not the callers papering over it. (c) A dedicated blocking
thread pool for scope waiters — rejects the zero-extra-thread invariant and the
work-stealing SSOT.

**Evidence.** Red→green at the scheduler layer:
`scheduler_scope_nested_saturation_completes` (30 s deadlock → 0.01 s pass) and
`scheduler_scope_recursive_fork_join_is_sound` (the drive-shaped log2-depth
recursive fork-join, analytical arithmetic-series oracle, `W ∈ {1,2,4}`, 5×
repeat clean). Full `moirai-executor` (77) and `moirai-iter` (191) suites green;
clippy clean. Evidence tier: type/analysis (deadlock-freedom argument above) +
empirical (deterministic reproduction and repeat-clean regression).

**Follow-up.** With `scope` sound, a parallel non-indexed `drive` can be
reintroduced against this primitive with a parallelism-asserting test
(ISSUE-208 (c)); tracked separately so it lands as its own verified slice.

## ADR-020 (2026-07-13): Typed work-stealing capabilities

- Status: Accepted [major] · Refs: ISSUE-211

**Context.** `ChaseLevDeque` and the unused alternative `BlockBasedDeque` exposed owner-only
`push`/`pop` and thief-only `steal` through one `Sync` type. Cloning an `Arc` of
that type lets safe callers run multiple bottom-side operations concurrently,
invalidating the `UnsafeCell<MaybeUninit<T>>` aliasing proof. Exclusive access
to an owner endpoint also does not prove reclamation quiescence while stealer
endpoints remain alive.

**Decision.** `ChaseLevDeque` constructs one non-`Clone`, `Send + !Sync` owner
and cloneable `Send + Sync` stealers over private `Arc` storage. Owner
operations require `&mut self`; steal operations exist only on stealers. The
default `DeferredReclaim` ZST retains resized arrays until the final endpoint
drops; shared live array reclamation remains opt-in through the Moirai-owned
access-counted policy. Batch
steal returns an allocation-free owning iterator whose destructor drops an
unconsumed tail, so panic cannot leak transferred values.

Delete `BlockBasedDeque`: no production path consumes it, it duplicates the
canonical deque role, and safe node reuse requires a reclamation subsystem
solely for that unused alternative. Introducing `crossbeam-epoch` or a new
hand-rolled EBR would violate first-party ownership or add unjustified unsafe
synchronization. `SplitDeque` remains a distinct, consumed composition over the
canonical typed Chase-Lev endpoints.

The executor stores only stealers and external injectors in shared worker
state. Each worker thread owns its bottom-side endpoints on its stack. Nested
scope helping and diagnostic paths use shared steal endpoints; they never
recover owner access through TLS, raw pointers, or runtime alias checks.

**Rejected alternatives.** A mutex around the public combined deque encodes
ownership at runtime and adds hot-path contention. A thread-local raw owner
pointer recreates an unsafe aliasing contract. Treating `&mut owner` as a
quiescent reclamation proof ignores concurrently active stealers. Preserving
the unconsumed block deque via Crossbeam EBR adds an external production
substrate; implementing EBR locally for one dead API adds unsafe debt.

**Verification.** Compile-time auto-trait assertions and compile-fail doctests
encode capability separation; Loom checks bounded owner/thief interleavings;
generic contention tests check exact-once delivery; Miri checks endpoint drop
order and retired-storage lifetime. Criterion compares the endpoint migration
against the stored scheduler baseline. Evidence claims remain bounded to the
checks actually executed.

## ADR-021 (2026-07-15): Dedicated bounded lane for blocking work

- Status: Accepted [arch] · Refs: ISSUE-213

**Context.** `BlockingTask` currently supplies only a zero-sized routing marker.
Its jobs enter the same per-worker queues and worker set as synchronous and
async-ready jobs. A starvation construction with one blocking job per compute
worker prevents every compute worker from reaching a queued synchronous job.
Affinity offsets change placement but do not provide admission isolation,
backpressure, cancellation ownership, or a shutdown boundary.

**Decision.** `ThreadScheduler` owns one lazily initialized, Moirai-native
blocking lane with a bounded, per-blocking-worker synchronous queue. Ordinary
compute-only executors therefore allocate no blocking worker stacks or queue
storage. `BlockingTask` dispatch is
selected through the sealed `WorkClass` associated capability, so the public
work-class API remains zero-sized and statically routed. Blocking workers
execute the same `ScheduledJob` lifecycle boundary as compute workers, but
maintain separate pending and active counters. The scheduler's quiescence and
metrics surfaces aggregate both lanes, while compute-worker parking and
shutdown observe only compute-lane pending work. This prevents idle compute
workers from spinning behind blocking backlog.

Lane admission is non-blocking and returns typed resource exhaustion when the
selected bounded queue is full. A locality hint selects a blocking lane;
otherwise a thread-local ticket distributes submissions without a shared
round-robin atomic. Shutdown closes all lane queues, drains admitted jobs,
and joins the blocking workers. Queued task cancellation remains owned by the
existing lifecycle token: the lane drops a cancelled job only after its
worker dequeues it, so result publication and cancellation metrics retain one
canonical path.

**Rejected alternatives.** Keeping one worker set cannot prove starvation
freedom. An unbounded blocking queue violates the memory bound. A second
executor implementation would duplicate scheduling, lifecycle, and panic
containment logic. A global mutex around one blocking queue would serialize
all producers and add avoidable contention; per-worker bounded channels keep
the synchronization boundary local to admission and one receiver.

**Verification.** The implementation must provide value-semantic tests for
compute progress while every blocking worker is occupied, queue-full
backpressure, queued cancellation, graceful drain, and shutdown rejection.
Focused nextest and warning-denied Clippy are the primary evidence tier;
Criterion compares blocking admission and execution handoff after correctness
passes. No Tokio or Smol production dependency is introduced.

## ADR-022 (2026-07-25): moirai-iter nested fork-join runs on the scheduler scope

- Status: Accepted [arch] · Refs: ISSUE-219, PRs #97, #98, #99

**Intent.** Decide which runtime executes `moirai-iter`'s *nested* parallel
work — work whose jobs block on other jobs of the same runtime. Flat,
non-nesting fan-out is already decided (ADR-019: the indexed executor path);
this ADR covers the recursive fork-join shape, whose only production caller is
`parallel/sorting.rs`.

**Context.** `moirai_iter::base::ThreadPool` is an mpsc FIFO queue with a fixed
worker set and no work stealing. `execute` sends onto a channel; a worker that
blocks waiting for another job in that pool cannot run it. Its callers
nevertheless use it as a fork-join runtime: `par_merge_sort_impl` and
`par_sort_unstable_by_impl` fork one half onto the pool and block on it,
recursively.

Three defects fixed in isolation are one structural mismatch:

- **#97 (soundness).** `PoolJoinGuard::wait` discarded `recv()` results, so a
  dead worker was indistinguishable from a completed task;
  `ZeroCopyParallelIter::map` then `assume_init`-ed memory no worker wrote.
  Fixed by counting completions and asserting.
- **#98 (liveness).** The worker loop ran jobs without `catch_unwind`, so a
  panicking job killed its worker permanently. The pool shrank silently until
  `execute` queued work nobody could run, surfacing as an unrelated later hang.
  Fixed by catching the unwind in the worker.
- **#99 (starvation).** Deep enough recursion blocks every worker on a half that
  is still queued. Reproduced deterministically: a two-worker pool deadlocked at
  65,536 elements (≈1M elements on eight cores). Fixed by capping outstanding
  forks at `worker_count - 1`.

Each fix is a guard rail around the same gap: the pool has no nesting contract.
The fork budget in particular is a *global* cap — parallelism is bounded by pool
width rather than by the work tree, and every future blocking caller has to
remember an unenforced rule. Separately, the pool duplicates a role Moirai
already owns: `ThreadScheduler` is the work-stealing runtime, and ADR-019 gave
it a work-conserving nested-wait contract for exactly this shape.

**Constraints.**

1. One work-stealing runtime SSOT (ADR-019/020/021); no second scheduler, and
   no threads beyond the executor's.
2. Nested waits must be deadlock-free by construction, not by a caller-side
   budget rule.
3. Panic containment and drop safety of `merge`/`MergeGuard` are preserved; a
   comparator panic still propagates to the caller.
4. Scheduling refusal (`ShuttingDown`, `ResourceExhausted`) must not silently
   lose a branch of the work tree — a lost half is an unsorted slice.
5. No compatibility shim: the superseded path is deleted in the same change.
6. Regression tests keep the shape where a starvation regression trips
   nextest's 60 s terminate bound instead of hanging the suite.

**Options.**

*(1) Keep the pool and the guard rails.* Cheapest, already merged, and safe.
But parallelism stays capped at `worker_count - 1` outstanding forks regardless
of tree size; the rule is unenforced by types, so every future blocking caller
re-opens the defect class; and the duplicate runtime keeps attracting the audit
findings above.

*(2) Give `ThreadPool` work stealing.* Removes the starvation class at the
source. It also re-implements per-worker deques, stealing, parking, wake
handshakes, and panic containment — the precise code this audit keeps finding
defects in, and which `moirai-executor` already has under Loom and Miri
coverage. Two work-stealing runtimes in one process also over-subscribe cores.
Rejected as duplication of the scheduler SSOT.

*(3a) Route through `global().for_each_indexed`.* This is what `cache.rs` and
`iter_ops/parallel.rs` already use, and it never starves — but not because it
tolerates nesting. `for_each_indexed` *flattens*: when the caller is already a
scheduler worker or inside an indexed region
(`get_current_worker_id().is_some() || is_in_indexed_region()`,
`scheduler/data_parallel.rs`) it runs the whole `0..count` domain inline on the
current lane. Deliberate — recursively stealing unrelated outer jobs grows the
worker stack — but it means a recursive divide-and-conquer routed through it
collapses to sequential below the first level. It is the right primitive for a
flat index domain and the wrong one for a work tree. It does not solve the
parallelism ceiling; it lowers it.

*(3b) Route through `ThreadScheduler::scope` (selected).* `scope` is the
scheduler's fork-join primitive and already carries the property the pool
lacks: `drain_scope` makes a waiter that is itself a worker run queued work via
`next_job` instead of parking (ADR-019), so a nested scope cannot remove the
last runner from the pool. `SchedulerScope::spawn` takes borrowing
(`'scope`, non-`'static`) closures, and `flush()` is documented for exactly the
two-lane shape "schedule one branch, run the other on the caller lane". ADR-019
verified it against a log2-depth recursive fork-join
(`scheduler_scope_recursive_fork_join_is_sound`).

**Decision.** `moirai-iter`'s recursive fork-join runs on
`moirai_executor::global().scope::<SyncTask, _>` with `spawn` + `flush` for the
forked half and caller-lane execution for the other. `sorting.rs` drops
`ThreadPool`, `PoolJoinGuard`, `SendPtr`, and the fork budget; because scoped
jobs borrow, the raw-pointer type erasure the pool's `'static` bound forced
disappears with them. `ForkBudget`/`try_fork`/`end_fork` have no remaining
caller and are deleted from `base.rs`.

Scheduling refusal is handled at the fork site rather than propagated. The
scoped closure captures each half by unique borrow, so a job dropped before
execution leaves both halves usable on the caller:

- `Err(ShuttingDown)` / `Err(ResourceExhausted(_))` — the job was dropped
  without running and the caller-lane branch had not started, so both halves are
  sorted on the caller. This is the same admission-backpressure contract
  `for_each_indexed` already applies (inline execution of a rejected chunk).
- `Err(SpawnFailed(Panicked))` — the forked half panicked and the scope
  converted it; the caller panics, matching the pre-existing
  `PoolJoinGuard::wait` assertion and rayon's panic propagation.

Fork granularity is bounded by machine width, not by input size: a sub-slice is
forked only while it is larger than `len / (workers × 8)`, floored at the
existing sequential thresholds. Without this the recursion splits to the
threshold, so leaf count grows with input and every leaf pays for a scope —
measured below. This is not the deleted budget returning: it is a local,
static granularity floor with no global counter and no coupling to liveness.
Deadlock freedom comes from the work-conserving scope; the bound only decides
when a split stops paying for itself.

`ThreadPool` is *not* deleted here. It remains the `ShuttingDown` fallback for
the flat executor fan-outs in `cache.rs`, `iter_ops/parallel.rs`, and
`execution/parallel.rs`, which never nest and therefore never trip the
starvation class. Its documented contract narrows to "flat, non-nesting work
only"; new blocking-on-pool callers are prohibited. Removing it in favour of a
sequential fallback is filed as a follow-up, since it is a `pub` surface change
with its own migration.

**Rejected alternative within (3b).** Reusing `moirai_parallel::join_with`,
which is the same `scope`/`flush`/caller-lane shape. It cannot satisfy
constraint 4: it takes its branches by value, so a branch whose job is dropped
on `ResourceExhausted` is unrecoverable, and it `expect`s on the error — a
queue-full join panics today. Recovering a by-value closure needs a shared
slot (`Mutex<enum { Pending(F), Done(R) }>`) that the fork site does not, since
it can capture the halves by unique borrow instead. `join_with`'s admission
contract was filed as its own defect and has since landed (ISSUE-220) with
exactly that shared slot. Collapsing the sort fork site onto it is therefore
possible but not automatic: the slot costs an uncontended mutex per fork that
the reborrowing form does not, and the sort forks far more often than a
top-level `join`. That collapse is a measured decision, deferred to a
benchmark rather than assumed here.

**Failure modes.**

- *Helping-recursion stack growth.* A worker helping inside `drain_scope` may
  run an unrelated job that itself waits and helps, adding frames per nesting
  level. Bounded here by the sequential thresholds: recursion depth is
  `log2(len / 2048)` (stable) and `log2(len / 16_384)` (unstable), ≤ ~50 for any
  addressable slice. This is the risk `for_each_indexed` avoids by flattening;
  if it becomes load-bearing, the mitigation is a split-count bound derived from
  worker count, not a return to the pool.
- *Finer work tree.* Removing the budget lets the tree expand to `len/threshold`
  leaves instead of `worker_count - 1` forks, so scheduling overhead per leaf
  now matters. This was measured, not assumed, and it bound: at 4M elements on
  24 workers the stable sort (~2000 leaves at its 2048 floor) regressed while
  the unstable sort (~250 leaves at its 16,384 floor) improved. Hence the
  machine-width granularity bound above; the constant is a measured tuning
  parameter, and re-tuning it is a benchmark question, not a redesign.
- *Admission queue pressure.* Per-worker queues hold 256 jobs per priority;
  concurrent deep sorts can reach that. Covered by the caller-lane fallback
  above, which is correctness-preserving but silently sequential — it is not
  surfaced through a counter the way
  `ThreadScheduler::admission_caller_runs` surfaces the indexed path. Recorded
  as residual risk.
- *Global-executor coupling.* `par_sort` now depends on the process-wide
  executor rather than a private pool, so its parallelism follows executor
  configuration and its shutdown state. Intended (it is the point of one
  runtime), but it removes the ability to size a pool per sort in tests.

**Verification plan.**

1. Value-semantic sort tests (existing ordering, stability, by-key, duplicate,
   empty/single, panic/drop-count cases) stay green unchanged.
2. Starvation regression: a sort deep enough to exceed any plausible worker
   count, plus a sort invoked from *inside* a scheduler worker (nested through
   `for_each_indexed`), both asserting ordering. A regression cannot complete,
   so it trips nextest's 60 s terminate bound rather than hanging. Deterministic
   worker-count-1 proof stays at the scheduler layer, where ADR-019's nested
   tests own it.
3. Criterion before/after on a large `par_sort`, reported against a stored
   baseline. The parallelism claim is measured, not asserted; a regression
   blocks the change.
4. `cargo fmt --check`, `clippy --all-targets -D warnings`, `nextest`, and a
   `RUSTDOCFLAGS=-D warnings cargo doc` build for the affected packages.

**Measured outcome.** 24 workers, random `i32`, criterion sample size 10 (4 s
measurement on the large rows), before and after built and run back to back on
an otherwise idle host. Rayon's rows, whose code is unchanged, moved +3% to
+11% between the two runs — that spread is the noise floor, and anything inside
it is reported as no change.

| row | before | after | change |
| --- | --- | --- | --- |
| stable, 10 K | 91.5 µs | 57.4 µs | −45.7% |
| unstable, 10 K | 51.9 µs | 53.3 µs | no change |
| stable, 4 M | 28.43 ms | 28.06 ms | no change |
| unstable, 4 M | 33.93 ms | 30.66 ms | −9.8% |

The small-input gain is per-fork cost: a scope replaces an mpsc send plus a
per-fork completion channel. The large rows are flat to modestly better, which
is the honest reading of the parallelism claim — 23 outstanding forks already
filled a 24-worker machine, so lifting the ceiling buys throughput only where
the tree must expand further than the pool is wide. What this change delivers
at this size is the removal of the starvation class and its guard rails, not a
large speedup. `par_sort` remains ~2.5× (stable) and ~4× (unstable) slower than
rayon on the 4 M rows; that gap predates this change and is its own item.
