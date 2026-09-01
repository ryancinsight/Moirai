# ADR-005: Unified Thread Scheduler for Executor Work Classes

Status: Accepted

**Date**: 2026-05-22
**Context**: HybridExecutor scheduler replacement
**Revision**: 2026-08-28 — ADR-036 retains the 14-word inline capacity
while removing cache-line alignment that amplified every bounded injector slot.
**Revision (second)**: 2026-08-28 — scoped jobs publish completion only after
their borrowing task call frame and captures have been destroyed; Miri exposed
the prior early-publication lifetime violation.
**Revision (third)**: 2026-08-28 — an unhinted multi-batch scope selects its
base worker once and distributes successive physical batches across workers;
reselecting after each admission could route every batch to one occupied lane
and deadlock a saturated nested scope.
**Revision (fourth)**: 2026-08-31 — public indexed CPU fan-out and reduction
route through `SyncTask`; classifying them as `BlockingTask` violated ADR-021's
dedicated-lane boundary and lazily constructed blocking workers.
**Revision (fifth)**: 2026-08-31 — final external scheduler ownership is
tracked independently from worker-held state so implicit teardown drains and
joins workers without weakening their lifetime anchor.
**Revision (sixth)**: 2026-08-31 — shutdown elects one join owner, never waits
while retaining worker-handle locks, and linearizes compute admission against
worker exit through the existing pending-work publication.

### Decision

`HybridExecutor` uses one scheduler facade for synchronous, blocking, and async-ready work. Sync and async-ready jobs use the compute worker set; `BlockingTask` uses the bounded lane defined by ADR-021. Workload shape is encoded with zero-sized marker types (`SyncTask`, `AsyncTask`, `BlockingTask`) implementing a sealed `WorkClass` trait. The scheduler stores heterogeneous jobs only at the executor queue boundary because closure output types are not homogeneous at runtime.

### Rationale

- One worker set removes the prior multi-engine split between worker queues and ad hoc async polling.
- Scheduler workers retain strong ownership of shared state through task return.
  A separate atomic count tracks only external `ThreadScheduler` handles; its
  final transition invokes the existing synchronous, idempotent shutdown path.
  This avoids the ownership cycle in the former `Arc::strong_count` drop guard
  without converting workers to weak references or adding scheduling-path work.
- Concurrent shutdown callers elect one cold-path join owner. The owner moves
  compute and blocking handles out of their mutexes before joining; scheduler
  workers that lose the election return so the owner can join them, while
  external callers wait on the existing quiescence condition variable. This
  prevents same-pool and cross-lane join cycles without adding scheduler-path
  allocation or lock traffic.
- Compute admission increments the pending counter before its one shutdown
  observation. The producer and no-work worker use a SeqCst StoreLoad handshake:
  either admission observes shutdown and rolls back, or the worker observes the
  pending publication and stays alive to drain it. Blocking admission remains
  linearized by its bounded queue mutex and closed state.
- ZST work-class markers keep dispatch monomorphized before heterogeneous queue storage.
- Per-worker priority queues preserve priority ordering without cloning scheduler algorithms per task class.
- Per-task lifecycle records remove global registry lock contention from task execution start/completion paths.
- Task handles use a shared one-shot result slot instead of per-task `std::sync::mpsc` result channels.
- The public task result slot uses a single-producer atomic state machine over one initialized result cell, so completed joins avoid mutex-protected result storage while preserving cancellation and panic values.
- Public result handles wait with bounded spin followed by an explicit `WAITING` result-state and one inline registered parked thread. Completion unparks the registered waiter only, preserving single-consumer semantics without condvar wait overhead, waiter-mutex overhead, or READY/park lost wakes.
- Small scheduled closures use naturally aligned inline erased storage sized to 14 machine words. The consumed state is encoded by swapping the stored drop function to a no-op after execution, recovering one payload word without increasing the queue element footprint. Oversized or over-aligned closures allocate one typed `Box<F>` behind the same inline job trampoline instead of using `Box<dyn FnOnce>` or a separate raw-pointer heap job variant. The scheduler still has one heterogeneous erased boundary, but common small jobs avoid a heap allocation without forcing cache-line alignment through every bounded injector slot. [ADR-036](0036-natural-alignment-for-inline-scheduler-jobs.md) owns the storage-alignment decision.
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
- **Revision 2026-08-29 — the registry mutex is removed for the multi-producer case; the 2026-05 rejection is narrowed, not overturned.** The executor now holds `Arc<TaskRegistry>` with an atomic id counter and an `RwLock` over the block directory, so registration takes `&self`.
  - *What the earlier rejection established, and still establishes.* The 2026-05 evidence measured single-producer nanosecond diagnostics and concluded that "replacing the registry mutex alone is not supported by the measured cost split" — lock-only at 26.297–31.281 ns against slot initialization at 108.28–133.47 ns and timestamp publication at 161.60–177.75 ns. That conclusion is confirmed here, not contradicted: single-producer spawn throughput *falls* about 4% under this change (3.126 → 2.989 M spawns/s, disjoint ranges over three runs). An uncontended `RwLock` read acquire costs slightly more than an uncontended mutex acquire, and no single-producer benefit was expected or found.
  - *What that evidence did not cover.* It never measured multi-producer registration. Under concurrent producers the mutex does not merely cost its acquire — it serializes: measured on an 8-core pin, executor spawn throughput *declined* from 3.126 M/s at one producer to 3.082 M/s at eight, while the same scheduler reached without the registry rose from ~4.7 M/s to ~9.2 M/s. With this change executor spawn rises monotonically to 4.895 M/s at eight producers: **1.59× at eight, 1.41× at four**, with disjoint ranges across three runs.
  - *The rejection's own failure condition does not recur.* The 2026-05 allocator was rejected because `task_scheduling_overhead` regressed to 558.97–595.53 ns. That benchmark is unchanged here: 5.5257 µs baseline against 5.5348 µs, medians 0.16% apart with heavily overlapping confidence intervals. This is a noisy 10-sample benchmark and the claim is "no detectable change", not an improvement.
  - *What is preserved.* The design refused in 2026-05 allocated one `Arc<TaskState>` per task and split id allocation from registration. Neither returns: slots remain dense registry-owned blocks at one allocation per 1024 tasks, and registration remains a single call that allocates the id and claims the slot together. The benchmark contract still bans that design's markers and now also bans re-introducing the global registry mutex.
  - *Residual.* Single-producer spawn is ~4% slower, accepted for a runtime whose purpose is concurrent submission. The cost split still says slot initialization and timestamp publication dominate the per-operation path; those remain the next targets, and this revision does not claim otherwise.
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
- An unhinted scope selects one base worker before admitting its first physical batch and wraps successive batches across the worker set. An explicit locality hint remains attached to every physical batch. This keeps the scheduling decision independent of state changes caused by earlier admissions and guarantees that worker-sized coalescing can occupy distinct lanes under saturation.
- Single scoped jobs use stack-owned scope state, direct scheduler closures, and no chunk vector or wrapper closure allocation.
- Scoped task execution owns a distinct completion closure. It destroys the borrowing task and its call frame before invoking completion, and dropping an unexecuted job drops the completion token without calling it. Directly scheduled and indexed task panics reach scheduler failure metrics; a batched scoped panic remains scope-local because the enclosing physical batch completes. This ordering prevents the caller from reclaiming stack state while a worker still carries a protected borrow.
- Typed indexed fan-out uses `ThreadScheduler::for_each_indexed`, exposed through `HybridExecutor::for_each_indexed` and `Moirai::for_each_indexed`.
- The two public indexed facade operations classify their CPU-bound work as
  `SyncTask` and therefore use the compute worker set. The lower executor seam
  remains generic over `WorkClass`; `BlockingTask` remains reserved for work
  that may block, including the general scoped facade contract under ADR-021.
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
