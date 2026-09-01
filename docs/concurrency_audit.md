# Concurrency safety audit — residual risk & decision log

Append-only log of adversarial concurrency/memory-safety audit rounds. Each
round records what was fixed, what was investigated and found sound (so it is
not re-chased), and real-but-deferred items.

Run the tests that back these findings via the
[concurrency verification runbook](concurrency_testing.md) (loom models + stress
tests + the standard gate).

## Round 24 (2026-07-13) — iterator leak fix and runtime architecture audit

**Fixed (P0 resource lifetime).** Three parallel-iterator helpers moved values
out of owned vectors wrapped in `ManuallyDrop` and never deallocated the source
buffers: collect-into-existing-storage leaked one allocation, while full and
shortest interleave leaked both inputs. Existing value and `Drop` assertions
remained green because every element was moved/dropped exactly once; they could
not observe the backing allocations. Miri-nextest reported the leaks directly.
The helpers now consume owned `IntoIter`s into retained/exact-capacity outputs,
which preserves non-`Clone` movement and releases every input allocation. The
targeted Miri regressions are clean. Evidence tier: machine-checked (Miri) plus
value-semantic unit tests.

**Resolved in ISSUE-211.** `ChaseLevDeque` and the unused `BlockBasedDeque`
exposed owner-only `push`/`pop` operations as safe `&self` methods while the
deque types were `Sync`. Safe `Arc` clones could therefore run multiple owners
and race raw `UnsafeCell` slot accesses. The selected architecture is a
non-clonable Chase-Lev owner endpoint plus cloneable
stealer endpoints; all consumers must migrate atomically under an ADR. Evidence
tier: deductive aliasing/API proof; loom verification belongs to delivery.

**Resolved (P0 liveness/backpressure, ISSUE-212).** Scheduler admission uses the
fallible bounded-queue operation. Saturation returns typed resource exhaustion,
rolls back pending state, drops the rejected job once, and terminally completes
an unstarted lifecycle. A saturated one-worker regression verifies rejection
and recovery without sleeps. Evidence tier: value-semantic tests and Clippy.

**Filed (P0 starvation, ISSUE-213).** `BlockingTask` placement still executes on
the unified compute worker pool. A full worker count of blocking tasks can starve
all scheduler progress. The accepted direction is a bounded Moirai-owned
blocking lane; Tokio and Smol remain comparison references, not dependencies.

**Closed in Round 25 (resource accounting, ISSUE-214).** The prior
`ShardedResourcePool::clear` race reset counters between recycle's reservation
and bin insertion, leaving stored items hidden behind zero counters or enabling
a later decrement underflow. Round 25 supplies the linearizable ownership
protocol; stronger atomic orderings alone were insufficient.

**Runtime dependency verdict.** Rayon and Tokio remain dev-only differential
and performance references. Crossbeam remains a queue/deque oracle unless a
measured local deficit justifies adoption. `async-stream` is syntax rather than
a runtime capability; Smol duplicates scheduler ownership; Embassy applies only
to a declared embedded/no-std executor; Glommio/io_uring mechanisms belong only
at an optional Linux PAL completion-I/O boundary with typed buffer-lifetime and
cancellation contracts. No production dependency was added.

**Investigated and refuted.** `bin_index(u64::MAX) == 64` does not expose an
out-of-bounds resource-pool path in the current design: recycle rejects any item
larger than `max_bytes / 4`, and take checks a shard's retained-byte count before
indexing the requested bin. A shard cannot retain enough bytes to enter that
branch for bin 64. No bin-count change is justified.

## Round 25 (2026-07-15) — FIXED: resource-pool clear linearizability

**Root cause.** `ShardedResourcePool::recycle` reserved retained counters before
publishing into its bin, while `clear` drained bins and reset the counters as
separate operations. A clear could therefore publish zero counters while a
reserved item was still in flight, hiding the item or causing a later counter
decrement to underflow.

**Fix.** Recycle now holds the existing target-bin guard from reservation through
publication. Clear acquires every bin guard in deterministic order before
draining and resetting the shard counters, then drops guards before dropping
resources so destructors cannot re-enter a held lock. No shard-wide lock was
added to steady-state take/recycle operations.

**Evidence.** The pool-local deterministic interleaving regression pauses recycle
after reservation, waits for clear to reach the exact target bin, then verifies
clear completion and exact retrieval/counter behavior. Focused nextest passes;
package nextest passes 20/20, warning-denied all-target Clippy and rustdoc are
clean, doctests run 0/0, and the new Criterion baseline measures a 28.088 ns
median with a 27.984–28.190 ns confidence interval. No speedup claim is made
without a same-machine pre-change comparator. Provider PR #70 is ready for
review and is merged as `368acbd`; the provider integration is closed.

## Round 23 (2026-07-03) — FIXED: `join()` quiescence lost-wakeup (Dekker, AcqRel→SeqCst)

**Bug (HIGH — liveness, latent `join()` hang).** `ThreadScheduler::join` and a
worker completing the last job form a store-buffer (Dekker) handshake across two
atoms that was ordered AcqRel/Acquire, not SeqCst:
- worker (`execute_job`): `active_workers.fetch_sub(1, AcqRel)` (→0, publishes
  quiescence) then `notify_quiescent` `join_waiters.load(Acquire)`.
- joiner (`join`): `join_waiters.fetch_add(1, AcqRel)` then `is_quiescent`
  `active_workers.load(Acquire)`.

AcqRel gives no StoreLoad barrier, so the store-buffer outcome is permitted: the
joiner reads a stale `active != 0` and parks on the condvar while the worker
reads a stale `join_waiters == 0` and never signals — the quiescent scheduler
never wakes the joiner, hanging `join()`. The joiner's 256-iteration pre-register
spin makes the window narrow but not empty. This is the same hazard the park/wake
handshake (Round 19 / `loom_wake_handshake.rs`) closed with SeqCst; the join path
had been left at AcqRel.

**Reproduced:** `tests/loom_join_quiescence.rs` modeling the production
orderings — loom reports the lost wakeup reachable with AcqRel, unreachable with
SeqCst (+ the loom StoreLoad `fence`, the same modeling device the wake-handshake
uses).

**Fix.** The four handshake accesses are now SeqCst so they share one total order
that forbids the store-buffer outcome:
`execute_job` `active_workers.fetch_sub(1, SeqCst)`; `notify_quiescent`
`join_waiters.load(SeqCst)`; `is_quiescent` `pending_tasks`/`active_workers`
loads SeqCst; `join` `join_waiters.fetch_add(1, SeqCst)`. The hot-path cost is
the completion-side `fetch_sub` barrier — free on x86 (`lock sub` is already a
full barrier); SeqCst loads are plain `mov` on x86. The committed
`loom_join_quiescence.rs` models the fixed (SeqCst) protocol and is pinned to
stay in sync. moirai-executor suite (80) green, clippy clean. Evidence tier:
machine-checked (loom) + type/analysis (memory-model derivation).

## Round 22 (2026-07-03) — loom model of the `LifoSlot` take-side exclusion

**Added:** `tests/loom_lifo_slot.rs` (`#![cfg(loom)]`) — exhaustive-interleaving
model of the per-worker `LifoSlot` protocol (`schedule/runtime/types.rs`). The
slot is `unsafe impl Sync` and moves jobs with `ptr::read`/`MaybeUninit`, so its
load-bearing safety property is that a READY job is consumed by **exactly one**
taker: two takers would `ptr::read` the same `ScheduledJob` twice — a
double-move / use-after-free / double-free, the heap-corruption class this
crate's nested-scope work has fought. `LifoSlot::pop`/`steal` are also on the
ADR-019 `drain_scope` help path (`next_job`/`steal_job`), so this slot's
soundness underwrites the help-while-waiting fix.

Three models, all green (loom enumerates every interleaving; its `UnsafeCell`
flags any concurrent access):
- `pop_and_steal_take_the_job_exactly_once` — owner `pop` vs thief `steal`:
  exactly one takes the job, never both (double-`ptr::read`), never neither.
- `concurrent_steals_take_the_job_exactly_once` — two thieves race the `2->3`
  CAS; exactly one wins.
- `replace_push_and_steal_conserve_both_jobs` — owner replace-`push` vs `steal`:
  the old and new jobs are conserved exactly once each (evicted/stolen/resident),
  never aliased.

Evidence tier: machine-checked (loom exhaustive model) — the strongest available
for this mutual-exclusion property, upgrading it from the hand analysis in the
`LifoSlot` SAFETY comments. Orderings in the model mirror production (empty push
CAS 0->1 Acquire, replace CAS 2->1 AcqRel, pop CAS 2->1 Acquire, steal CAS 2->3
Acquire; publishing `store(2, Release)` pairs with the take CAS Acquire; taker
`store(0, Release)` pairs with the next push's 0->1 Acquire) and are pinned to
stay in sync. Run: `RUSTFLAGS="--cfg loom" cargo test -p moirai-executor --test
loom_lifo_slot --release`.

## Round 21 (2026-07-03) — NUMA-aware steal review; help-while-waiting adversarial tests

**Reviewed:** commit `bcaf0bf` (NUMA-aware two-pass victim selection in
`steal_job`). Verdict: **no correctness defect.** `worker_numa_nodes` is a
`Box<[Option<usize>]>` set once at construction and read-only thereafter (no new
locks/atomics/shared mutation). Pass 1 prefers same-NUMA-node victims; Pass 2 is
the original full-ring randomized scan, so every victim with work is still
reachable — coverage and starvation-freedom are preserved. `steal_batch` into
the thief's own local queue is unchanged, so the help path added in ADR-019
(`drain_scope` → `next_job` → `steal_job`) inherits the same aliasing guarantees.
Regression-checked: `scheduler_scope_nested_saturation_completes` and
`scheduler_scope_recursive_fork_join_is_sound` stay green (0.02 s) on the merged
HEAD, ×4 repeat.

**Perf note (not a correctness issue, deferred — needs NUMA hardware).** On a
multi-node system where a thief's same-node victims are momentarily empty (the
common state right after a fork-join barrier, when same-node peers have just gone
idle), `steal_job` runs Pass 1 (O(worker_count), all misses) *then* Pass 2
(O(worker_count)) — a 2× steal scan during the post-barrier steal storm. Pass 1
also scans the full ring and `continue`s past off-node workers rather than
iterating a precomputed per-node victim list (O(node_size)). Both are load-
balance-neutral but worth a criterion pass on real NUMA hardware; filed as a
follow-up perf item (external blocker: no multi-socket machine in CI).

**Test-coverage gap (filed).** The NUMA change ships no `moirai-executor` test.
On single-node/`None`-topology CI (VMs, containers) Pass 1 is skipped or covers
only the all-same-node case, so the **cross-node fallback** (Pass 1 empty → Pass
2 steals across nodes) is never exercised. A deterministic white-box test needs a
node-assignment injection seam on the scheduler constructor — owned by the NUMA
author; filed as ISSUE-209 rather than added here to avoid editing that hot file
under concurrent authorship.

**Resolution (2026-09-01).** `MOI-SINGLE-NODE-STEAL-SCAN-2026-09-01` adds the
test-only assignment seam and a synchronized three-worker regression. The sole
free worker first misses an empty same-node peer, then executes a target queued
behind an occupied cross-node victim. This closes ISSUE-209 without sleep or a
platform NUMA dependency.

**Added (this round):** `scheduler_scope_nested_panic_propagates_and_pool_survives`
— adversarial guard that a panic in a nested scoped job under help-while-waiting
surfaces as `SpawnFailed(Panicked)` from the nested scope, its sibling still
runs, and the outer scope completes without deadlock/corruption. `W ∈ {1,2,4}`,
green.

## Round 20 (2026-07-03) — nested scope unsoundness; parallel `drive` reverted

**Finding (HIGH — memory safety).** An attempt to make
`moirai_iter::parallel::ParallelIterator::drive` genuinely parallel by fanning
its recursive `Consumer` split through `moirai_parallel::join_with::<Parallel>`
(which uses `ThreadScheduler::scope` → `spawn` + `wait`) is **unsound under
nesting** and was reverted.

Two independent problems:
1. **Deadlock by construction (park-without-help).** `SchedulerScopeState::wait`
   (`schedule/runtime/types.rs`) spins then parks on a condvar; the waiting
   thread does *not* participate in work-stealing. A recursive/nested
   fork-join therefore deadlocks whenever spawned `left` tasks and their
   waiters exhaust the pool — provably so with a single worker: the main thread
   parks awaiting its spawned branch while the sole worker parks awaiting its
   own nested branch, both awaited tasks un-stolen, no runner.
2. **Heap corruption under concurrent nested scopes (empirical).** A
   nested-saturation probe (an outer parallel drive whose every element runs an
   inner parallel drive) aborted with `STATUS_HEAP_CORRUPTION` (0xC0000374) in
   ~0.16 s — not a hang. Single-level fork-join drive was fine; the corruption
   appears only when many inner `scope`s run concurrently inside outer scope
   jobs, indicating a data race in the scope machinery under nesting (the
   `NonNull<SchedulerScopeState>` handed to scoped jobs, the per-scope job
   buffer, or reentrant `schedule_job`).

**Action.** Reverted the parallel drive; `drive` is sequential-by-contract again
(documented on the trait method in `moirai-iter/src/parallel/traits.rs`). Added
`nested_iteration_produces_correct_values` as a value-semantic regression guard
(nested iteration must stay correct). The happens-before edge in `join_with`
itself is sound (`complete_task`'s `AcqRel` decrement pairs with `wait`'s
`Acquire` load), so single-level result publication is race-free — the
unsoundness is scope *nesting*, not the single-level join.

**RESOLVED at the scheduler layer (ADR-019, ISSUE-208 option (a)).**
`ThreadScheduler::scope` now waits via `drain_scope`: a worker-thread waiter is
work-conserving (runs its own `next_job` + `execute_job` until the scope drains,
timed-parking on the scope condvar only when nothing is runnable), so a worker
never parks while holding runnable nested work — deadlock-free by construction,
and the scope's stack-owned state stays live *and progressing* until every
borrowing job completes, closing the aliasing race. A non-worker waiter still
parks (the pool drains its jobs). `next_job(worker_id)` touches only the owner's
single-owner deque, adding no new cross-thread deque aliasing.

Reproduction (red→green, `moirai-executor` tests):
- `scheduler_scope_nested_saturation_completes` — nested scope inside a scoped
  job; 30 s deadlock (W=1) → 0.011 s pass across W ∈ {1,2,4}.
- `scheduler_scope_recursive_fork_join_is_sound` — the drive-shaped log2-depth
  recursive fork-join (analytical arithmetic-series oracle), W ∈ {1,2,4}, 5×
  repeat clean (guards the nondeterministic heap-corruption path).

Full moirai-executor (77) + moirai-iter (191) suites green; clippy clean.

**Remaining (ISSUE-208 (c), separate slice).** Reintroduce a parallel
non-indexed `drive` against the now-sound `scope`, with a parallelism-asserting
test. Until then the non-indexed `parallel/` surface stays sequential;
scheduler-owned parallelism is `Moirai::for_each_indexed` / `map_reduce_indexed`.

ISSUE-217 extends the same accepted wait contract to those indexed operations.
RITK exposed a saturated production shape: parallel CMA candidate evaluation
nests masked-histogram indexed reductions. The executor first removes its
hidden 256-index grain floor so explicit `Parallel` policy schedules the 9–18
expensive candidates; adaptive profitability remains owned by `Adaptive`.
Caller-originated indexed regions use `drain_scope`, while worker-nested regions
flatten onto the current outer lane to bound stack depth without losing outer
parallelism;
`nested_indexed_saturation_completes` synchronizes two outer workers before
nesting and asserts fan-out plus map/reduce against the closed-form
arithmetic-series sum. Balanced quotient/remainder bounds ensure every selected
worker-plus-caller lane receives a physical chunk when count exceeds the cap.

## Round 19 (2026-06-28) — lock-hold contention sweep + loom-modeled wake handshake

Reduced per-operation lock-hold across the async sync primitives and the core
histogram, and model-checked the scheduler's lost-wakeup handshake.

### Reduced — O(n) → O(log n) waiter/receiver scans (moirai-async)
Every `moirai-async` sync primitive kept its waiters/receivers in a linear
`Vec`/`VecDeque` of `(id, …)` and did `position`/`find`/`retain`-by-id on each
poll/drop, holding the single state `Mutex` for O(n) — so lock-hold grew with
waiter/subscriber count. Re-keyed each by a monotonic id in a
`BTreeMap<u64, _>` (O(log n) lookup/remove; in-order iteration stays FIFO since
ids are monotonic, so all fairness semantics hold): `Notify`, `Semaphore`,
`RwLock` (both read/write queues), `Watch`, `Broadcast`. The subtle
state-machine regressions (permit storage/transfer, reader-batch grant,
cancellation waker-clear) all pass.

### Reduced — histogram hot-path atomics 3 → 2 (moirai-core)
`Histogram::record` did three contended `fetch_add`s per sample (a bucket,
`sum`, `count`). `count` is redundant — each record increments exactly one
bucket, so `count == Σbuckets` always — so it was removed and `count()` now
derives from the bucket sum. Drops a globally-contended atomic from every
record and removes the `sum`/`count` false-sharing. SSOT (per-bucket counts
are the single source of the sample count).

### Verified sound — the SeqCst park/wake handshake (closes the deferred M3 risk)
The `pending_tasks` ↔ idle-bitset Dekker handshake (`scheduler/core.rs` +
`idle.rs`) carried a deferred "can the `SeqCst` be weakened to cut the barrier
cost?" question. Added a `cfg(loom)` exhaustive model
(`tests/loom_wake_handshake.rs`; the crate now wires the `loom` dev-dep +
`unexpected_cfgs` lint mirroring `moirai-scheduler`) of the four-access
ordering. Finding: with the SC StoreLoad barrier present no interleaving loses
a wakeup; weakening any access to Acquire/Release readmits the store-buffer
outcome (lost wakeup). The `SeqCst` is therefore **necessary and sufficient** —
it cannot be weakened (no contention win there), and an explicit production
fence is **unwarranted** (x86's `lock` RMW already fences; bare `SeqCst` atoms
carry the guarantee per the Rust memory model).

loom caveat (documented in the test): loom does not fully model the SC total
order for bare `SeqCst` atomics in the store-buffer shape — it needs an
explicit `fence(SeqCst)` to represent the StoreLoad barrier (same device as
`loom_chase_lev.rs`). The fence lives in the model only, not in production.

### Investigated and rejected — H1 (per-spawn registry `Mutex`)
H1 proposed sharding/lock-freeing the per-spawn `Mutex<TaskRegistry>`. A
soundness+performance review closes it as **not viable**, on two independent
grounds:

1. **Already A/B-tested and rejected (authoritative).** A concurrent
   registry allocator (`ConcurrentTaskRegistry` + `next_task_id: AtomicU64` +
   `allocate_task_id`) was previously implemented, benchmarked, and rejected:
   it regressed `task_scheduling_overhead` (558–595 ns vs the mutex baseline)
   while only marginally helping the isolated spawn/join number — see
   `docs/adr/0008-scheduler-route-consumption-and-transport-ownership-boundary.md`
   (the "Lock-free registry allocator A/B" and "rejected after a
   scheduling-gate regression" entries). The lock is **not** the bottleneck:
   per the ADR attribution it is ~26–31 ns (lock-only) / ~44–50 ns (full
   mutex registration), while slot initialization (108–133 ns) and lifecycle
   timestamp publication (161–177 ns) dominate. `source_contracts.rs`
   (`executor_registry_registration_rejects_regressed_lock_free_allocator`)
   enforces the rejection; a sharded reimplementation in this round failed it
   as designed and was reverted (not test-weakened).
2. **The real next target is slot-init / timestamp publication**, not the
   lock (ADR). Any future registry work should attack those, keeping the
   accepted dense-block `Arc<Mutex<TaskRegistry>>` shape.

### Fixed — registry slot-aliasing UB (was pre-existing, both borrow models)
miri (Stacked **and** Tree Borrows) reported UB in the registry: any `&mut`/`&`
spanning the `Box<[Option<TaskState>]>` slot slice (`cleanup_completed`'s
`&mut *block.slots`, and `register_task_with_id`'s `&mut`-IndexMut) retags the
whole slice and invalidates a live `TaskLifecycleToken`'s `NonNull<TaskState>`
into a sibling slot. miri only *caught* it in the single-threaded cleanup test
(the threaded executor is not run under miri), but the same `&mut`-slice retag
sits on the production register path, so it was a real latent defect, not
test-only.

Fix: slots are now `Box<[UnsafeCell<Option<TaskState>>]>`, with all slot access
encapsulated in `TaskStateBlock` accessors (`get`/`insert`/`clear`/`states`)
that touch a slot only through its own `UnsafeCell` raw pointer — never a
`&mut`/`&` over the slice. The token pointer is derived from a *shared* view of
the freshly-written state (`NonNull::from((*cell).as_ref()…)`), not a `&mut`,
because the token only ever reaches the state's interior-mutable (atomic/mutex)
fields; a `&mut`-derived pointer is what Tree Borrows disables on later shared
reads. `UnsafeCell` is zero-cost: same dense inline layout, address stability,
and no per-task allocation, so the ADR's accepted dense-block policy and its
benchmark contracts hold (contract source strings updated to the accessor
shape, not weakened). Verified: `cargo miri test -p moirai-executor
registry::tests` passes under **both** `-Zmiri-stacked-borrows` (default) and
`-Zmiri-tree-borrows`; full workspace green.

## Round 18 (2026-06-23) — built the deferred Windows readiness reactor

Completed the last big deferred component: a real readiness reactor on Windows.

### Added — `WsaPollReactor` (`windows/poll.rs`)
The IOCP backend signals completions of *posted overlapped ops*, not socket
*readiness*, so it could never wake the readiness-based `net.rs` futures; Windows
async sockets fell back to a 100%-CPU busy-poll. Replaced IOCP with a `WSAPoll`
reactor (the Windows `poll(2)` analogue), which reports per-socket
readable/writable and is therefore the right signal. Key properties:
- **Level-triggered** by nature → self-heals the same lost-edge race the
  epoll/kqueue level-triggered fix closed (round 17).
- **Self-cleaning**: a socket closed without `unregister_fd` surfaces as
  `POLLNVAL` and is dropped from the interest set, so a stale fd never wedges the
  poll loop. (No net.rs deregistration plumbing needed.)
- **Wake**: a loopback UDP socket interrupts a blocking `WSAPoll` promptly on a
  new registration or shutdown.

`get_active()` now activates the process-global reactor on **every** platform
(epoll/kqueue/WSAPoll), so Windows async I/O is driven by real readiness instead
of busy-polling. `IocpReactor` was removed (dead/broken). Concurrent
`poll_events` from the reactor thread and executor workers is safe (stateless
WSAPoll + locked interest map + single-take wakers), matching pre-existing Linux
behaviour.

### Verification (all on Windows — the live platform)
Full workspace 758 tests pass with the reactor active; the entire async net suite
(loopback, backpressure, readiness-before-data, cancellation, EOF) now routes
through the reactor; plus direct reactor tests for readiness delivery, wake
interruption, and `POLLNVAL` self-cleaning.

### Remaining note (not a defect)
The reactor's `process_pending_tasks` uses a noop waker, but its task queue
(`reactor.spawn`) has no production callers — the live path is fd-readiness →
executor waker, so this is moot. A real Arc-waker would only matter if reactor
task-spawning is ever used.

## Round 17 (2026-06-23) — completed the deferred pal reactor readiness fix

Resolved the big round-16 deferred item: the pal I/O reactor lost-edge model.

### Fixed
- **epoll/kqueue lost-edge hang (Linux/BSD)** — switched both backends from
  edge-triggered (`EPOLLET` / `EV_CLEAR`) to **level-triggered**. The reactor
  registers a waker only after a `WouldBlock`, so an edge-triggered fd dropped any
  readiness arriving in the register window or already present at registration →
  permanent hang. Level-triggered re-reports readiness every `epoll_wait`/`kevent`
  until the I/O consumes it, which self-heals that race and the
  `unregister`+`register` interest-widening window — fixing the bug **without** a
  core.rs change (the net.rs futures register the executor's real waker, which the
  reactor then wakes). (`unix/epoll.rs`, `unix/kqueue.rs`)
- epoll/kqueue timeout `as c_int`/`as time_t` truncation → saturating casts.

### Verification note
This Windows/MSYS2 host cannot compile-check the `#[cfg(unix)]` epoll/kqueue code
(the MSYS2 rustc has no cross std and shadows rustup). Verified instead by: (1) the
changes are trivial (removing flags from OR-expressions; saturating casts); (2) the
non-trivial `libc::c_int::MAX`/`libc::time_t::MAX` cast syntax was compiled+run via
a standalone snippet; (3) level-triggered semantics are textbook; (4) the full
Windows workspace (755 tests) still passes. Linux/BSD CI confirms compilation.

### Resolved as non-issues (no change needed)
- **cache.rs `ZeroCopyParallelIter::map` panic-leak** — the type is **dead code**
  (declared only to satisfy a benchmark source-contract, never constructed), so the
  leak is unreachable; and even if reached it is memory-safe and panic-path-only
  (matches rayon). A full fix needs per-chunk owned `Vec<R>` (hot-path allocation
  cost) — not worth it for unreachable code.

### Remaining DEFERRED (genuine major feature, Windows-only)
- A readiness-capable **Windows reactor** (AFD/`\Device\Afd` polling) so Windows
  async sockets don't busy-poll via the fallback. The IOCP backend cannot deliver
  readiness (only overlapped-op completion); Windows currently returns `None` from
  `get_active()` and uses the working cooperative fallback. This is an ADR-worthy
  feature, not an incremental fix.

## Round 16 (2026-06-23) — platform layer (pal reactor/timer), iter/utils unsafe

Fanned out adversarial audits over the previously-uncovered `moirai-pal` (I/O
reactor + epoll/kqueue/iocp event loops + timer) and the `unsafe` in `moirai-iter`
/ `moirai-utils`. The iter/utils parallel-fan-out and SIMD came back essentially
clean (disjoint-by-chunk raw pointers; SIMD uses unaligned loads + runtime ISA
gates; sort/merge panic-safe via `MaybeUninit` + restore guards).

### Fixed
- **pal `with_active` UAF-on-panic** — a panic in the closure skipped the manual
  thread-local restore, leaving a dangling reactor pointer a later `get_active()`
  would dereference. Now an RAII guard restores on unwind. (`reactor/tls.rs`)
- **pal `Timer::new` overflow panic** — `Instant::now() + Duration::MAX` panicked;
  now clamped + `checked_add`. (`timer.rs`)
- **iter `numa_free` munmap-on-heap UB** — the mmap→global-alloc fallback made the
  free path probe with `munmap` on a possibly-heap pointer (UB). Linux `numa_alloc`
  now returns null on mmap failure (mmap-only), so `munmap` is always correct.
- **utils `prefetch_range_read` address overflow** — saturating/checked arithmetic.
- **async `io::compat`** — feature-gated `tokio-compat`-only imports (warning-clean).

### Verified clean / non-issues
- `moirai-iter` (base/cache/sorting/sources/pair/prefetch/parallel) and
  `moirai-utils` (queue/atomic/simd/memory) + `moirai-parallel` (ops) — no UB,
  data race, or double-free. SIMD `target_feature(avx2)` calling `_mm_dp_ps`
  (sse4.1) is sound (Rust's `avx2` transitively implies `sse4.1`).

### Real but DEFERRED (significant, Linux/non-Windows; not the live platform)
- **pal I/O reactor readiness model** — two independent audits flag that the
  edge-triggered (`EPOLLET`) registrations with a register-after-`WouldBlock`
  pattern have a lost-edge window (readiness arriving between the failed syscall
  and `epoll_ctl` is never redelivered → hang), and that the Windows IOCP backend
  cannot deliver *readiness* (it only completes posted overlapped ops). **On
  Windows `get_active()` returns `None`**, so the live platform uses the busy-spin
  fallback and never hits these — but the Linux/BSD reactor needs a level-triggered
  (or re-check-after-arm) redesign + a real `Arc`-based waker through
  `process_pending_tasks`. This is an ADR-worthy reactor rearchitecture.
- epoll/kqueue timeout `as c_int`/`as time_t` truncation (saturate) — Linux/BSD
  only; deferred because they can't be compiled/verified on this Windows host.
- `iter/cache.rs` `map` leaks already-computed results if the user closure panics
  (cross-thread disjoint `MaybeUninit` writes make cleanup-tracking intricate).

## Round 15 (2026-06-23) — completed the remaining transport mocks

Resolved the round-14 newly-discovered mock stubs. Key clarification: the
codebase's "serialization" is a hand-rolled **rkyv-*style*** archive system
(`safe_channel.rs`: `ArchiveSerialize`/`ArchiveView`/`ArchivedMessage`), NOT the
`rkyv` crate.

- **`UniversalChannel`/`UniversalSender`/`UniversalReceiver` mocks** → removed.
  Their `send`/`recv` ignored the argument and returned `Closed`; a channel
  generic over arbitrary `Send` `T` cannot serialize without a serialization
  bound — which is exactly what the **already-working, tested**
  `ArchivedUniversalSender<T: ArchiveSerialize>`/`ArchivedUniversalReceiver<T:
  ArchiveView>` provide. Surfaced those at the transport crate root and the
  `moirai` umbrella as the canonical typed channel; removed the mock test (and its
  commented-out assertion).
- **`IpcTransport` stub** → replaced with a **real** shared-memory IPC transport
  (`src/ipc.rs`) backed by `moirai_core::ipc::SharedQueue<IpcFrame>` (a Pod
  length-prefixed frame, ≤ ~4 KiB/message). Used directly (not in
  `TransportManager`, to avoid the `Local`-address overlap with
  `InMemoryTransport`); Unix/Windows only. 3 tests: in-process FIFO round-trip,
  cross-handle attach (two handles = two processes), oversized/non-local rejection.

### Real but DEFERRED (next-round candidates)
- A real IPC transport selectable through `TransportManager` would need a distinct
  `Address::Ipc` variant (rippling ~41 `Address::` match sites) and a cross-process
  segment-creation protocol (`SharedQueue::create` re-inits metadata, so two
  concurrent first-touch creators race). `IpcTransport` today is direct-use and
  single-frame; multi-frame fragmentation + manager integration is an ADR-worthy
  feature.
- `moirai-async` `io/traits.rs`/`io/mod.rs` carry pre-existing conditional unused
  imports that warn under reduced feature sets (clean under `--all-features`).

## Round 14 (2026-06-23) — deferred items from round 13 completed

All three named round-13 deferred items implemented with documentation and tests.

- **transport `MessageRouter`/`ConnectionManager` stubs** → made real.
  `MessageRouter<T: Transport>` routes published messages to subscriber addresses
  through a shared transport (was a throwaway `InMemoryTransport` per send that
  dropped every message); added `unsubscribe`/`subscriber_count`. `ConnectionManager`
  gained `state`/`is_connected`/`connected_addresses` so tracked state is usable.
  4 new value-checked tests. (`transport/lib.rs`)
- **`unified_channel` send-drops-on-Full + duplicate logic** → `send` now
  delegates to `try_send` (DRY: ~45 lines of duplicated overflow logic removed)
  with the consume-on-`Full` contract documented and `try_send` as the recovery
  path; `overflow_count` documented as advisory (a missed drain never loses a
  message — `recv` pops directly from the overflow queue). New test. (`unified_channel/core.rs`)
- **no `loom` wired** → added a faithful loom model of the Chase-Lev steal/pop
  protocol (`tests/loom_chase_lev.rs`, `cfg(loom)`-gated) mirroring the exact
  production atomic orderings; loom exhaustively verifies exactly-once across all
  pop/steal interleavings. `loom` is a `cfg(loom)`-only dev-dep (normal builds
  unaffected). Run: `RUSTFLAGS="--cfg loom" cargo test -p moirai-scheduler --test loom_chase_lev`.

### Newly discovered / DEFERRED
- `moirai-transport` `UniversalSender::send`/`UniversalReceiver::recv` and
  `IpcTransport` are still mock stubs (return constant errors, ignore input).
  Making them real needs a serialization design (the archive/`payload` path) or
  an IPC backend (could reuse `moirai-core::ipc::SharedQueue`); the
  `test_universal_channel` has a commented-out assertion. Out of this round's
  named scope — either implement or remove (breaking, needs an ADR).

## Round 13 (2026-06-23) — broadened audit (async sync, security, networking, transport)

Fanned out adversarial audits over previously-uncovered areas. The async
executor/waker/timer/result_slot were independently verified **sound** (one
fragility note: `AsyncTask`'s `Sync` rests on an external `future_lock` decoupled
from the `UnsafeCell` — consider folding into `Mutex<ErasedTaskFuture>`).

### Fixed (7 new tests; workspace 746 pass)
- **http codec OOM (untrusted input)** — `read_response` sized allocations from
  wire-supplied Content-Length / chunk-size / EOF / header lengths with no cap.
  Added a `max_response_bytes` budget enforced at the single `fill()` chokepoint
  + upfront Content-Length rejection. (`moirai-http/codec.rs`, `lib.rs`)
- **security auditor unbounded memory + O(n²)** — `Vec` + per-insert O(n)
  `retain`, bounded only by a week/month retention → millions of entries under a
  spawn storm. Now `VecDeque` + hard count cap + front-pop eviction +
  `checked_sub`. (`security/auditor.rs`)
- **broadcast missing `Drop`** — cancelled `BroadcastRecv` left a stale waker
  (spurious-wake + retention); added the `Drop` clearing it. (`sync/broadcast.rs`)
- **transport unbounded blocking** — accepted/connected TCP streams had no
  read/write timeout; a stalled peer pinned the thread. Added `NETWORK_IO_TIMEOUT`
  on `NetworkTransport` and the remote-task server. (`transport/network.rs`,
  `remote_task/server.rs`)
- **rate-limiter numerical hardening** — saturating window-advance arithmetic +
  documented approximate semantics. (`security/limiter.rs`)

### Investigated and found SOUND (do not re-chase)
- **`block_based.rs` (round-11 "non-shippable" claim) — REMOVED BY ADR-020.** The
  `head != tail` fast path is correct (stealers touch only the head block via
  `top`, owner pops the tail block; disjoint while unequal). `bottom` has one
  writer (the single owner) + Acquire/Release reader stealers. The old safe API
  failed to encode that single owner, and `&mut` access to one owner did not
  prove cloned stealers quiescent. No production consumer justified retaining a
  second deque plus a separate reclamation subsystem, so ADR-020 deletes it.
- **limiter line-107 "overflow panic" — FALSE on 64-bit** (`(elapsed/dur)*dur ≤
  elapsed < u64::MAX`); clear-loop window logic is correct. Only the TOCTOU
  (inherent to a multi-counter sliding window, approximate by design) is real.
- **async executor / waker / result_slot / timer driver — SOUND** (uses
  `std::task::Wake`/`Arc`, not a hand-rolled `RawWakerVTable`; state machines and
  park/notify orderings verified).

### ISSUE-211 verification (2026-07-13)

Typed deque endpoints pass scheduler nextest 23/23, executor nextest 80/80, and
benchmark contracts 69/69. Compile-fail doctests pass 2/2; Clippy and rustdoc
are warning-clean for the touched crates. The one-owner/one-stealer Loom model
passes, and Miri validates inline-item destruction, unconsumed batch-tail
destruction, and storage lifetime after owner drop with default features
disabled. Default-feature Miri stops at Mnemosyne's unsupported Windows
`GetEnvironmentVariableA` call before deque execution; this is a tooling limit.
Criterion medians for 256 elements are 965.64 ns deferred reclamation, 5.4225
us shared epoch, and 5.7232 us split deque. These are empirical measurements,
not cross-machine performance proofs.

### Real but DEFERRED (next-round candidates)
- `moirai-transport` `MessageRouter`/`ConnectionManager` (lib.rs): unwired,
  untested stubs; `publish` builds a throwaway `InMemoryTransport` per send and
  silently drops the message (the tested router is `moirai-core`'s). Redesign to
  a shared transport or remove — breaking public API, needs an ADR.
- `moirai-core` `unified_channel`: `overflow_count` is an advisory mirror read
  outside the lock (drain liveness); `send` drops `T` on `Full` instead of
  returning it. Document or restructure.
- No `loom` model checking is actually wired (only referenced in a comment) —
  the lock-free deques rely on hand-proof + stress tests.

## Round 12 (2026-06-23) — deferred items from round 11 completed

All seven round-11 deferred items implemented, each with documentation and
value-checked tests (13 new tests; workspace 739 pass).

- **net listener accept-cancel reservation leak** — RAII `ReservationGuard`
  releases the reservation on any early exit (error or future cancellation),
  disarmed only after `add_connection_reserved`. (`listener.rs`)
- **net `TcpStream::drop` leak + addr collision** — connections keyed by unique
  `ConnectionId`; `Drop` untracks by the id captured at accept, never re-querying
  a possibly-reset socket. (`types.rs`, `stream.rs`, `listener.rs`)
- **`resource_pool.recycle` capacity overshoot** — reserve-before-insert so
  concurrent recyclers evict on a shared, fresh view; bounded by `saturating_sub`.
  (`resource_pool.rs`)
- **ipc `SharedQueue` hardening** — zero-capacity, size-overflow, and over-aligned
  `T` rejected via `layout_for`; `T: bytemuck::Pod` closes the invalid-bit-pattern
  read; capacity recorded in the header and validated by `open`. (`ipc/queue.rs`)
  NOTE: the round-11 `memory.rs` munmap concern was a **false alarm** — `mmap` and
  `munmap` use the same `self.size`, so the lengths are consistent.
- **timer `wheel.rs` tombstone growth** — `active` membership index makes `cancel`
  a no-op for non-live ids, preserving `cancelled ⊆ active`. (`timer/wheel.rs`)
- **`adaptive.flush_batch` unbounded spin** — bounded no-progress retries, then
  requeue and return `WouldBlock` backpressure. (`zero_copy/adaptive.rs`)
- **hybrid channel `Send` soundness cliff** — documented SAFETY on the manual
  impls plus a self-validating compile-time guard that makes a future `Clone` on
  either SPSC half a build error. (`channel/hybrid.rs`, `communication/ring_buffer.rs`)

## Round 11 (2026-06-23)

### Fixed (this round)
- **Executor producer/worker lost-wakeup** — `schedule_job` incremented
  `pending_tasks` with `Release`, breaking the SeqCst store-buffer handshake
  with a parking worker (`idle_workers` set + `pending_tasks` load, both SeqCst).
  Now `SeqCst`, kept before the queue push (so `execute_job`'s decrement cannot
  underflow). `core.rs`.
- **Executor id≥64 wake stall** — single `AtomicU64` idle map could not address
  workers beyond bit 63; on >64-worker pools they were unreachable by the wake
  lottery and stayed parked under load. Replaced with multi-word `IdleBitset`
  (`idle.rs`); every worker registers (unified `wait_for_work`).
- **`FutexMutex` non-Linux/Windows fallback lost-wakeup** — `unlock` stored
  `locked` (Release) then read `waiters` (Relaxed); StoreLoad reorder skipped a
  concurrently-registering waiter. Added paired `SeqCst` fences.
- **NUMA backoff overflow** — `record_failure` exponential delay now
  `saturating_mul`.

### Investigated and found SOUND (do not re-chase)
- **`moirai-scheduler/deque/chase_lev.rs` batch stealing — CORRECTED BY
  ADR-020.** CAS-failure speculative copies were non-dropping, but callback
  panic after a successful multi-item transfer leaked the unconsumed tail.
  `StolenBatch` now owns and drops that tail.
- **`chase_lev.rs` retired-array reclamation — CORRECTED BY ADR-020.** `&mut`
  owner access did not prove cloneable stealer endpoints quiescent. Default
  `DeferredReclaim` retains retired arrays until the final endpoint drops.
- **`moirai-core/scheduler/deque.rs` grow/Drop double-free** — claimed the
  retired buffer's moved-out slots get re-dropped. FALSE: `Buffer::Drop` only
  deallocates and never drops elements; the live `[top,bottom)` range is dropped
  exactly once from the current buffer. Retired buffers carry no element drops.

### Real but DEFERRED (next round candidates, ranked)
- `moirai-async/net/listener.rs` `accept`: connection reservation
  (`try_reserve`) leaks if the accept future is cancelled before commit — wrap in
  an RAII guard that releases on drop, disarmed only after `add_connection`.
- `moirai-async/net/stream.rs` `TcpStream::drop`: pool entry / `active_connections`
  leak when `peer_addr()` fails at drop (abrupt reset). Store the peer addr at
  construction; key the pool on a unique id, not `SocketAddr`.
- `moirai-sync/resource_pool.rs` `recycle`: shard-wide capacity invariant enforced
  with bin-local locks → concurrent recyclers overshoot the configured cap
  (retention beyond budget). Reserve count atomically before insert, or hold a
  shard-level critical section over decide+insert.
- `moirai-core/ipc/{queue,memory}.rs`: `capacity * size_of::<T>()` overflow,
  `meta_size` vs `align_of::<T>()` assumption, and `munmap(ptr, caller_size)`
  using a possibly-wrong length. Bound `T: bytemuck::Pod`, `checked_mul`, store
  and unmap the exact mapping length. (Cross-process/untrusted-input hardening.)
- `moirai-async/timer/wheel.rs`: `cancel` inserts a tombstone into `cancelled`
  for already-fired ids → unbounded growth. Latent (production timer path is
  `driver.rs`/`registration.rs`; `TimerWheel` is currently test-only).
- `moirai-core/communication/zero_copy/adaptive.rs` `flush_batch`: unbounded spin
  on persistent `Full`/`WouldBlock` ignoring a stalled-but-open receiver (liveness).
- `moirai-core/channel/hybrid.rs` + `communication/ring_buffer.rs`: manual
  `unsafe impl Send` on the hybrid halves suppresses the auto-trait check that a
  `!Sync` SPSC ring relies on; sound today only because neither half is `Clone`.
  Encode the SPSC invariant at the type level (one `#[derive(Clone)]` from UB).

## Round 26 (2026-07-15) — FIXED: blocking-lane starvation isolation

**Root cause.** `BlockingTask` previously changed only the static affinity
offset, so a blocking closure could occupy every compute worker and prevent
sync or async-ready work from making progress.

**Fix.** `ThreadScheduler` now lazily creates a bounded Moirai-owned blocking
lane. Each lane worker has a priority-aware bounded queue and its own producer
lock; no blocking worker stacks or queue storage are allocated for compute-only
executors. Blocking pending/active counters remain separate from compute
counters, while quiescence and public metrics aggregate both lanes. Queue-full
admission returns typed resource exhaustion, and shutdown closes, drains, and
joins the lane before compute-worker handles are joined.

**Evidence.** Value-semantic nextest passes 86/86, including compute progress
with every blocking worker occupied, priority order, queue saturation,
queued-task cancellation, graceful drain, and shutdown rejection. Warning-denied
all-target/all-feature Clippy, rustdoc, and doctests pass. Criterion
The latest local Windows GNU run reports `blocking_lane_schedule_join` at
479.79 ns [469.43, 496.85] and `blocking_lane_concurrent_producers` at
180.90 us [177.66, 184.00]. These rows measure different workloads and are
not compared to each other; no speedup claim is made without a stored
pre-change baseline.
