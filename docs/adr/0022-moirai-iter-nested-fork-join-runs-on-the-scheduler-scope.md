# ADR 0022: moirai-iter nested fork-join runs on the scheduler scope

Status: Accepted

- Date: 2026-07-25
- Change class: [arch]
- Refs: ISSUE-219, PRs #97, #98, #99

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
