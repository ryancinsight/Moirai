# ADR 0019: Help-while-waiting scheduler scope (nested-scope soundness)

Status: Accepted

- Date: 2026-07-03
- Change class: [arch]
- Refs: ISSUE-208, concurrency_audit.md Round 20

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
