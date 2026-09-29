# ADR 0019: Help-while-waiting scheduler scope (nested-scope soundness)

Status: Accepted

- Date: 2026-07-03
- Change class: [arch]
- Refs: ISSUE-208, concurrency_audit.md Round 20
- Revision 2026-09-29: pin the scoped-lifetime and scheduler-membership rules
  after PR #513 identified the first invalid access behind the residual crash.

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
- **Use-after-free (identified).** `SchedulerScope<'scope>` was covariant in
  `'scope`, so safe code could shrink the lifetime and enqueue a job borrowing a
  body-local. The body dropped that value before `flush` scheduled the job.
  Concurrent nested scopes exposed the invalid read as
  `STATUS_HEAP_CORRUPTION` (0xC0000374).
- **Foreign-worker unwind (identified).** Worker IDs are process-wide. A worker
  from one scheduler could open a scope on a smaller scheduler, index beyond its
  worker table during `drain_scope`, and unwind while jobs still borrowed the
  opener's frame.

**Decision.** Make the scope waiter *work-conserving*. `scope` calls
`drain_scope(&state)` instead of `state.wait()`:

- `SchedulerScope` is invariant in `'scope`. The public parallel scope carries
  separate `'scope` and `'env` lifetimes, matching `std::thread::Scope`, so a
  spawned job can borrow the environment and cannot borrow a body-local.
- If the caller is one of **this scheduler's workers**, confirmed by matching
  its registered thread, it runs jobs from that scheduler until the scope is
  empty. It spins briefly, then timed-parks only when peers are executing the
  remaining jobs. The worker never parks while holding runnable pending work.
- A non-worker or a worker owned by another scheduler parks while this
  scheduler drains its jobs. It never indexes this scheduler with a foreign ID.
- An unwind cannot escape `drain_scope` while borrowed jobs remain live. A
  double panic while dropping a job's panic payload also aborts inside job
  execution, matching scoped-thread safety requirements.

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

**Evidence.** `compile_fail,E0597` doctests reject body-local borrows at both
scope surfaces. `scope_opened_from_another_schedulers_worker_completes` pins the
foreign-worker arm. `scheduler_scope_nested_saturation_completes` preserves the
deadlock correction, and `scheduler_scope_recursive_fork_join_is_sound` checks
the drive-shaped arithmetic-series oracle at `W ∈ {1,2,4}`. At revision
`c889d2d8`, 481 executor, iterator, and parallel tests passed, including the
unchanged nested value oracle; the standalone 300-pass nested workload also
passed. Evidence tier: type-system rejection plus value-semantic regressions.

**Follow-up.** With `scope` sound, a parallel non-indexed `drive` can be
reintroduced against this primitive with a parallelism-asserting test
(ISSUE-208 (c)); tracked separately so it lands as its own verified slice.
