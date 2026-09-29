# ADR 0019: Help-while-waiting scheduler scope (nested-scope soundness)

Status: Accepted

- Date: 2026-07-03
- Change class: [arch]
- Revision 2026-09-29: [PR #515](https://github.com/ryancinsight/Moirai/pull/515)
  retracts [PR #514](https://github.com/ryancinsight/Moirai/pull/514)'s unsupported
  attribution of the residual iterator crash to the scope defects fixed in
  [PR #513](https://github.com/ryancinsight/Moirai/pull/513).

**Context.** `ThreadScheduler::scope` fans borrowing jobs onto the scheduler
and keeps its stack-owned `SchedulerScopeState` alive until every scoped job
completes. A worker that parks in `SchedulerScopeState::wait` cannot run nested
scoped work. With one worker, the only runner then waits on jobs it must execute;
with more workers, nested waits can saturate the pool. Help while waiting
addresses this scheduling deadlock.

The recorded `STATUS_HEAP_CORRUPTION` (0xC0000374) and libtest access failures
are observations, not a localized invalid access. Neither making the waiter
progress nor fixing a separate lifetime defect establishes their cause.

**Decision.** Keep the scope waiter work-conserving through
`drain_scope(&state)`, with the lifetime and ownership constraints added by
[PR #513](https://github.com/ryancinsight/Moirai/pull/513):

- `SchedulerScope` is invariant in `'scope`. The public parallel scope carries
  separate `'scope` and `'env` lifetimes, so spawned jobs may borrow the
  environment but cannot borrow values local to the scope body.
- Only a worker belonging to this scheduler, established by its registered
  thread identity, helps drain its jobs. It spins briefly and timed-parks when
  other workers are executing the remaining work.
- Other callers park while the owning scheduler drains jobs. A foreign worker
  ID cannot index this scheduler's worker table or select its owner deque.
- A drain unwind cannot escape with borrowed jobs outstanding; this path
  aborts. A panic while dropping a job's panic payload also aborts.

`next_job(worker_id)` accesses the caller's owner deque and steals through the
multi-consumer operations. The membership check preserves the owner-side
restriction; it does not prove every unsafe queue/storage path sound.

Indexed fan-out and indexed map/reduce use `drain_scope` for the same nested
wait shape. Their chunk count follows logical work and worker-plus-caller
lanes. `Adaptive` owns its profitability threshold; explicit `Parallel` is not
silently overridden by an index-count heuristic.

**Alternatives rejected.** Flattening only the iterator terminals leaves
nested scope callers exposed to the scheduling deadlock. A separate blocking
pool adds threads instead of allowing existing workers to run nested jobs.
Closing the residual crash diagnosis from passing scope regressions is rejected
because those regressions do not distinguish its candidate causes.

**Evidence and limits.** The saturation and recursive fork-join tests exercise
nested progress and their value oracles. The `compile_fail,E0597` doctests
exercise rejection of a body-local borrow. The
`scope_opened_from_another_schedulers_worker_completes` regression exercises
one scheduler's worker opening a scope on another scheduler.

Those two added cases do not reproduce the original iterator trigger:
`moirai-iter/src/parallel/sources.rs::drive_split` declares the branch storage
and result slots before entering `global().scope`, and its nested drives use
the same global executor. At the reviewed revision `c889d2d8c2809bd0caedfc5ff7e112831d016dc3`,
neither a body-local borrow nor a foreign scheduler is demonstrated on that
path. Their relevance to the residual crash remains an unverified hypothesis.

The historical diagnosis at `7ba6d0ad2750e484cebaba667b6c2bf1fc458faa`
(`docs/backlog.md`, `MOI-EXECUTOR-SIZING-2026-09-10`) records three candidate
arms with no faults across 20,000 standalone passes per arm and 1,200
single-pass process launches per arm. The rare libtest failure remained
unlocalized and the arms did not separate. These are recorded negative
exposures, not proof of absence or a fresh reproduction. The later 300-pass
standalone success reported by PR #514 cannot strengthen them into a diagnosis.

**Remaining work and overturning evidence.**
[MOI-EXECUTOR-SIZING-2026-09-10](../../backlog.md#MOI-EXECUTOR-SIZING-2026-09-10)
remains open. Closure requires a crash dump, sanitizer, or equivalent checker
localizing the first invalid access, or a controlled reproducer separating the
candidate scheduler arms. Preserve `nested_iteration_produces_correct_values`
and its workload while gathering that evidence. Caller-help, default sizing,
and the dependent fork-join latency work remain gated on that diagnosis.
