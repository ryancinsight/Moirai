# ADR 0017: moirai-iter Disposition (prune vs continue)

Status: Accepted

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

## Revision 2026-07-03 - resolved: continue-and-make-real

This revision supersedes the record's original
`Proposed - BLOCKED ON OWNER ADJUDICATION` status. The owner adjudicated,
so the decision is `Accepted`; the adjudication and its executed
consequences are recorded below verbatim from the monolith.


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
