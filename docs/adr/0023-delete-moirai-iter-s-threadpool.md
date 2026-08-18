# ADR-023: Delete `moirai-iter`'s ThreadPool

Status: Accepted

- Date: 2026-07-26
- Change class: [major]
- Refs: ISSUE-222, ADR-022

**Intent.** Finish what ADR-022 deferred: remove `moirai_iter::base::ThreadPool`
rather than keep it as a narrowed fallback, and define what replaces it.

**Context.** ADR-022 moved the one nesting caller off the pool and narrowed the
pool's contract to "flat, non-nesting work only", leaving it as the
`ShuttingDown` fallback for three indexed fan-outs plus the pool
`ParallelContext` owned. That left the crate with two runtimes for one role —
the condition the audit that produced PRs #97, #98 and #99 kept finding defects
in, and a standing invitation for a fourth caller to reintroduce the nesting
class the ADR-022 work removed.

Two facts make the fallback unnecessary rather than merely narrow. The retry
only exists for `ShuttingDown`, which is precisely when standing up worker
threads is least defensible, and the work it retries is a flat index domain —
"run it somewhere" is satisfied by the caller's own thread. Since ISSUE-221,
admission backpressure never reaches the fallback at all: the scheduler runs a
rejected chunk on the submitting lane.

**Decision.** Delete `ThreadPool`, `get_shared_thread_pool`, `PoolJoinGuard`
and the job erasure behind them. The three fan-outs fall back to running the
index domain on the caller; `pool_fallback_permitted` becomes
`sequential_fallback_permitted` with its retry policy unchanged (only a clean
`ShuttingDown` re-runs; a partial run still panics). `ParallelContext` keeps
its chunking but schedules through `global().for_each_indexed`, so several
contexts share one worker set instead of each starting a pool.

Each fallback was a second copy of its executor closure — the same body written
twice, free to drift. Binding the closure once and lending it to
`for_each_indexed` lets the fallback re-run that same body, so the duplication
goes with the pool rather than being ported.

**Consequences.** `moirai_iter::ThreadPool` and `moirai::ThreadPool` leave the
public API: [major]. No consumer in the stack referenced either — every other
`ThreadPool` in the tree is rayon's, in comparison benchmarks. External callers
wanting a fan-out use `moirai_parallel`'s operators or
`moirai_executor::global()`; both run on the one work-stealing scheduler, which
is the point. There is no shim: a compile error naming a deleted type is a
better migration signal than a forwarding wrapper that silently keeps a second
runtime alive.

`ParallelContext::execute_iter` gains a fix in passing. It collected chunk
results from a channel until the senders dropped, so a panicking chunk ended
the collect early and returned a short `Vec` with `Ok` — measured at 32 of 40
items against the parent revision. Results now land in per-index slots and a
missing chunk surfaces as a panic.

**Rejected alternatives.** Keeping the pool for shutdown alone: it preserves a
whole runtime, its worker threads and its defect surface, to serve a path that
a `for` loop covers. Falling back to `std::thread::scope`: fresh threads per
call, and equally pointless during shutdown. Deprecating rather than deleting:
`#[deprecated]` on a type whose entire purpose was to be the second runtime
keeps that runtime compiled, constructible and reachable.

**Verification.** `moirai-iter` and `moirai` suites, warning-denied Clippy,
rustdoc and a workspace `--all-targets` check. `execute_iter` gains order and
completeness tests plus a truncation regression, verified red against the
parent revision. Evidence tier: empirical (value-semantic tests, red→green on
the truncation defect) plus type-level (the deleted API cannot be reached).
