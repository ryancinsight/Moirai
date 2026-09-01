# Moirai Development Checklist

**Target**: Unreleased

## MOI-ITER-CONTEXT-PARALLELISM-PROBE-2026-09-01 — Remove async-context control-plane allocations [patch] [perf] — in progress

- **Outcome:** remove attributed full-topology discovery and adjacent closure /
  completion-container allocations from parallel-context async terminals.
- **Acceptance:** an unchanged warmed public ready-future map allocation census
  checks every ordered value and separates context/concurrency/output costs; a
  correction reuses the process-wide count, borrows the terminal operation,
  drains for-each completion without a throwaway vector, and leaves explicit
  async/hybrid concurrency limits unchanged. Paired Criterion evidence,
  debug/release Nextest, warning-denied
  Clippy/Rustdoc, cross-target check, doctests, benchmark smoke, SemVer, and
  independent review pass.
- **Scope / non-goals:** `execution/base.rs`, the existing private
  process-parallelism capability, one focused public allocation/value test, the
  retained execution-context Criterion binary and contract coverage, CHANGELOG,
  and PM state. No scheduler topology, executor construction, public API,
  future-buffer implementation, workload, or timeout change.
- **Risk / change:** internal `[patch]`; stop after measurement if topology is
  not a repeat allocation source or if sharing the count changes configured
  async/hybrid limits or public values.
- **Integrator / lease:** Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d` on
  `perf/iter-context-parallelism-probe`; source lease discharged at `7fee7dd`;
  independent review green; PR #225 delivery remains; last update 2026-09-01.
- **Evidence:** the warmed public 1,024-item ready-future map moves from 1,118
  allocations / 152,616 gross bytes to 1,035 / 114,816, with all ordered values
  exact. The paired Criterion median moves from 81.025 us (95% CI
  79.601-82.117 us) to 63.307 us (95% CI 63.041-63.545 us), 21.9% lower with
  disjoint intervals. The residual 1,024-scale allocation count is
  dependency-owned buffered-future node storage and remains outside this
  control-plane increment. Exact-source gates pass: focused allocation and
  contract tests; debug and release all-feature suites 233/233 with two
  configured skips each; warning-denied all-target/all-feature Clippy, Rustdoc,
  and AArch64 Windows all-target check; 4/4 doctests; benchmark smoke;
  formatting/diff checks; and cargo-semver-checks 223/223 under patch.
  Independent committed-object review of `7fee7dd` found no blocking issue; it
  did not rerun local tests or inspect revision-attested raw Criterion samples.

## MOI-ITER-ZERO-COPY-TOPOLOGY-PROBE-2026-09-01 — Remove count-only topology discovery [patch] [perf] — complete

- **Outcome:** measure and, only if attributed, remove full NUMA/cache topology
  discovery from zero-copy iterator construction when the operation needs only
  the process-available lane count.
- **Acceptance:** an unchanged allocation census separates iterator
  construction from a 1,024-item sequential map and checks every value; a
  correction must preserve empty/small/parallel chunk planning and public
  values, eliminate the attributed constructor allocations, and retain a
  non-regressing paired Criterion row. Debug/release Nextest, warning-denied
  Clippy/Rustdoc, doctests, benchmark smoke, SemVer, and independent review pass.
- **Scope / non-goals:** one private process-parallelism helper and its two
  count-only iterator callers, focused allocation/value tests, the existing
  cache-iterator Criterion binary, CHANGELOG, and PM state. No execution-context
  or scheduler topology, executor construction, dispatch threshold, public API,
  workload, or timeout change.
- **Risk / change:** internal `[patch]`; stop with the instrument if topology
  discovery is not the measured allocation source or if process-available
  parallelism changes the existing chunk formula.
- **Integrator / lease:** Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d` on
  `perf/iter-zero-copy-topology-probe`; lease: none; source `66bc88b`; PM
  `06e9878`; PR #224 merged with history as `f689900`; last update 2026-09-01.
- **Evidence:** at exact entry source `8960e3b`, warmed construction made 82
  allocations totalling 13,184 gross bytes before a sequential map made its sole
  8,192-byte output allocation. The candidate caches the process parallelism
  count once and removes full Themis topology construction; warmed construction
  now makes zero allocations and the map retains exactly one 8,192-byte output
  allocation, with all 1,024 ordered values checked. The same unchanged
  Criterion row moves from a 7.0749 us median (95% CI 6.8996-7.1410 us) to
  368.13 ns (95% CI 362.14-368.34 ns), 94.8% lower with disjoint intervals.
  Local gates pass: focused allocation coverage 7/7; debug and release
  all-feature suites 230/230 with two configured skips each; warning-denied
  all-target/all-feature Clippy, Rustdoc, and AArch64 Windows all-target check;
  4/4 doctests; unchanged six-row benchmark smoke; formatting/diff checks; and
  cargo-semver-checks 223/223 under patch. Independent review of `871a1c9`
  found an oversized-element
  modulo-by-zero prefetch boundary; the fix-forward clamps prefetch spacing to
  one element and adds an exact-once parallel regression. Exact corrected source
  `66bc88b` passed independent static re-review. Every repository check on PR
  #224 passed; the external non-required `recurseml/analysis` service errored.

## MOI-ITER-MAP-DIRECT-OUTPUT-2026-09-01 — Remove shard-local map outputs [patch] [perf] — post-merge fix-forward

- **Outcome:** write chunked `ParallelIter::map` results directly into one
  ordered full-output allocation, removing shard-local output vectors and the
  second full-result flatten allocation.
- **Acceptance:** record an unchanged 131,072-item public allocation baseline
  that reaches fan-out; empty, one-chunk, full-chunk, and ragged values remain
  exact; non-`Clone` outputs move once; success and mapper-panic drop counts are
  exact; warm output storage uses one full allocation with no per-shard output
  vectors; a retained paired Criterion row does not materially regress; Miri,
  debug/release tests, warning-denied Clippy/Rustdoc, doctests, benchmark smoke,
  SemVer, and independent review pass.
- **Scope / non-goals:** `moirai-iter` chunked `ParallelIter::map`, private
  initialization cleanup, value/drop/allocation tests, one existing Criterion
  binary, CHANGELOG, and PM state. No `reduce`, scheduler, chunk threshold,
  public API, timeout, or workload reduction.
- **Risk / change:** internal `[patch]`; unsafe direct initialization must remain
  panic-safe for initialized prefixes, zero-sized outputs, and ragged tails.
  Stop with the instrument if the baseline misses fan-out, lacks the duplicate
  full output, or paired intervals establish a regression.
- **Integrator / lease:** Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d` on
  `fix/iter-map-topology-probe`; PR #222 merged as `2a782b9`; PR #223 merged as
  `5f2882e`. The post-merge Workspace gate exposed a Linux-only repeat-probe
  allocation and closure continues in the zero-copy topology increment; source
  lease discharged. Last update 2026-09-01.
- **Entry baseline:** the unchanged warmed public map at 131,072 `u64` values
  makes 114 allocation calls totalling 3,815,568 gross allocated bytes, 3.64×
  the 1,048,576-byte final output. The retained x86-64 Windows Criterion row
  measures a 1.3115 ms median (95% CI 1.3023–1.3420 ms). Input construction is
  outside both measured regions; source values and every ordered result are
  checked before accepting the instrument.
- **Candidate evidence:** PR #222's initial Windows result was 85 calls and
  1,062,720 gross bytes, but hosted Linux run `33493866766` falsified its
  platform-independent attribution with 417 calls and 1,361,702 gross bytes.
  `ParallelIter::new` still materialized a NUMA/cache topology snapshot solely
  to derive a logical-processor count. The fix-forward uses process-available
  parallelism directly; the unchanged warmed ledger now structurally requires
  exactly three allocations and bounds gross bytes to the 1,048,576-byte final
  output plus 6.25%. The retained median is 0.29612 ms (95% CI
  0.28512–0.30076 ms), 77.4% below entry with disjoint intervals. Original
  direct-output debug/release coverage was 226/226, focused Miri coverage was
  3/3, exact-baseline SemVer passed 223 checks, and independent review of
  `7f7b279...924f5d9` was GREEN. Fix-forward debug and release Nextest pass
  227/227 with two configured skips in each profile; warning-denied
  all-target/all-feature Clippy and Rustdoc pass; 4/4 doctests, formatting, and
  the focused Criterion target pass. Independent static review of
  `e009262...c98d979` is GREEN. Post-merge Workspace run `33496098579`, job
  `99818541140`, observed seven calls rather than three because Rust 1.97's Linux
  `available_parallelism` path allocates cgroup path/read storage on every call.
  The successor caches that process count once; the focused ledger passes 7/7
  locally, with hosted Linux closure pending on the successor revision.

## MOI-ASYNC-WAKE-BATCH-ALLOCATION-2026-09-01 — Remove redundant batch-wake allocation [patch] [perf] — done 2026-09-01

- **Outcome:** batch grants drain pending waiter ids directly into one result
  reserved from the pending-count upper bound, removing the redundant id-buffer
  allocation and geometric result growth while retaining wake-after-unlock.
- **Acceptance:** record the unchanged exact allocation baseline first; FIFO,
  cancellation, grant payload, and all `Notify`/`Condvar`/`RwLock` value
  semantics remain exact; batch grant uses one result allocation rather than a
  second transient id allocation; focused/release tests, allocation census,
  warning-denied Clippy, Rustdoc/doctests, benchmark smoke, and independent
  review pass.
- **Scope / non-goals:** private `WaitQueue::grant_all`, its batch-grant tests,
  one retained allocation/performance instrument, CHANGELOG, and PM state. No
  public API, queue representation, lock/wake ordering, scheduler, timeout, or
  zero-allocation broadcast claim.
- **Risk / change:** internal allocation-only `[patch]`; stop without a source
  change if the exact baseline does not attribute a redundant allocation or if
  direct draining changes cancellation/value semantics.
- **Integrator / lease:** Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d`;
  reviewed source head `b690f88`; PR #221 merged with history preserved as
  `07b3958d38c87b1e7b37139551f709b6fa583dd7`; lease: none. Last update
  2026-09-01.
- **Evidence:** exact instrument commit `1cfb9be` measured six allocations and a
  666.99 ns median (664.49–669.19 ns) for 64 public `Notify` waiters. The
  candidate retains one owned-waker allocation and measures 343.12 ns
  (342.52–344.04 ns), a 48.6% median reduction with disjoint intervals. Debug
  and release Nextest each pass 99/99; warning-denied all-target/all-feature
  Clippy, warning-denied Rustdoc, doctests, and the focused Criterion smoke pass.
  The exact directory-baseline SemVer check passes 223/223 applicable checks
  with 31 skips and requires no version change. Independent review corrected
  the original exact-capacity wording for lazy cancellation holes; exact
  cumulative re-review is GREEN.

## MOI-PAR-STANDARD-TERMINAL-STREAM-2026-09-01 — Remove standard-terminal materialization [minor] [perf] — done 2026-09-01

- **Outcome:** preserve one-call `Sum<Item>` / `Product<Item>` semantics while
  letting sources and compatible adapters expose their logical stream directly,
  without first collecting it into a full-size vector.
- **Acceptance:** lawful batch-sensitive/custom accumulators and empty/value
  semantics remain unchanged; a warmed borrowed copied/map/filter standard sum
  drops its full-stream allocation to zero; baseline and candidate Criterion
  medians use one retained instrument; non-streaming adapters keep the default
  path; debug/release Nextest, allocation, Clippy, Rustdoc/doctest, benchmark
  smoke, SemVer, and independent review gates pass.
- **Scope / non-goals:** the `ParallelIterator` sequential-item seam, compatible
  sources/adapters, standard-terminal allocation/value tests, one existing
  Criterion binary, Rustdoc, CHANGELOG, and PM state. No reassociated-terminal,
  scheduler, fan-out, threshold, workload, timeout, or release change.
- **Risk / change:** additive defaulted trait method, `[minor]`; no implementor
  migration. Stop with the instrument only if the exact baseline has no
  full-stream allocation or candidate timing materially regresses.
- **Integrator:** Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d` on
  `perf/iter-standard-terminal-stream`.
- **Lease:** none. Reviewed source head `e7483e2`; PR #220 merged with history
  preserved as `c06feba8d009645228e7e86f101c190dcc5f352c`. Last update
  2026-09-01.
- **Evidence:** the unchanged 65,536-item borrowed copied/map/filter standard
  sum returns the reference value and makes one warmed allocation. With the
  retained `seq_iter` seam it makes zero. Paired Criterion medians
  baseline→candidate are 546.649→262.779 ns (1,024),
  25.7769→8.37063 µs (32,768), and 415.962→33.7676 µs (131,072), reductions of
  51.9%/67.5%/91.9%; all paired confidence intervals are disjoint.
- **Local gates:** debug and release `moirai-iter` Nextest pass 220/220 with
  two configured skips; warning-denied all-target/all-feature Clippy and
  Rustdoc pass; 4/4 doctests and 72/72 benchmark contracts pass; the complete
  retained Criterion binary passes its 28.4-second single-iteration smoke.
  Directory-baseline SemVer analysis passes 196 checks with 58 inapplicable
  skips and reports no required version change. Independent review found one
  stale `seq_try_fold` materialization statement; source `e7483e2` corrects the
  `seq_iter` delegation contract, and exact cumulative re-review is GREEN.
  Post-merge lockfile, workspace, Loom, supply-chain, and fuzz checks are GREEN.

## MOI-CI-DRAFT-GATE-2026-09-01 — Suppress draft pull-request runners [patch] [ci] — complete

- **Outcome:** draft pull requests schedule no repository jobs; opening or
  reopening a non-draft pull request and marking a draft ready preserve every
  existing workflow command and workload.
- **Acceptance:** all eight independent jobs reject draft pull-request events;
  ready/non-draft pull requests retain their path-filtered selections; push,
  schedule, and manual events are unchanged; normalized YAML differs only by
  the explicit pull-request activity lists and draft predicates; hosted draft
  and ready transitions confirm the event semantics.
- **Scope / non-goals:** `.github/workflows/{rust-ci,python-ci,book-pages}.yml`,
  CHANGELOG, and this item's PM regions. No command, feature, matrix, cache,
  permission, workload, assertion, or timeout change.
- **Integrator:** Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d`.
- **Lease:** none. Reviewed head `bc30759`, PR #219, merge `9603564`; last
  update 2026-09-01.
- **Entry evidence:** draft PR #218 launched all five Rust jobs; the final ready
  run passed in 6s–2m12s. The Python and book workflows expose three more
  independent jobs with the same unguarded draft-event behavior.
- **Candidate evidence:** source `8ad506d`, draft PR #219. PyYAML parses all
  three workflows and removing only the activity lists and draft predicates
  yields the exact prior definitions. Draft runs `33485286665`, `33485286034`,
  and `33485286617` complete skipped: all eight jobs have zero steps. GitHub's
  current reusable-workflow syntax explicitly permits `jobs.<job_id>.if`.
  The exact ready-head runs then passed every unchanged repository job:
  Rust `33485592298` (11s–2m22s), Python `33485591545` (55s–2m9s), and book
  `33485592200` (45s build; pull-request deploy correctly skipped).

## MOI-PAR-SUM-CONTRACT-2026-09-01 — Preserve standard terminal semantics [minor] [perf] — complete

- **Outcome:** restore `ParallelIterator::sum` and `product` to the standard
  `Sum<Item>` / `Product<Item>` contracts and retain the measured parallel
  fold as explicit reassociated terminals.
- **Acceptance:** lawful custom accumulators that do not implement
  `Sum<Self>` / `Product<Self>` compile and preserve whole-stream values;
  batch-sensitive accumulators prove standard and reassociated semantics are
  distinct; primitive, empty, floating-point, allocation, benchmark-contract,
  Clippy, Nextest, Rustdoc, doctest, and SemVer gates pass.
- **Integrator:** Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d`.
- **Lease:** none. Reviewed head `7c7dfc1`, PR #218, merge `bd435b8`;
  last update 2026-09-01.
- **Local evidence:** debug and release `moirai-iter` Nextest pass 219/219 with
  two configured skips; warning-denied all-target/all-feature Clippy, Rustdoc,
  3/3 doctests, both retained Criterion smoke binaries, and 72/72 benchmark
  contract tests pass. Directory-baseline `cargo-semver-checks` passes 196
  checks with 58 inapplicable skips and reports no required version change;
  the packed-object baseline mode could not complete because its local clone
  exceeded the tool's in-memory entry limit. Independent cumulative review
  found and closed two evidence defects: source `ee0d08a` corrects the stale
  materialization claim, and source `3f8d729` derives the floating bound from
  the actual two 4,096-item leaf folds plus one merge. Exact correction
  re-review is GREEN. Hosted lockfile, fuzz, Loom, supply-chain, and workspace
  gates pass; the workspace gate completes in 2m12s.

## MOI-ATLAS-CONFORMANCE-RATCHET-2026-08-31 — Restore the stack hygiene ratchet [patch] — complete

- Source `e13ae98` merged through PR #203 as `a98154b`; the exact scan is
  oversized 37, existence-only 33, commented-code 6, sequentially-consistent
  sites 76, and target forks 0, with no baseline or timeout change.
- Independent exact-Git review and hosted run `33445215503` are green. Local
  evidence is affected Nextest 650/650, Loom 5/5, warning-denied all-target
  Clippy, complete SIMD smoke, Rustdoc, doctests, formatting, and diff checks.

## MOI-ADAPTIVE-THRESHOLD-PREMISE-2026-08-31 — `Adaptive` needs body-cost evidence [patch] [perf] — documented 2026-08-31, decision open

- **Corrected finding.** `ADAPTIVE_PARALLEL_THRESHOLD = 1024` is an element
  count, while the crossover depends on `n * per_element_cost`. The original
  uncommitted best-of-block probe reported an approximately 11.9 us floor and
  was not reproducible. The retained Criterion instrument at `0a267f9`
  measures the entry one-multiply crossover between 8,192 and 16,384 elements.
- **Candidate crossovers** (four workers, identical inputs and arithmetic,
  Criterion 10 samples / 1 second measurement):

  | body | crossover | evidence |
  |---|---|---|
  | one multiply | 4,096-8,192 | serial/parallel medians 1.532/1.954 us at 4,096 and 3.028/2.004 us at 8,192 |
  | `sqrt` + `ln_1p` | 512-1,024 | serial/parallel medians 1.536/1.951 us at 512 and 3.047/2.096 us at 1,024 |
  | 24 chained FMAs | below 512 | serial/parallel medians 14.287/4.924 us at the smallest retained size, 512 |

- **Implication.** The count-only threshold still cannot fit all bodies. At
  1,024 a multiply-only reduction remains slower in parallel, while the
  compute-heavy bodies already benefit. `Sequential` and `Parallel` remain the
  explicit choices for callers with known body cost; changing the count without
  a consumer-weight model remains rejected.
- **Deliberately not raised.** Raising it to fit a cheap body would serialize
  heavy consumers over exactly the range where they win: apollo's
  spherical-harmonic mode loops fold an expensive body through `Adaptive` at
  these sizes, and the retained compute-heavy rows already win at 1,024. One
  count cannot serve both, and changing it without consumer evidence trades
  one measured regime for an unmeasured consumer regression.
- **Re-open trigger:** a consumer supplies a representative body-cost
  distribution or a body-cost-aware policy is specified without adding
  per-element dispatch.

## MOI-DISPATCH-FLOOR-2026-08-31 — Keep indexed CPU work on compute workers [patch] [perf] — done

- **Integrator:** Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d`.
- **Lease:** none. Instrument `0a267f9`, source/docs candidate `d2fc4d4`, and
  raw-sample evidence correction `1ad24e3` are published in PR #205; base
  `eed1c54`; last update 2026-08-31.
- **Finding.** The public `Moirai::{for_each_indexed,map_reduce_indexed}` facade
  classified CPU-bound data-parallel work as `BlockingTask`. After ADR-021,
  that marker uses the dedicated blocking lane, so indexed-only runtimes lazily
  construct blocking workers and pay the wrong scheduling path. The generic
  executor seams and `Moirai::scope` retain their existing work-class contracts.
- **Candidate.** Route only the two CPU-bound facade methods through `SyncTask`.
  A thread-provenance regression asserts exact reduction values, compute-worker
  execution, and no blocking-lane execution. No persistent barrier, scheduler
  topology, task-count, arithmetic, or benchmark change is required.
- **Measured evidence.** Raw `map_reduce_indexed` sample medians fall from
  2.190 to 1.765 us at 1,024 (-19.4%), 2.022 to 1.653 us at 4,096 (-18.2%),
  3.039 to 2.173 us at 16,384 (-28.5%), and 8.113 to 4.015 us at 65,536
  (-50.5%). The one-multiply parallel median falls 43.4% at 4,096 and 32.9%
  at 8,192; the serial controls move +0.5% and -0.7% respectively.
- **Acceptance.** The floor measured by the same probe drops materially, and
  the threshold item is re-derived against it. Exact indexed coverage,
  allocation and panic contracts, nested use, warning-denied Clippy, package
  Nextest, Rustdoc/doctests, release tests, and same-instrument Criterion
  baseline evidence pass on the exact candidate. Reject a candidate that
  changes values or moves cost into an unmeasured lifecycle boundary.
- **Verification.** Runtime all-feature Nextest passes 28/28, the release
  routing regression and warmed allocation census pass 1/1 each, benchmark
  contracts pass 1/1, and warning-denied all-target/all-feature Clippy,
  Rustdoc, doctests 7/7, formatting, and diff checks are green. Independent
  exact-Git review of `eed1c54..d2fc4d4` is GREEN. PR #205 merged with history
  preserved as `84793eb`.

## MOI-CHUNK-ARRAYS-2026-08-29 — Homogeneous multi-buffer chunks [minor] — done

- **Delivered:** provider PR #200 / head `620398c` / merge `2b9c806`; all checks passed, including zero warmed six-buffer allocations and non-idempotent exactly-once values.
- **Consumer:** Kwavers SWE source `0b2d7baf` landed through fully green PR #670 / merge `fb24b26d`; lease: none.

## MOI-CONTRACT-AUDIT-STALENESS-2026-08-29 — Source-pinning contracts can audit stale or foreign sources [patch] — review

- **Integrator:** claude-fable session 5050c72a. **Lease:** none; provider
  source and focused verification are complete. **Last update:** 2026-08-31.
- **Delivered:** `support.rs` now embeds every audited file at compile time
  through an `EMBEDDED_SOURCES` table of `include_str!` values (generated once
  and kept as source). `read_benchmark` resolves a path against
  that table instead of `fs::read_to_string`, so Cargo tracks each audited file
  as a build dependency of the test binary and the paths resolve relative to
  `support.rs` rather than to `CARGO_MANIFEST_DIR`. Assertions are unchanged —
  only the byte source moved.
- **Guard liveness, before (defect reproduced, not argued):** with the test
  binary fresh, editing an audited source left it byte-identical
  (`cargo build --test benchmark_contracts` → `Finished ... 0.22s`, md5
  unchanged), so the verdict was never a function of the binary. Two worktrees
  sharing `CARGO_TARGET_DIR` then collide on one output name: with the main
  tree's `moirai-gpu/src/task.rs` violating
  `gpu_task_adapter_uses_moirai_block_on_not_pollster` and a lane clean, a lane
  build followed by a main-tree run reported `1 passed` — a **PASS against
  violating source**, exactly the filed observation. Corroboration in the
  shared cache: a `benchmark_contracts-*.exe` dated 2026-08-29 still carries
  `D:\atlas\worktrees\moirai-registry-growth` as its baked audit root.
- **Guard liveness, after:** the same edit now recompiles the test binary
  (`Compiling moirai-benchmarks`) and the contract **fails**; a repeat of the
  cross-worktree sequence also rebuilds, because the embedded bytes differ, so
  the main tree audits the main tree. A renamed or deleted audited file is now
  a **compile error** instead of the previous silently-empty alias read.
- **Meaning-preserving cleanups forced by the change:** three alias leaves named
  files that no longer exist (`moirai-pal/src/reactor/{future,task}.rs`,
  `moirai-iter/src/async_iter.rs`) and were contributing empty strings; they are
  removed. `moirai-pal/src/reactor/{registration,kqueue_transition}.rs` are
  audited directly elsewhere, so no reactor coverage is lost.
- **Residual, stated rather than worked around:** one audit asserts a file's
  *absence* (`simd_iter_backup.rs`). `include_str!` cannot express that — naming
  a missing file is a compile error, not a readable fact — so it keeps
  `benchmark_path` and with it the `CARGO_MANIFEST_DIR` root. It is the only
  remaining run-time path resolution in the suite and is documented as such at
  the function.
- **Gates:** `cargo fmt --all -- --check` clean; `cargo clippy --workspace
  --all-targets --features moirai-benchmarks/registry-diagnostics -D warnings`
  clean; `benchmark_contracts` 70/70.

- **Original filing.** The `benchmark_contracts` suite is this repo's guard against
  regressing accepted designs: it asserts that named shapes are present in, or
  absent from, specific source files. Two properties of *how* it reads them let
  it report a pass it did not verify.
- **1. The audited files are not compile inputs.** `support.rs` reads them with
  `fs::read_to_string` at run time, so changing the code under test does not
  make the test binary stale. Cargo reuses the old binary and the assertions
  re-run against whatever the binary happens to read — or, when nothing forces
  a rebuild, do not re-run meaningfully at all. **Observed 2026-08-29:**
  `executor_registry_registration_rejects_regressed_lock_free_allocator`
  reported PASS against source that violated three of its required markers;
  `touch`ing the test file and re-running turned it red immediately. A guard
  that passes on violating source is worse than no guard, because it is
  believed.
- **2. The root path is baked at compile time.** `benchmark_path` is
  `Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)`. `env!` resolves when
  the binary is *compiled*, so under the stack-mandated shared
  `CARGO_TARGET_DIR` a binary compiled from one worktree audits **that
  worktree's** sources even when run from another. **Observed the same day:**
  two contract tests failed in a lane against an unrelated uncommitted edit
  sitting in the repo's main tree; rebuilding from the lane gave 70/70. Results
  therefore depend on which worktree last built — and this fleet routinely runs
  several worktrees against one target directory.
- **Fix direction, one change for both:** make the audited files compile-time
  inputs via `include_str!`. Its paths resolve relative to the containing
  source file (no `CARGO_MANIFEST_DIR`, so no foreign-tree read), and Cargo
  tracks included files as dependencies, so editing audited source rebuilds the
  test. The `SPLIT_MODULE_ALIASES` table stays expressible as a const table of
  `include_str!` values. Where a path must stay dynamic, resolve the repo root
  from the test's own file rather than the manifest env var.
- **Acceptance:** editing an audited file without touching the test makes the
  relevant contract fail on the next `cargo nextest run`; and a contract run in
  a worktree reports on that worktree's sources with another worktree dirty.
  Both are falsifiable — verify each against the current behaviour first.

## MOI-BENCH-REQUIRED-FEATURES-2026-08-31 — Diagnostic bench breaks the all-targets gate [patch] — done 2026-08-31

- **Integrator:** Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d`.
  **Lease:** none. Source commit `8b8110b`; last update 2026-08-31.
- **Outcome:** declare `registry-diagnostics` as a required feature of
  `result_handle_diagnostics`; default all-target verification now omits the
  target whose diagnostic API is disabled, while feature-enabled execution is
  unchanged. An embedded-manifest regression pins the target-feature relation.
- **Evidence:** default `cargo check --offline -p moirai-benchmarks
  --all-targets`; focused Nextest 1/1; default and feature-enabled all-target
  warning-denied Clippy; feature-enabled Criterion `--test` smoke of every row
  completed in about 45 seconds; fmt, diff, and standalone lock hash
  `dab0b1f06ba224ac29d750921e033d4135f9765c` are clean. Independent exact-object
  review is GREEN; PR #206 merged with history preserved as `c13fbf5`.

## MOI-FLAKY-JOIN-PRECONDITION-2026-08-28 — Make the join test's precondition deterministic [patch] — review

- **Outcome:** `scheduler_join_waits_for_queued_and_active_work` asserted
  `has_work()` after scheduling eight trivial jobs on a two-worker pool, with
  nothing ordering the assertion before the workers drained the queue. The
  jobs now park on a gate the test holds until it has observed the outstanding
  work, so the state the test name claims — two workers active inside a job,
  six queued behind them — is pinned rather than raced for. The gate is
  released before `join()`, which blocks the test thread.
- **Why it matters:** this is the failure that turned moirai `main` red at
  `fc2dce94` (PR #191's merge run, `tests.rs:994`), and the same SHA
  `ff41a098` appears in the run list as both a success and a failure. An
  intermittently red gate stops being read.
- **Evidence (the race is structural, not machine-specific):** the original
  passes 60/60 locally, so local repetition proves nothing — this host wins
  the race consistently where a contended 2-core runner does not. Injecting a
  50 ms delay before the assertion makes the **original fail** (the workers
  drain and `has_work()` reports quiescence) and the **gated version pass**
  under the identical delay. That isolates the dependence on timing and shows
  the gate removes it. Fixed version also 60/60 under repetition.
- **Integrator:** claude-fable session 03d80d33. Gates: `fmt --check` clean,
  Clippy `-D warnings` clean, `moirai-executor` nextest 126/126.
- **Last update:** 2026-08-28.

## MOI-PAR-TERMINALS-2026-08-28 — Parallel terminals and index-range splits [patch] — review

- **Outcome:** compatible `ParallelIterator` terminals stop routing through
  `seq_items()`. `FoldConsumer` and `ShortCircuitConsumer` fold shards through
  the existing `Consumer` protocol, `ParallelIterator::seq_try_fold` is the
  streaming base under them, borrowed drive shards are zero-copy `split_at`
  subslices, and owned shards stop splitting at the dispatch threshold.
  Standard `sum` and `product` remain whole-stream trait invocations because
  `Sum` and `Product` expose no lawful partial-output combine operation; the
  measured shard fold is available through explicit reassociated terminals.
  Closes `MOI-PAR-ITER-SEQUENTIAL-TERMINALS-2026-08-27` and
  `MOI-PAR-ITER-SPLIT-COPY-2026-08-27`.
- **Integrator:** claude-fable session 03d80d33 subagent. Lease: none.
- **Evidence:** pinned criterion medians at 131072 elements —
  `par_iter().map(f).sum_reassociated()` 391.25us to 11.18us (35x, and 6.7x
  under Rayon),
  `count`/`min`/`max` 1.156ms to 36.93us (31x), `find_any` with an early match
  412.92us to 5.13us (80x, and 1.85x under Rayon), `find_any` with no match
  431.25us to 20.58us (21x). Same-code run-to-run spread on this hybrid-core
  host reaches 32%, so movements under about 1.6x are not resolved; every
  claim above is far outside that band. Allocation probes pin sub-linear
  allocation independently of the host. Workspace Nextest 899/899 with 7
  configured skips, warning-denied all-target Clippy, workspace doctests, and
  `cargo fmt --check` all pass.
- **Measured and rejected:** folding `partition`/`unzip` across shards was
  2.4x slower at 1024 elements and 2.3x at 32768 — slower below the dispatch
  threshold too, so the cost is the pair-of-vectors accumulator moved through
  the fold closure, not the parallelism. They stay sequential; parallelising
  them needs a size-hinted output collection.
- **Residual:** `fold`, `try_reduce`, `for_each_with`/`_init` and
  `try_for_each_with`/`_init` stay sequential by contract (one threaded
  accumulator or state value), as do standard `sum` and `product` (one complete
  logical stream). `position_first`/`position_last` stay sequential because the
  non-indexed protocol cannot hand a shard a correct logical offset through a
  length-changing adapter. Compatible shard-folding terminals stream instead of
  collecting; standard `sum` and `product` materialize the logical stream once
  to preserve the trait contract. Owned splits still copy down to the threshold.
- **Last update:** 2026-08-28.

## MOI-PAR-ADAPTER-DRIVE-COLLECT-2026-08-28 — Adapters collect in drive [patch] — review

- **Integrator:** claude-fable session 03d80d33 subagent. Lease: released.
- **Last update:** 2026-08-29.
- **Outcome:** twenty-six adapter and source `drive` implementations collected
  the whole logical stream into one vector before any split, discarding the
  source's shards for every chain containing them. Six are converted to push
  their transform into a consumer and drive the base, as `map`, `filter`,
  `inspect`, `copied`, and `cloned` already did: `filter_map` and `flat_map`
  through new `FilterMapConsumer`/`FlatMapConsumer`, `flatten` through the
  latter with the identity expansion, `update` through `MapConsumer`, and the
  two block adapters directly, their item stream being identical to the base's.
- **Correction to the item as written:** it listed `Filter` among the
  unconverted. `Filter` and `Inspect` already drove through consumers on
  entry; only `FilterMap` and `WhileSome` in `filter.rs` collected.
- **Not converted, twenty, each recording its reason at the `drive` that
  stays.** Absent logical offset — `Consumer::split_at` carries the *source's*
  split point, which a length-changing adapter invalidates: `enumerate`,
  `positions`, `Map::positions`, the borrowed position stream, `take`, `skip`,
  `step_by`. Prefix dependency, where a shard cannot know an earlier shard
  already stopped: `while_some`, `take_any_while`, `skip_any_while`. Chunk
  boundaries are logical positions, so a split not aligned to one emits a short
  chunk per shard: `chunks`. Adjacency across a shard boundary: `intersperse`.
  Fixed `combine(left, right)` order: `rev`. Two sources split in lockstep:
  `zip`, `zip_eq`, `interleave`, `interleave_shortest`. Threaded state:
  `map_with`, `map_init`. `chain` needs its left branch's logical length to
  split, which it cannot know without the materialization the conversion
  removes.
- **Evidence — allocated bytes, machine-independent.** Borrowed chains at
  65536 elements: 524288 bytes (one full copy of the stream) to **512**, for
  `filter_map`, `flat_map`, and the two stacked. `flatten` 1097728 to 598528
  against a 573440-byte source clone. Owned chains gain 524288 to 786944,
  which is the signature of actually splitting: an owned source has no
  zero-copy split, and the collect shape's tell was allocating exactly the
  source and never splitting it. `enumerate` is byte-identical either side,
  confirming the probe discriminates.
- **Evidence — wall clock**, 131072 elements, Ultra 9 285K pinned to CPUs 0-7,
  one chain per process, 200 warm-up and 200 measured iterations, median with
  a 95% percentile interval. `filter_map().sum()` 373.10us to 16.20us
  (**23.0x**); `flat_map().sum()` 842.10 to 58.80 (**14.3x**);
  `flatten().sum()` 1017.80 to 93.00 (**10.9x**);
  `filter_map().flat_map().sum()` 420.50 to 28.70 (**14.6x**). Control:
  `filter().sum()` 16.40 against 16.80, unchanged as its code is.
- **Owned-source chains are unresolved on this host, not improved.** `update`
  measured 1.0x to 1.5x across repeats. The calibration is an internal
  control: owned `map().sum()`, whose code is byte-identical on both branches,
  itself measured 315.60, 422.70, and 424.70us across three runs of the same
  binary and showed an apparent 1.44x branch difference in one pairing. This
  host cannot resolve below roughly 1.5x for this chain class. The mechanism
  that could produce a real owned-source cost is recorded above: splitting an
  owned source copies, and the collect shape never split at all.
- **Method note.** A first measurement pass ran every chain in one process and
  reported `filter().sum()` at 54.30 against 15.80 on identical code; that was
  cross-contamination from the preceding chain and disappeared under
  one-chain-per-process. Numbers from the batched pass are not reported.
- **Contract.** The audit read seven of twelve adapter files; `filter`, `flat`,
  `map`, `ref_ops`, and `slice_ops` are now in the audited set and pass the
  existing prohibitions unchanged. A new contract pins each converted `drive`
  shape, requires every remaining collect site to carry its recorded reason,
  and holds a non-increasing baseline of collect sites at 20, down from 26.
  Both failing arms were verified against `origin/main` sources.
- **Residual risk.** `drive_split` calls `consume`, not `drive`, on each half,
  so a source splits exactly two ways however long it is; the wins above come
  from removing the materialization rather than from shard count. Raising the
  split depth is separate work and is not attempted here.

## MOI-PAR-INHERENT-SUM-BYPASS-2026-08-28 — Inherent sum overrides run single-threaded [patch] — review

- Unowned. Four inherent `sum` implementations shadow the trait terminal and
  run the whole chain on one thread (`adapters/map.rs:151,174`,
  `adapters/filter.rs:28,92`). They were written to avoid the intermediate
  vectors the trait path used to build; the trait path no longer builds them
  and is now parallel, so these are slower than the method they shadow —
  `par_iter().copied().map(..).filter(..).sum()` measures 35.55us at 131072
  where the trait path measures 11.18us on comparable work. Fix direction:
  delete all four and let the chains take the trait terminal, updating the
  audit contract markers that pin their doc lines.
- **Delivered 2026-08-28** by claude-fable session 03d80d33: all four impl
  blocks deleted (`Map<Chunks<I>>`, `Map<Enumerate<Interleave<StepBy<..>>>>`,
  `Filter<Map<Flatten<I>>>`, `Filter<Map<Copied<VecRefParIter>>>`), 191 lines
  removed against 15 added. The chains now resolve to the trait terminal and
  compile unchanged, so no caller is affected.
- **Contract strengthened rather than merely relaxed:** the four doc lines
  moved from the *required* list to the *prohibited* one, so the shadowing
  shape cannot return under its original wording, and the terminal that
  replaced it stays pinned.
- **Evidence:** pinned to four P-cores at high priority, 200 warm-up
  iterations then 400 timed samples of
  `par_iter().copied().map(..).filter(..).sum()` over 131072 `u64`:
  **29.70us → 16.40us median (1.81x)**, p10–p90 `29.30–30.50` against
  `15.70–17.40` — fully disjoint, far outside this host's ~32% same-code
  spread. An earlier 40-sample run showed an overlapping tail that a longer
  warm-up removed, so the tighter run is the reported one. Gates: `fmt
  --check` clean, Clippy `-D warnings` clean, workspace Nextest 899/899 with
  7 configured skips, workspace doctests pass.

## MOI-ASYNC-CANCEL-LANE-GATE-2026-08-28 — Work-class-matched cancellation gate [patch] — done 2026-08-28

- **Delivered:** PR #191 / implementation `3dcc0aa` / merge `fc2dce9`; independent exact-head review is green; lease: none.
- **Evidence:** both queued cancellation cases pass 2/2 in 28 ms with an asserted `Queued` precondition; all-feature warning-denied Clippy and workspace Nextest 890/890 pass locally. Hosted run `33197943762` passes every job; workspace job `98939799878` completes in 2m11s.

## MOI-CI-FUZZ-SCOPE-2026-08-28 — Event-scoped parser fuzzing [patch] — done 2026-08-28

- **Delivered:** PR #189 / implementation `6885c1c` / merge `ff41a09` registers both parser targets and separates deterministic pull-request seed execution from bounded scheduled/manual mutation campaigns; lease: none.
- **Evidence:** hosted PR job `98929756503` executes every committed seed once in 89 seconds versus the 263-second entry run. Manual job `98931077319` runs both 180-second campaigns concurrently in 181 seconds; the exposed scheduler defect is fixed by PR #190 and exact closure run `33197943762` is green.

## MOI-LOCAL-QUEUE-FOOTPRINT-2026-08-28 — Retained local-queue storage [patch] [arch] — done 2026-08-28

- **Delivered:** PR #187 / implementation `cf84fc29` / merge `32f9d08d` selects the 128-slot default after independent exact-head review at `cdb494e2`; lease: none.
- **Evidence:** the exact five-candidate probe records 1,572,864 direct retained bytes at 128 versus 3,145,728 at 256 across 24 workers, overlapping controlled warm confidence intervals, zero warm fan-out allocations, and zero retained bytes after teardown. Provider Nextest passes 328/328; host, AArch64, documentation, SemVer, and all hosted gates pass.
- **Consumer:** Apollo observes the same retained pool footprint and zero global/direct warm-transform allocations from 1,024 through 262,144 elements.

## MOI-WAKE-CORRECTNESS-2026-08-27 — Lost-wake, channel fence, ZST deque alloc [patch] — done 2026-08-27

- **Delivered:** PR #171 merged as `6641293` (integrator review: exactly-once
  poll CAS makes the inline-wake rung race-free; ZST dangling base coherent
  with the states-array protocol; fmt cure `0584fb0`).
- **Integrator:** claude-fable session 03d80d33.
- **Outcome:** three defects closed with regression coverage: (1) `Waker::wake`
  and the repoll-reschedule path discard injector-admission failure, stranding a
  QUEUED task no later wake can rescue; (2) the hybrid channel's Dekker
  park/produce protocol lacks the StoreLoad fence the tree's futex_mutex
  documents, permitting a last-message hang; (3) `chase_lev::storage::Array`
  passes a zero-size layout to the allocator for ZST elements — library UB
  reachable from safe code.
- **Acceptance:** admission-rejection wake regression, park/unpark last-message
  stress regression, and ZST deque push/pop/steal coverage pass under the
  committed nextest budgets; warning-denied Clippy passes.
- **Evidence:** warning-denied workspace all-target Clippy passes; Nextest
  passes 256/256 (1 configured skip) across executor, core, scheduler, and
  utils, including 8 new regression tests; `moirai-core` doctests pass 3/3
  (1 ignored); the loom channel models pass 5/5, whose negative controls
  prove the modeled protocol loses wakeups without the fence and with
  inverted registration. Residual: the hybrid channel itself has no loom
  model (std parking is not loom-swappable), so its fence placement rests on
  the futex_mutex/mpmc-waiter precedent plus stress coverage.
- **Last-update:** 2026-08-27.

## MOI-INLINE-POLL-DEPTH-2026-08-27 — Bound nested inline-wake polls [patch] — done 2026-08-31

- **Delivered:** source `d94ef46`, PM `c9eacef`, PR #207, merge
  `c35205b067e0428904bcebdc40862126106c68dd`; lease: none. Cross-task rejected
  wakes are bounded to one nested inline poll and deeper wakes complete with
  typed `TaskError::ResourceExhausted`.
- **Evidence:** independent exact-Git review GREEN; executor Nextest 129/129,
  focused debug/release 3/3, warning-denied Clippy/Rustdoc, doctests, and all
  repository checks green. The external `recurseml/analysis` service errored
  and is not a required repository check.

## MOI-REGISTRY-UNBOUNDED-2026-08-27 — Bound the task registry [minor] — partly delivered; the bulk needs a contract decision

- **Growth confirmed by measurement (2026-08-29).** A counting global allocator
  over a live executor: live heap rises linearly with completed tasks at
  **~74 bytes each** — +3.71 MB per 50 000, +14.57 MB by 200 000, i.e. about
  **74 MB per million**. The filed ~100 MB/million was the right order.
- **`cleanup_completed` works; nothing calls it.** A standalone registry
  retains 737 728 bytes over 10 000 completed tasks and releases 737 600 of
  them on one `cleanup_completed(ZERO)` call. The function is not the problem.
- **DELIVERED: the one retention that could be fixed without changing any
  contract.** `register_waker` stored unconditionally, with no completion
  check. `wait_for_task` documents the race: it checks completion, registers,
  then re-checks. If the task completes between the check and the store,
  completion has already taken the absent waker and will never take again, so
  the stored waker — and whatever it owns, typically an `Arc` to async task
  state — is held for the life of the slot, which is never reclaimed.
  `register_waker` now re-checks completion after storing and reclaims the
  stranded waker, which is race-free in both directions because only one
  `take` can win and a spurious wake is always permitted. Regression test
  asserts the payload `Arc` strong count returns to 1; it fails on the
  unfixed code.
- **CORRECTION to this item's own text:** "completion wakers are retained" was
  wrong as written. The normal path does take and wake the waker
  (`mark_completed_since`); only the post-completion registration above
  stranded one.
- **NOT delivered, and it is not a wiring job — it is a contract decision.**
  Sweeping completed slots automatically changes documented behaviour:
  `cancel_task` states "Cancelling an already-completed task is a no-op
  `Ok(())`" and "An unknown task ID is an error", so a swept id turns a
  documented `Ok` into an `Err`. `task_status`/`task_stats` likewise go from
  answering forever to answering for a window. Wiring `cleanup_completed` into
  the idle hook as filed would ship that silently. A second obstacle: the
  sweep is O(blocks x 1024) under the directory write lock, so a naive
  periodic call reintroduces exactly the spawn stall that
  `MOI-SPAWN-GLOBAL-MUTEX` removed.
- **Design worth considering instead of a plain sweep:** ids are monotonic, so
  a completed-watermark ("every id at or below N is completed and swept") plus
  a small exception set answers `is_completed` and `cancel_task` in O(1) for
  swept ids while releasing their 80-byte `TaskState`. That preserves both
  documented contracts; only `task_stats` timing detail is genuinely lost, and
  that loss should be a stated policy rather than a side effect. Bounded,
  cursor-based sweeping keeps the lock hold short.
- Evidence: `registry/registry.rs`, `registry/state.rs:59-73`,
  `hybrid/manager.rs:31-41` (the cancel contract), `schedule/runtime/worker.rs:57-95`
  (the hook and its existing 500 ms throttle).

## MOI-SPAWN-GLOBAL-MUTEX-2026-08-27 — Shard spawn registry locking [arch] — ON MAIN, review still owed (revises ADR 0005)

- **Integrator:** claude-fable session 03d80d33. PR #196.
- **It reached `main` by an unintended route, and that is worth knowing.** This
  reverses a decision ADR 0005 records as rejected, so PR #196 was deliberately
  left unmerged for review. It landed anyway: the board-filing branch for
  `MOI-CONTRACT-AUDIT-STALENESS-2026-08-29` was cut with `git switch -c` while
  the main tree still had `perf/moirai-spawn-registry` checked out, so it
  inherited commit `3575f9b`, and merging that filing (PR #198) carried the
  registry change onto `main`. Nobody reviewed the ADR revision.
- **Consequence:** the change is on `main` and green, but the review it was
  held for has not happened. Reverting is a live option and needs no
  justification beyond wanting the review — the change is self-contained in
  `moirai-executor` plus its contract markers and diagnostic rows. Re-read the
  ADR 0005 revision before deciding; the single-producer regression is real.
- **Process lesson:** branch from `origin/<default>` explicitly, never from
  whatever the tree happens to have checked out.
- **Delivered:** the executor holds `Arc<TaskRegistry>`; registration takes
  `&self` through an atomic id counter and an `RwLock` over the block
  directory. The spawn path resolves its block under the shared guard and
  never clones the block `Arc`, because the scheduled token borrows the slot.
- **Measured (8-core pin, 3 runs, medians; disjoint ranges where claimed):**
  executor spawn 1 producer 3.126 → 2.989 M/s (**4% slower**, disjoint),
  4 producers 3.256 → 4.581 (**1.41x**), 8 producers 3.082 → 4.895
  (**1.59x**). The shape is the real result: before, throughput *fell* from
  one to eight producers; after, it rises. Control (raw scheduler, unchanged
  code) held at ~4.7 → ~9.2 M/s in both states.
- **Attribution:** the same probe reaching the scheduler *without* the
  registry scaled while the executor path did not, which is what located the
  cost in registration rather than in scheduling.
- **The prior rejection's failure condition does not recur:**
  `task_scheduling_overhead` — the benchmark that killed the 2026-05 attempt —
  measures 5.5257 µs baseline vs 5.5348 µs, medians 0.16% apart with
  overlapping intervals. That benchmark is noisy at 10 samples, so this is
  "no detectable change", not an improvement.
- **What is preserved:** dense registry-owned blocks at one allocation per
  1024 tasks, no per-task `Arc<TaskState>`, and single-call registration. The
  benchmark contract still bans the rejected design and now also bans
  re-introducing the production `task_registry: Arc<Mutex<TaskRegistry>>`.
  The `registry_mutex_lock_only` attribution row became meaningless and was
  replaced by `registry_shared_acquire_only` rather than deleted, so the
  tradeoff stays measurable.
- **Gates:** `fmt --check` clean, Clippy `-D warnings --all-features` clean,
  workspace Nextest 900/900 (7 configured skips), doctests pass,
  `cargo semver-checks -p moirai-executor` reports no update required.
- **Residual:** single-producer spawn is ~4% slower; the cost split still says
  slot initialization and timestamp publication dominate per-operation cost
  and remain the next targets. `MOI-REGISTRY-UNBOUNDED-2026-08-27` is
  unaffected but now easier — `cleanup_completed` takes `&self`.
- **Last update:** 2026-08-29.

## MOI-PAR-ITER-SEQUENTIAL-TERMINALS-2026-08-27 — Parallel terminals collect sequentially [patch] — done 2026-08-28

- **Delivered:** closed by `MOI-PAR-TERMINALS-2026-08-28`; folding consumers
  replace compatible `seq_items()` terminals. Standard `sum` and `product`
  retain whole-stream trait semantics, with explicit reassociated methods for
  the shard-folding path.

## MOI-PAR-ITER-SPLIT-COPY-2026-08-27 — Index-range splits over Vec::split_off [patch] — done 2026-08-28

- **Delivered:** closed by `MOI-PAR-TERMINALS-2026-08-28`; borrowed shards are
  zero-copy subslices and owned shards stop splitting at the dispatch threshold.

## MOI-IDLE-BIT-REPARK-2026-08-27 — Re-set idle bit before re-park [patch] — done 2026-08-31

- **Delivered:** source `4d5db90`, PR #208 merge `c4c5dbe`, hosted fix
  `f6bc6e2`, PR #209 merge `990a3ae`; independent reviews and all required
  repository checks are GREEN. The non-required external analysis service errored.

## MOI-SCHEDULER-DROP-LEAK-2026-08-27 — Unreachable Drop shutdown guard [patch] — done

- **Delivered:** reviewed head `b8a5325`, PR #210 merge `f82d83d`; all required
  repository checks passed. Final external-handle drop now closes and joins the
  retained worker set without changing successful scheduling semantics.

## MOI-PARTIAL-SPAWN-CLEANUP-2026-08-27 — Drain workers on partial spawn failure [patch] — done

- **Delivered:** refactor `e5abeb5`, source `31361ce`, reviewed head `5662e8a`,
  PR #211 head `400a2bb`, merge `316bf8f`; every repository check passed. A
  failed construction drains all workers it started before returning its error.

## MOI-EXECUTOR-LOOM-CI-2026-08-31 — Execute scheduler Loom models [patch] — done

- **Delivered:** reviewed head `b8a5325`, PR #210 merge `f82d83d`; all required
  repository checks passed. The bounded workflow now executes all 16 committed
  channel and scheduler Loom models.

## MOI-STEAL-BATCH-GATE-HOIST-2026-08-27 — Enter steal resize gate once per batch [patch] — review

- **Integrator:** claude-fable session 5050c72a. **Lease:** none; provider
  source, instrument, and focused verification are complete. **Last update:**
  2026-08-31.
- **Finding (line citations corrected).** `steal_batch`
  (`deque/chase_lev.rs:478-528`) looped `steal`, and each `steal` opened
  `enter_steal_access` (`:288-303`) — a `SeqCst` `fetch_add`/`fetch_sub` pair on
  the counter every thief shares — plus a reclaim guard. At
  `MAX_BATCH_STEAL = 16` that is up to 32 contended RMWs on one line to move at
  most 16 elements. The filed `:203-218` citation had drifted.
- **Delivered.** `steal` keeps the gate and the guard and delegates to a new
  `steal_within_access`; `steal_batch` takes both once and loops that. No
  ordering, no claim-protocol step, and no `Acquire` array load moved.
- **The tradeoff, measured rather than assumed.** `resize` spins until
  `steal_accesses` reaches zero, so one long hold replaces many short ones and a
  racing resize can now wait behind a whole batch. The instrument therefore
  measures both sides:
  `benchmarks/benches/thread_schedule_comparison/steal_batch_gate.rs`
  (criterion, sample_size 20, 2 s measurement, 300 ms warm-up, ~35 s for the
  group, 0.41 s under the single-iteration smoke). `thief_drain` times K thieves
  draining a pre-filled deque, where no resize can run; `owner_growth_under_
  thieves` times only the owner's push loop from the 16-slot minimum, so its
  nine resizes race live thieves. Both rows assert exactly-once transfer and a
  closed-form payload sum, so a row cannot go fast by losing work.
- **Result** (medians with 95% intervals, one bench binary, only
  `chase_lev.rs` swapped between arms):

  | row | per-item | per-batch | change |
  | --- | --- | --- | --- |
  | `thief_drain/2` | 612.14 µs [607.09, 616.14] | 467.05 µs [464.58, 470.11] | **−23.2%** (p=0.00) |
  | `thief_drain/4` | 875.65 µs [872.68, 878.70] | 769.46 µs [764.05, 774.24] | **−12.0%** (p=0.00) |
  | `thief_drain/8` | 1.2067 ms [1.2004, 1.2128] | 1.0760 ms [1.0728, 1.0791] | **−12.0%** (p=0.00) |
  | `owner_growth/2` | 491.42 µs [484.41, 498.62] | 401.42 µs [395.11, 407.90] | **−19.6%** (p=0.00) |
  | `owner_growth/4` | 828.34 µs [821.54, 836.88] | 828.59 µs [823.34, 835.43] | none (p=0.68) |
  | `owner_growth/8` | 1.6478 ms [1.6245, 1.6671] | 1.6552 ms [1.6439, 1.6692] | none (p=0.62) |

- **Instrument noise, stated because it bounds the verdict.** This host is a
  hybrid Core Ultra 9 285K and thread placement is not controlled. Replaying the
  *identical* build against its own baseline moved medians by at most 6.2%
  (`owner_growth/2`), so only changes beyond that carry a verdict; every result
  above is either well outside it or an explicit "no change". An earlier
  four-point ladder that included a single-thief row drifted up to 52% on that
  row and 42% on `owner_growth/4`; the committed ladder starts at two thieves
  because contention is the subject and the one-thief row measured placement.
- **Verdict: the hoist earns its place.** No thief count shows an owner-side
  regression, so the starvation risk the hold creates does not materialize at
  this batch bound — the traffic removed from the shared line outweighs the
  longer hold. A batch also no longer stalls mid-flight on a resize opening
  between two of its items.
- **Loom coverage, stated honestly.** `tests/loom_chase_lev.rs` passes, but it
  models the `top`/`bottom`/fence/CAS ordering protocol over a loom-tracked
  slot store; it models neither `steal_accesses`/`resizing` nor `steal_batch`.
  The change moves no modeled ordering, so the model still applies unchanged —
  and it supplies no evidence about the hoist. That evidence is the resize-gate
  stress coverage (`deque_concurrency::chase_lev_exactly_once_small_capacity_
  forces_resize` and `..._high_thief_contention`, both green) plus the gate's
  own contract: per-batch entry keeps a thief inside the access region strictly
  longer than per-item entry, never shorter, so `resize` never republishes
  earlier than before. See MOI-LOOM-RESIZE-GATE-2026-08-31.
- **Gates:** `cargo fmt --all -- --check` clean; `cargo clippy --workspace
  --all-targets --features moirai-benchmarks/registry-diagnostics -D warnings`
  clean; `moirai-scheduler` + `moirai-executor` nextest 162/162;
  `benchmark_contracts` 70/70; `moirai-scheduler` doctests 2/2;
  `RUSTFLAGS="--cfg loom"` `loom_chase_lev` 1/1.

## MOI-LOOM-RESIZE-GATE-2026-08-31 — Model the resize gate under loom [patch] — merge

- **Integrator:** Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d` on
  `fix/scheduler-resize-gate-loom`; source candidate `10fcb51`; lease: none;
  independent review, PR publication, hosted collection, and merge remain; last
  update 2026-09-01.
- **Finding:** modelling the prior `resizing` flag and `steal_accesses` counter
  separately exposed an ABA admission window: a resize can claim, republish,
  and clear the flag between a thief's flag load and counter increment, after
  which the thief enters against the stale generation. Sequential consistency
  on two atomics does not make that compound transition indivisible.
- **Correction:** one `AtomicUsize` now encodes the exclusive owner claim in bit
  zero and each thief contribution as two. A thief's single steady-state RMW is
  ordered either before the claim (so the owner waits) or after it (so the
  thief backs out before storage access). Resize and shared reclamation use the
  same claim. This removes one atomic field and reduces uncontended thief entry
  from one load plus one increment plus one validation load to one increment;
  no latency claim is made without a paired measurement.
- **Acceptance:** the bounded model must cover racing admission/publication,
  retry while claimed, entry after publication, a single held access, and a
  multi-step batch hold. The production stress/value suites, warning-denied
  checks, Rustdoc/doctests, workflow parse, and full workspace Nextest remain
  required before review.
- **Evidence:** the model imports the production gate transition source and
  passes 5/5 focused cases; the complete workflow selection passes 22/22 Loom
  cases across 11 binaries. Scheduler debug and release suites pass 34/34 each;
  the all-feature workspace passes 930/930 tests with 7 configured skips.
  Warning-denied host all-target/all-feature Clippy, warning-denied cfg-Loom
  Clippy, warning-denied AArch64 Windows all-target/all-feature check, workspace
  Rustdoc, workspace doctests, formatting, diff checks, and YAML parsing pass.
  The existing six-row `steal_batch_gate` Criterion smoke passes; no timing
  claim is made because no revision-attested pre-change sample exists for this
  gate correction.
- **Independent correction:** the first review found that the post-publication
  case's channel supplied a separate ordering edge and that the backlog still
  carried the old lease. Candidate `10fcb51` removes that channel, uses only
  relaxed signals before publication, requires a claimed-state backoff, and
  asserts generation one after admission. The focused Loom model and
  warning-denied focused cfg-Loom Clippy pass after the correction; backlog and
  checklist now agree on merge status and no active lease.
- **Review:** independent read-only review of exact PM head `bb6087f` against
  `316bf8f` is GREEN; PR #212 merged with history preserved as `207273e3`;
  every repository check completed successfully after merge.

## MOI-SPIN-BUDGETS-2026-08-27 — Bound the no-yield spin loops [patch] [perf] — done

- **Delivered:** source `c7f670e`, reviewed artifact `b08bcf8`, PR #213 head
  `e2ef5b3`, merge `c43ae3f`; every repository check passed. Finite cooperative
  waits preserve locality; paired raw medians satisfy the non-regression rule
  without an improvement claim.

## MOI-AARCH64-SIMD-CFG-2026-08-27 — cfg-local SIMD lengths [patch] — review

- **Integrator:** Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d`.
- **Lease:** none; provider source and focused verification are complete.
- **Outcome:** scalar fallbacks compile warning-free when the x86 native-vector
  blocks are absent, preserving the existing x86 chunk/tail behavior.
- **Acceptance:** strict host Clippy, Moirai utility tests, and strict AArch64
  all-target compilation pass; Apollo's cross-check proceeds past this provider.
- **Evidence:** warning-denied host Clippy passes; `moirai-utils` Nextest passes
  32/32 in 0.33 seconds; warning-denied AArch64 all-target check passes. Apollo's
  same cross-check reaches its own pre-existing Stockham and cfg-warning debt
  after compiling `moirai-utils` cleanly.

## MOI-INDEXED-SCOPE-ALLOC-2026-08-26 — stack-owned indexed completion [patch] — done

- **Delivered:** provider `1572ec9` / PR #190 / merge `7eefe6e`; exact allocation coverage records zero warmed indexed allocations, and hosted run `33197082236` passes all jobs including corrected liveness contracts.
- **Consumer:** Apollo `5ca9deb4` plus `424b6ae1` merged in PR #125 as `26fc4ab8`; the bounded census records zero warm complex and multidimensional allocations; lease: none.

## MOI-PACKAGE-REPRO-001 — self-contained workspace packaging [patch] — complete

- [x] Add explicit `0.5.0` requirements to the benchmark and integration-test
      path dependencies so Cargo can package the unpublished harnesses.
- [x] Move the runtime examples under `moirai/examples/`, point every
      `[[example]]` target and documentation link at the crate-owned files, and
      set the facade README path to the package-local README.
- [x] Complete binding and harness package metadata and update the route
      contract assertion for the versioned transport dependency.
- Evidence: standalone `cargo package --workspace --locked` packages and
      verifies every workspace member with no warnings; pinned Clippy passes
      with `-D warnings`, Nextest passes `801/801` with 6 configured skips,
      doctests pass with 1 ignored case, and workspace rustdoc completes.

## MOI-SCHED-EXACT-002 — Chase-Lev slot ownership [patch] — complete

- [x] Claim each slot generation before moving a non-`Copy` value so a thief
      cannot race owner reuse of a wrapped ring slot; publish the correct
      generation for bottom pops and advance it for steals.
- [x] Quiesce thief accesses during resize and copy generation state with live
      values while retaining the allocation-free `MaybeUninit` storage contract.
- [x] Add a non-`Copy` drop-count regression and update the artificial
      index-wrap fixture to initialize generation state through the test seam.
- [x] Use strong arbitration CAS operations so `Retry` reports contention,
      not a permitted weak-CAS spurious failure, at single-steal contracts.
- Evidence: pinned Moirai scheduler nextest passes 27/27, including resize,
      index wrapping, single- and eight-thief exactly-once contention, batch
      contention, split-deque consumers, and non-`Copy` drop accounting.
      Pinned warning-denied Clippy passes for all scheduler targets, and the
      pinned Loom Chase-Lev model passes 1/1. Nightly Miri passes all 16
      deque-focused unit tests, including the panic-repair memmove; the full
      19-test crate invocation reaches the remaining NUMA test, which calls
      Themis' Windows NUMA FFI unsupported by Miri.

## ATLAS-MOIRAI-AUDIT-076 — Isolated provider re-verification — closed 2026-08-16

- [x] Re-run the locked workspace gate set from an isolated checkout at the
      current provider head and record exact results in `GAP_ANALYSIS.md`.
- [x] Reconcile release and trusted-publisher status with the existing
      provider boundaries; do not represent the external PyPI account blocker
      as a code defect or release completion.
- [x] Complete the provider-local audit documentation and hosted checks.
      Local gates are green; hosted validation remains the release boundary.

## ATLAS-MOIRAI-BOOK-TEST-2026-08-20 — executable book examples [patch] — in progress

- [x] Enable the shared Pages workflow's `mdbook-test` path with Rust `1.97.0`
      and `cargo-package: moirai-runtime`.
- [x] Add explicit `extern crate moirai;` declarations to both included book
      example sources so rustdoc can resolve the staged facade library.
- Evidence: format, locked runtime example check, example Clippy with
      `-D warnings`, and `mdbook build` pass locally. PR #144 at `4d9bfb0` has
      hosted Rust, Python, and Pages jobs queued; clean hosted execution is the
      acceptance gate because the shared Windows target mixes historical rlibs.

## MOI-SEC-077 — dependency advisory closure — open residuals

- [x] Upgrade the direct PyO3 dependency from `0.22.6` to `0.29.2`, replace
      the removed `Python::allow_threads` API with `Python::detach`, and pass
      the full Rust workspace gate.
- [x] Remove the unused benchmark-only `statistical` dependency and its
      deprecated `rand_os` transitive chain.
- [x] Add `deny.toml` and a pinned supply-chain CI job. The configured check
      passes advisories, bans, licenses, and sources; duplicate-version and
      workspace path-dependency wildcard diagnostics remain warnings.
- [x] Align the pinned action annotation with cargo-deny-action 2.1.1 and set
      `unused-ignored-advisory = "deny"` with structured residual reasons;
      cargo-deny 0.20.2 passes the locked graph with both residual advisories
      encountered and no unused-ignore diagnostics.
- [ ] Replace or remove RSA signing and verification before exposing it to an
      attacker-observable service. `rsa 0.9.10` remains under
      `RUSTSEC-2023-0071`; no safe upstream release exists, so the advisory is
      an explicit cargo-deny residual rather than a hidden pass.
- [ ] Replace the indirect `paste` dependency pulled by the wgpu Metal stack
      when a safe upstream route exists (`RUSTSEC-2024-0436`).

## MOI-CI-EXACT-001 — exact-head Rust and Loom verification [patch] — complete

- [x] Dispatch the committed Rust workflow against the Atlas-pinned provider
      head `a6337abe7fee865d92872c1364d3870a8ee398f1`.
- [x] Verify both the workspace gate and Loom channel models at that exact
      head.
- Evidence: hosted Rust Workspace run `31870317060` passed both `Workspace
  gate` and `Loom channel models` at the exact provider head. The Atlas
  gitlink already matches this head; no pointer change is required.

## MOI-INTERLEAVED-065 — event-synchronized execution coverage [patch] — complete

- Owner: Codex on `test/moirai-interleaved-synchronization`.
- [x] Replace sleep/poll synchronization and wall-clock performance assertions
      in `tests/src/interleaved_execution_tests.rs` with task joins and bounded
      completion channels.
- [x] Exercise the interleaved error result branch and assert the exact
      success/error partition; retain exact work, cascade-stage, and resource
      integrity assertions.
- Evidence: `cargo fmt --all -- --check` and configured Nextest
  `-p moirai-tests interleaved_execution_tests::` pass, 6/6 tests. The test
  module contains no `std::thread::sleep`, `Instant`, or wall-clock polling.

## MOI-MNEMOSYNE-PACKAGE-1 — package identity [patch] — complete

- Owner: Codex on `codex/moirai-mnemosyne-package`.
- [x] Bind the facade and core aliases to their `mnemosyne-memory` package
      identities.
- [x] Refresh dependency resolution and pass the focused core check.

## MOI-THEMIS-PACKAGE-1 — package identity [patch] — complete

- Owner: Codex on `codex/moirai-themis-package`.
- [x] Bind `themis` to package `themis-topology` 0.10.1.
- [x] Refresh dependency resolution and pass focused gates.
- [x] Merge before rerunning dependent Hephaestus provider CI.

## MOI-NUMA-002 — facade NUMA policy reaches the scheduler [minor] [arch] — complete

- Owner: Codex on `codex/moirai-numa-policy`.
- [x] Forward the facade `numa_aware` value through the core and executor
      feature seams.
- [x] Preserve topology-aware defaults and make explicit disablement skip
      worker-node assignment construction.
- [x] Add facade configuration and scheduler assignment regressions.
- [x] Synchronize README, changelog, backlog, and ADR-027.
- [x] Pass the exact default-head hosted Rust workspace gate.
- Evidence: local formatting, standalone locked metadata, warning-denied
  Clippy, configured Nextest 118/118 with two configured skips, seven
  doctests, and warning-denied rustdoc pass. Default-head Rust Workspace run
  `31787962637` and Python Bindings run `31787962649` pass at merge commit
  `38e936a`.

## MOI-THEMIS-CPU-001 — provider-owned CPU topology [patch] — complete

- Owner: Codex; delivered through PR #118 and merged as `57c4ec4`.
- Scope: default worker-count and chunk-sizing decisions in `moirai-core`,
  `moirai-executor`, `moirai-iter`, `moirai-parallel`, and
  `moirai-scheduler`; benchmark contract fixes are delivered separately under
  ISSUE-216.
- [x] Route each default CPU-count decision through Themis topology detection
      with the existing standard-library fallback preserved.
- [x] Add value-semantic partition-order coverage for the Melinoe-backed
      parallel adapter.
- [x] Pass focused all-target checks, warning-denied Clippy, configured
      Nextest (430/430 across the affected crates and 68/68 benchmarks),
      doctests, and rustdoc.
- [x] Replace host-dependent priority timing and worker-blocking ABA test
      synchronization with event-gated queue coverage and joined task results;
      the complete `moirai-tests` package passes 36/36 under Nextest.
- [x] Merge PR #118 after the hosted required checks passed, then advance the
      Atlas gitlink to the exact current default head `2f639dc`.

## MOI-CI-224 — Rust workspace gate [patch] — complete

- Owner: Codex on `codex/fix-atlas-sha`.
- [x] Add a pull-request and main-branch Rust workspace gate for formatting,
      warning-denied Clippy, configured Nextest, doctests, and rustdoc.
- [x] Pin all third-party actions to commit SHAs, constrain permissions, and
      bound the job runtime.
- [x] Verify the gate commands locally against the affected workspace packages.
- [x] Restore standalone Git source records for the Melinoe, all six Mnemosyne,
      and Themis packages in `Cargo.lock`; the hosted diagnostic also refreshed
      the stale registry checksums and versions required by the current graph.
- [x] Remove all `[[patch.unused]]` records emitted only by the Atlas
      development overlay; the standalone lock must contain no overlay state.
- [x] Confirm the hosted workflow is green after the standalone lock refresh and
      deterministic test repair: Rust Workspace run `31566422283` passed at
      `9ec4b02`; current default documentation, Python, and Pages workflows
      pass at `2f639dc`.

## MOI-PAR-062 — borrowing parallel scope [minor] — complete

- Provider acceptance is complete; the borrowing scope facade, direct
  `moirai-core` ownership, higher-ranked lifetime bound, borrowed completion,
  return-value coverage, and focused local/hosted gates are all delivered.
  No downstream consumer pin is pending for this item.

- Owner: Codex `/root` (composed from preserved peer work).
- Scope: the `moirai-parallel` borrowing scope facade, its direct dependency
  ownership, value-semantic tests, and release documentation.
- Acceptance: multiple tasks borrow caller-owned values, complete before the
  scope returns, preserve a body return value, and compile without exposing an
  escaping scheduler lifetime.
- [x] Preserve and complete the peer's public scope facade.
- [x] Add direct `moirai-core` ownership and the higher-ranked lifetime bound.
- [x] Add borrowed completion and return-value coverage.
- [x] Pass focused local and exact-head hosted gates; merge Moirai.

## MOI-SCHED-061 — bounded indexed admission [patch] — provider and downstream complete

- Owner: Codex `/root` (stale-peer takeover after one hour without a write or
  commit in the claimed scope).
- Scope: indexed scheduler admission, its diagnostics and value-semantic
  saturation tests, release documentation, and the downstream Kwavers
  serialization workaround. Other scheduler policies are non-goals.
- Acceptance: a full worker admission queue executes each rejected indexed
  chunk exactly once on the caller, map-reduce preserves the mathematical
  result, caller-run panics become `SpawnFailed(Panicked)` only after scheduled
  scope work drains, the scheduler remains reusable, and the recovery event is
  observable without allocating on the healthy path.
- [x] Preserve the stale peer's caller-runs intent.
- [x] Add one shared panic boundary for inline indexed work.
- [x] Add a relaxed monotonic admission diagnostic.
- [x] Add deterministic saturated fan-out, reduction, panic, and reuse coverage.
- [x] Pass focused local and exact-head hosted gates; merge Moirai.
- [x] Downstream-only follow-up: Kwavers consumed the merged Moirai pin and
      closed the admission-specific serialization workaround in `KW-CI-068`.
      The broader architecture workflow may still use `--test-threads=1` for
      unrelated workload isolation; that is not an admission workaround.
      This does not block the completed Moirai provider implementation.
- [x] Provider hygiene follow-up: document the scheduler diagnostics surface,
      add the module-level allowance required by the pinned Melinoe 0.9.0
      `thread_cached!` expansion, and preserve the shared-provider boundary
      without a local cache duplicate.
- Evidence (2026-08-06): rustfmt, diff check, all-target/all-feature check,
      warning-denied Clippy, workspace Nextest **784/784** (6 configured
      skips), doctests, `moirai-parallel` **32/32**, and
      `moirai-executor` **91/91** (1 configured skip) pass offline. The
      scheduler wait test passed in the definitive run; an earlier isolated
      failure was non-reproducible across five focused retries and is retained
      as a stability watchpoint rather than hidden.

## MOI-REL-061 — Rust crate releases [patch] — in progress

- Owner: Codex `/root`.
- Scope: collision-free facade identity, published Mnemosyne package aliases,
  registry metadata, locked archives, and crates.io release automation.
- Acceptance: every reusable workspace crate packages from a clean checkout,
  publishes in dependency order, resolves from crates.io, and uses the OIDC
  release workflow for subsequent versions.
- [x] Select the available `moirai-runtime` facade package identity while
  preserving `use moirai::...` for Rust consumers.
- [x] Bind Mnemosyne dependencies to their registry package identities.
- [x] Pass the clean-checkout metadata and workspace nextest gates.
- [ ] Publish every reusable workspace crate and verify the sparse index.
- [ ] Register each crates.io Trusted Publisher against the release workflow.

## MOI-REL-060 — Python wheel releases [patch] — blocked

- Owner: Codex `/root`.
- Scope: `moirai-python` distribution metadata and documentation, a pinned
  cross-platform release workflow, the protected GitHub publishing
  environment, release-facing root documentation, the Linux shared-memory
  size boundary that blocks the binding gate, and this owner-keyed entry.
  Other native runtime behavior and workspace crates are non-goals.
- Acceptance: a GitHub Release tagged `moirai-python-v<version>` builds locked
  wheels for supported CPython versions on Linux, Windows, and macOS; installs
  and imports each wheel; validates distribution metadata against the tag;
  attests and attaches the exact artifacts to the GitHub Release; and publishes
  those same wheels to PyPI through GitHub OIDC.
- [x] Make Cargo the Python distribution version source of truth.
- [x] Add pinned cross-platform wheel CI and release workflows.
- [x] Synchronize Python, root, changelog, toolchain, and Nextest contracts.
- [x] Build, install, import, and exercise a production wheel locally.
- [x] Pass workflow lint and focused Rust/Python binding gates.
- [x] Protect the `pypi` environment with the `moirai-python-v*` tag policy.
- [x] Pass exact-head hosted CI.
- [x] Merge the release PR.
- [ ] Register the PyPI pending trusted publisher after account verification.
- Blocker: PyPI rejects `ryanclanton@outlook.com`; registration reopens when
  the account has a PyPI-accepted email address and completes verification.
- Evidence: checksum-verified actionlint 1.7.12 accepts both workflows; locked
  Cargo metadata and Rust 1.95 formatting, warning-denied all-target Clippy,
  configured Nextest 1/1, doctests, and warning-clean rustdoc pass. A locked
  CPython 3.13 wheel builds as `moirai-python` 0.4.0, installs into an isolated
  environment, imports, reports the requested two-worker native lifecycle, and
  passes both Python tests. GitHub environment `pypi` accepts only
  `moirai-python-v*` tags. Hosted run `29799529159` then exposed an unchecked
  Unix `usize`-to-`off_t` conversion in `moirai-core`; the owner-local fix
  validates zero and out-of-domain lengths before acquiring a shared-memory
  descriptor and covers both boundaries through the public `SharedMemory`
  contract. The Windows host passes warning-denied all-target core Clippy and
  70/70 configured Nextest cases. Replacement hosted run `29800011266` passes
  the Windows wheel job and exposes a pre-existing unconditional non-Linux
  `AtomicBool` import in the Linux binding closure; that import is now
  target-gated. Exact-head hosted run `29800253930` passes formatting,
  warning-denied binding lint, native binding and Unix shared-memory boundary
  tests, binding doctests, and all three production wheel build/install/import
  smoke jobs. PR #82 carries the merge-ready delivery. PyPI publisher
  registration remains blocked on account verification.

## MOI-SCOPE-059 — scoped multi-job memory safety [patch] — done

- Owner: Codex Tyche integration.
- Scope: `moirai-executor/src/schedule/{job,runtime/scheduler}` and focused
  scoped-dispatch regression coverage.
- [x] Reproduce Tyche's multi-job borrowed-slice access violation without
      Tyche.
- [x] Publish the final zero count while holding the wait lock, and require
      both caller and worker waiters to acquire that lock before destroying the
      stack-owned scope state.
- [x] Verify the 64-round borrowed-chunk regression, the bounded one-completion
      Loom model, and the complete executor package.
- Evidence: configured Nextest passes `moirai-executor` 88/88 with one
  cfg-gated skip; warning-denied all-target/all-feature Clippy, rustfmt,
  doctests, and rustdoc pass. The bounded Loom model passes 1/1. Miri cannot
  reach the regression on Windows because Themis NUMA detection calls the
  unsupported `GetNumaHighestNodeNumber` FFI; it reports no result for this
  invariant. PR 81 contains the delivery; Moirai has no repository engineering
  workflow beyond the non-gating Copilot workflow.

## Moirai 0.4.0 release artifact closure [patch]

- [x] Take over the stale release-artifact increment after one hour without
      file or commit activity.
- [x] Synchronize the workspace version, changelog release section, checklist
      target, and benchmark artifact contract at 0.4.0.
- [x] Verify the focused artifact contract, formatting, and warning-denied
      benchmark test target before publication.
- Evidence: configured Nextest passes the artifact contract 1/1 with 67
  unrelated cases filtered; nightly rustfmt and warning-denied Clippy for the
  benchmark contract target pass.

## MOI-ASYNC-058 — synchronization stabilization [patch]

- [x] Take over the stale uncommitted synchronization/codec lane on `main`.
  Scope: `moirai-async/src/sync/{broadcast,mpsc,wait_queue}.rs`, timer
  regression coverage, `moirai-http/src/codec.rs`, affected examples, and this
  provider PM scope.
- [x] Verify FIFO/cancellation behavior, broadcast retention, and the
  cancellation-compaction regression through value-semantic configured Nextest
  coverage; run warning-denied Clippy and formatter checks before publication.
- [x] Absorb audit findings into the provider backlog/gap register and delete
  the untracked report instead of retaining a parallel status artifact.
- Evidence: configured Nextest passes `moirai-async` 88/88 and `moirai-http`
  9/9; warning-denied workspace Clippy, rustfmt, rustdoc, and doctests pass.

## Provider default-source convergence [major]

- [x] Remove direct Themis, Mnemosyne, and Melinoe revisions plus the local
  Melinoe patch from the workspace dependency SSOT.
- [x] Record ADR-033 (recorded as ADR 016 before the duplicate-number
  resolution): merged Mnemosyne 0.5/Core 0.2 requires Rust 1.95, so the
  workspace advances from 0.3.1 to 0.4.0 without a compatibility branch.
- [x] Refresh the lockfile against merged provider heads and prove one source
  identity for Melinoe, Themis, and Mnemosyne.
- [x] Verify the focused GPU consumer with Rust 1.95 compilation,
  warning-denied Clippy, configured Nextest, doctests, and rustdoc.
- [x] [patch] Preserve SPSC send-before-close ordering with a value-semantic
  drain regression that contains no wall-clock synchronization.
- Evidence: Rust 1.95 accepts `moirai-gpu` and Rust 1.94 rejects the declared
  package graph; warning-denied focused Clippy passes; Nextest passes 10/10;
  doctests pass 0/0; rustdoc is warning-clean; each provider resolves to one
  lock source identity; the SemVer major comparison reports no API check
  failures.

## Phase 31: Channel ordering model coverage

- [x] Re-audit the ordering residual: the MPMC waiter model already landed in
  `moirai-core/tests/loom_mpmc_waiter.rs` at `2ea17bb`; the missing primitive
  is the SPSC ring's publication/reclamation model.
- [x] Add the bounded capacity-two SPSC model with three messages, explicit
  release/acquire head/tail edges, value-semantic FIFO assertions, and a
  recorded preemption bound of four.
- [x] Add a hosted `Loom channel models` job that runs both MPMC and SPSC
  models under `RUSTFLAGS=--cfg loom` with the locked release profile.
- Evidence: `cargo fmt --all -- --check` passes locally. The Atlas overlay
  prevents the locked local metadata gate from running because preserved local
  first-party patches do not match this provider's committed lock; hosted CI
  is the authoritative clean-checkout oracle for the model job.

## Phase 32: Async wake deduplication ordering

- [x] [patch] Replace the async executor's `is_queued` `SeqCst` clear/swap with
  Relaxed operations. The flag only linearizes enqueue deduplication; the
  queue's slot sequence publishes task ownership with Release/Acquire.
- [x] Add `moirai-async/tests/loom_wake_dedup.rs`, exhaustively covering the
  dequeue/clear versus wake/swap race and asserting no duplicate or lost queue
  entry.
- [x] Extend the hosted Loom job to run the async executor model with the
  existing MPMC and SPSC models.
- Evidence: PR #131 merged at default `fd517fe`; exact-head workspace/Loom run
  `31800148163` and bindings/wheels run `31800148178` pass. The completion
  guard remains Acquire/Release and scheduler/MPMC protocols are unchanged.

## Phase 33: PAL reactor stop-flag ordering

- [x] [patch] Reduce `IoReactor::running` start, loop, and stop accesses from
  `SeqCst` to Relaxed. The flag carries only loop-control state; `stop()` keeps
  its independent platform wake operation for progress from a blocked poll.
- [x] Verify the focused reactor and async network stop paths through the
  hosted workspace and binding/wheel gates.
- Evidence: PR #132 merged at default `8830f1b`; exact-head workspace/Loom run
  `31800607186` and bindings/wheels run `31800607152` pass. The provider exact
  head has no production `SeqCst` accesses in
  `moirai-pal/src/reactor/core.rs`.

## Phase 34: Connection-pool reservation ordering

- [x] [patch] Reduce `ConnectionPool::reserved_connections` admission and
  release accounting to Relaxed operations. Admission serialization remains
  under the active-connections mutex, and every reservation increment has one
  paired decrement.
- [x] Add the bounded Loom model covering two serialized admissions racing one
  paired cancellation, with capacity and final-accounting assertions.
- [x] Verify the exact provider head through the workspace/Loom and
  bindings/wheel gates.
- Evidence: PR #133 merged at default `f766c6d`; exact-head workspace/Loom run
  `31801180700` and bindings/wheels run `31801180691` pass.

## Phase 30: NUMA helper removal and channel hierarchy closure

- [x] [major] Delete the unconsumed `moirai_iter::numa` API and its obsolete
  Rayon comparison benchmark. Themis owns placement, Mnemosyne owns allocation,
  and `moirai-parallel` owns scheduler-backed data-parallel work; no source
  consumer imports the removed iterator helper.
- [x] [patch] Split hybrid and MPMC channel implementation into vertical state,
  send, receive, future, and test modules. `HybridChannel<T>` is a zero-sized
  factory; endpoint halves own each live synchronization primitive.
- Evidence: `cargo nextest run -p moirai-core` 69/69;
  `-p moirai-iter` 185/185 (2 cfg-skips); `-p moirai-benchmarks` 68/68;
  warning-denied `moirai-core` Clippy; and formatter checks pass. This is
  compile-time/API-surface and value-semantic test evidence, not a throughput
  claim.

## Phase 29: Indexed caller-region flattening

- [x] [patch] Mark the caller lane while it participates in indexed fan-out
  and map/reduce so nested regions flatten on every outer lane, not only
  scheduler workers.
- [x] Verify exact caller/worker lane identity and value sums through nextest,
  then run warning-denied Clippy.
- [x] Pin the commit in RITK and pass the unchanged masked-CMA consumer gate.
- Evidence: `cargo nextest run -p moirai-executor` passed 83/83 with one
  cfg-gated skip; warning-denied all-target/all-feature Clippy passed. RITK
  pins merged Moirai main in its Atlas checkout action, and `cargo nextest run
  -p ritk-registration masked_cache --all-features --status-level fail` passes.

## Phase 28: Melinoe executor capability migration

- [x] [major] Construct Melinoe's validated `ParallelExecutor` next to the real
  Moirai scheduler bridge with an explicit exact-once, completion, and lifetime
  safety proof.
- [x] [major] Remove the raw function-pointer registration call and update the
  workspace Melinoe contract to 0.9.0 and Mnemosyne facade to 0.3.0.
- [x] Verify the real Melinoe routing path plus Moirai executor Clippy, 83/83
  nextest (one cfg-gated test skipped), doctests, and rustdoc against Melinoe
  `bb07447`, Themis `6140468`, and Mnemosyne 0.3.0 at `df2994f`.

## Phase 27: GPU pollster boundary removal
- [x] [patch] Added `moirai_executor::block_on` as the Moirai-owned
  current-thread parking wait primitive for synchronous async boundaries.
- [x] [patch] Replaced `moirai-gpu`'s `GpuTaskAdapter` `pollster::block_on`
  call with `moirai_executor::block_on` and removed `pollster` from the
  `wgpu-backend` feature dependency list.
- [x] [patch] Added a benchmark source contract that rejects reintroducing
  `pollster` into `moirai-gpu`'s manifest or task adapter.
- Evidence: `rustup run nightly rustfmt --edition 2021
  moirai-executor\src\lib.rs moirai-gpu\src\task.rs
  benchmarks\tests\benchmark_contracts\source_contracts.rs`;
  `rustup run nightly cargo check -p moirai-gpu --features wgpu-backend`;
  `rustup run nightly cargo check -p moirai-executor --no-default-features`;
  `rustup run nightly cargo tree -p moirai-gpu --features wgpu-backend -i
  pollster` reports no matching package; `rustup run nightly cargo nextest run
  -p moirai-benchmarks gpu_task_adapter_uses_moirai_block_on_not_pollster
  --status-level fail --no-fail-fast` passed 1/1.

## Phase 26: Socket stale-wake regression ✅
- [x] [patch] Added a real loopback TCP regression for the July 2 stale-wake
  async bug: `timeout(stream.read(...))` completes through the timer while the
  socket read waker remains registered, then peer readability wakes the stale
  reactor slot after the async task has completed.
- Evidence: `rustup run nightly cargo nextest run -p moirai-async
  timeout_read_stale_socket_wake_does_not_repoll_completed_task
  --status-level fail --no-fail-fast`.

## Phase 25: Transport stale export cleanup
- [x] [patch] Removed `moirai_transport::core_zero_copy`, a stale re-export of
  deleted `moirai_core::communication::zero_copy`, so Atlas consumers compile
  against the current `moirai_core::communication` surface.
- Evidence: `rustup run nightly cargo fmt -p moirai-transport --check`;
  `rustup run nightly cargo check -p moirai-transport`.

## Phase 24: Stateful Chunk Parallel Provider API ✅
- [x] [patch] Added `moirai_parallel::for_each_chunk_mut_with_state` so
  consumers can run mutable chunk kernels with one reusable scratch state per
  scheduled worker shard.
- [x] [patch] Re-exported the API from `moirai-parallel` and covered it with a
  value-semantic test that proves every chunk is written from reusable state.
- [x] [patch] Added `for_each_chunk_triple_mut_enumerated_with` and
  `for_each_chunk_quad_mut_enumerated_with` for provider-owned fused updates
  across three or four equal-length mutable output buffers.
- [x] [patch] Re-exported the multi-output chunk APIs and covered them with
  value-semantic tests that prove chunk indices and all output buffers are
  written from the caller-provided closure.
- Evidence: `cargo fmt -p moirai-parallel --check`; `cargo check -p
  moirai-parallel`; `cargo nextest run -p moirai-parallel
  for_each_chunk_ --status-level fail --no-fail-fast` (6/6).

## Phase 23: Task Registry Stable Slot Access ✅
- [x] [patch] Kept `TaskStateBlock` slot storage private behind
  `UnsafeCell`-based `get`/`insert`/`clear`/`states` methods so lifecycle
  tokens receive stable `NonNull<TaskState>` pointers without exposing block
  internals.
- [x] [patch] Routed registry production paths, diagnostics, active/completed
  counts, and cleanup through the block accessor API.
- [x] [patch] Updated benchmark source-contract assertions to pin the dense
  `UnsafeCell<Option<TaskState>>` representation and zero-allocation stable-slot
  invariant.
- Evidence: `rustup run nightly cargo fmt -p moirai-executor --check`; `rustup
  run nightly cargo check -p moirai-executor --all-targets`; `rustup run
  nightly cargo clippy -p moirai-executor --all-targets -- -D warnings`;
  `rustup run nightly cargo nextest run -p moirai-executor` (61/61, 1 skipped);
  `rustup run nightly cargo doc -p moirai-executor --no-deps`.

## Phase 22: Sharded Task Registry ✅
- [x] [patch] Replaced the hybrid executor's single `Arc<Mutex<TaskRegistry>>`
  with `Arc<ShardedTaskRegistry>` so task registration and metadata reads route
  through per-shard registry locks.
- [x] [patch] Added sharded registry coverage proving dense global IDs,
  global-to-local metadata reporting, lifecycle-token completion, and unknown
  task lookup behavior.
- [x] [patch] Updated manager status/stat/wait paths to use the sharded
  registry facade directly and removed stale warning sources.
- Evidence: `rustup run nightly cargo fmt -p moirai-executor --check`; `rustup
  run nightly cargo check -p moirai-executor --all-targets`; `rustup run
  nightly cargo clippy -p moirai-executor --all-targets -- -D warnings`;
  `rustup run nightly cargo nextest run -p moirai-executor` (62/62, 1 skipped);
  `rustup run nightly cargo doc -p moirai-executor --no-deps`; `git diff
  --check`.

## Phase 21: Executor Lockfile and Rustdoc Hygiene ✅
- [x] [patch] Synchronized `Cargo.lock` with the existing `cfg(loom)`
  `moirai-executor` dev-dependency so locked builds resolve the model-checking
  dependency edge.
- [x] [patch] Removed redundant explicit `WorkScheduler` Rustdoc link targets
  from `moirai-executor::hybrid`, keeping the package rustdoc gate
  warning-clean.
- Evidence: `rustup run nightly cargo fmt -p moirai-executor --check`;
  `rustup run nightly cargo check -p moirai-executor --all-targets`; `rustup
  run nightly cargo clippy -p moirai-executor --all-targets -- -D warnings`;
  `rustup run nightly cargo nextest run -p moirai-executor`; `rustup run
  nightly cargo test --doc -p moirai-executor`; `rustup run nightly cargo doc
  -p moirai-executor --no-deps`; `git diff --check`.

## Phase 20: Async RwLock Waiter Map ✅
- [x] [patch] Completed the `moirai-async::sync::RwLock` waiter storage
  migration from `VecDeque` tuples to keyed `BTreeMap<u64, RwWaiter>` state.
- [x] [patch] Routed read/write future poll, cancellation, writer grant, and
  reader-batch wakeup through the same waiter state so the O(log n) removal
  contract compiles and preserves FIFO-by-monotonic-id handoff.
- [x] [patch] Fixed the `ConnectionId` Rustdoc peer-address link to
  `std::net::SocketAddr`, keeping rustdoc warning-clean.
- Evidence: `rustup run nightly cargo fmt -p moirai-async --check`; `rustup run
  nightly cargo check -p moirai-async --all-targets`; `rustup run nightly cargo
  clippy -p moirai-async --all-targets --all-features -- -D warnings`; `rustup
  run nightly cargo nextest run -p moirai-async`; `rustup run nightly cargo test
  --doc -p moirai-async`; `rustup run nightly cargo doc -p moirai-async
  --all-features --no-deps`; `git diff --check`.

## Phase 19: Concurrent Stream Module Export ✅
- [x] [minor] Completed the `parallel_stream` -> `stream` module rename by
  exporting `moirai_iter::stream` from `moirai-iter`.
- [x] [minor] Renamed the stream extension trait and methods to
  `ConcurrentStreamExt` / `concurrent_*`, matching the bounded-concurrency
  contract rather than promising CPU parallelism for every async item future.
- [x] [minor] Added fused `concurrent_filter_map` and `concurrent_filter`
  stream adapters with value-semantic coverage.
- Evidence: `cargo fmt --check -p moirai-iter`; `cargo clippy -p moirai-iter
  --all-targets --all-features -- -D warnings`; `cargo nextest run -p
  moirai-iter stream` -> 10 passed; `cargo doc -p moirai-iter --all-features
  --no-deps`.

## Phase 18: Default Provider Feature Contract
- [x] [patch] Added default `parallel` and `mnemosyne-memory` features to every
  Moirai package. Existing Mnemosyne-backed crates forward `mnemosyne-memory`
  to the established `mnemosyne` provider feature; non-provider leaf crates use
  zero-dependency markers.
- [x] [patch] Applied rustfmt-required import/closure formatting in existing
  Moirai iterator/reactor files so the formatting gate is clean.
- Evidence: `cargo metadata --no-deps --locked --format-version 1`; full Atlas
  feature-policy metadata audit; `cargo fmt --check`; `git diff --check`.
  Residual: compile/test gates were blocked before rustc by denied access to
  `target/debug/.cargo-lock`.

## Phase 17: Mnemosyne Worker Maintenance Integration ✅
- [x] Registered Moirai's global scheduler as Melinoe's `std` partition executor via pushed Melinoe commit `8140882`, so branded partition writes route through Moirai workers.
- [x] Added a value-semantic scheduler test proving Melinoe partition routing writes every branded cell exactly once through the registered Moirai executor.
- [x] Removed dead thread-local cache declaration from `moirai-core::pool::GlobalPool::get`; the active implementation uses the global pool path.
- [x] Added `mnemosyne` as an optional `moirai-executor` dependency and default feature.
- [x] Forwarded the top-level `moirai/mnemosyne` feature into `moirai-executor/mnemosyne`.
- [x] Routed idle worker-loop maintenance through `mnemosyne::Mnemosyne` defragmentation sweeps using the provider's top-level backend selector.
- [x] Updated Moirai's Mnemosyne pin to `938d0c2bc094d3bbe7745d68d60e05a531e0cfc2` so the executor consumes the exported provider selector.
- [x] Verification: `cargo fmt --check`; `cargo check -p moirai --locked`; `cargo test -p moirai-executor --features mnemosyne --locked`; `cargo clippy -p moirai-executor -p moirai --all-targets --all-features --locked -- -D warnings`.
- Evidence: compiler diagnostics, value-semantic scheduler/executor tests under the Mnemosyne feature, and clippy diagnostics.

## Phase 16: Default Parallel Branding Integration ✅
- [x] Enabled `moirai-parallel` Mellinoe integration by default so the parallel crate exposes branded partitioning without opt-in feature plumbing.
- [x] Added `melinoe` to the `moirai` facade default feature set alongside existing `parallel` and `mnemosyne` defaults.
- [x] Replaced serial async iterator map/filter/for_each execution with bounded concurrent polling while preserving ordered map/filter results.
- [x] Verification: `cargo fmt --check`; `cargo test -p moirai-iter execution::tests::async_context --locked`; `cargo test -p moirai-parallel -p moirai --locked`; `cargo clippy -p moirai-iter -p moirai-parallel -p moirai --all-targets -- -D warnings`; `cargo test --locked --workspace --examples`.
- Evidence: value-semantic async ordering tests, Mellinoe partitioning tests under default features, clippy diagnostics, and workspace example execution.

## Phase 15: Code Quality & Design Principles Enforcement ✅
- [x] **MAJOR**: Fixed clippy errors (match_same_arms, manual_let_else) for clean builds
- [x] **MAJOR**: Implemented underscored parameters (priority/locality hints in HybridExecutor)
- [x] **MAJOR**: Extracted magic numbers to named constants (SSOT/SOC compliance)
- [x] Applied cargo fix and cargo fmt for consistent code style
- [x] Fixed mixed attribute styles and redundant code patterns
- [x] Ensured no prohibited naming patterns (*_old, *_new, *_enhanced, etc.)
- [x] Verified no deprecated/redundant components requiring removal
- [x] Applied design principles (SOLID, CUPID, GRASP, DRY, KISS, YAGNI)
- [x] Maintained single implementations with flexible configuration
- [x] Enforced zero-cost abstractions and stdlib iterator usage

## Phase 14: Critical Infrastructure Fixes ✅
- [x] **MAJOR**: Fixed HybridExecutor to actually execute tasks (auto-start workers)
- [x] **MAJOR**: Fixed spawn_blocking result communication via proper channels
- [x] **MAJOR**: Fixed spawn_async implementation with polling-based runtime
- [x] Fixed clippy warnings that prevented clean builds (-D warnings compliance)
- [x] Fixed method naming conflicts (XorShiftRng API)
- [x] Replaced cfg(disabled) with proper Cargo feature flags
- [x] Fixed documentation compilation errors and broken links
- [x] Verified all examples work end-to-end (basic_usage, async_timer)

## Phase 13: Code Optimization and Cleanup ✅
- [x] Review and clean codebase following design principles
- [x] Consolidate channel implementations (DRY/SSOT)
- [x] Extract common iterator patterns into base module
- [x] Simplify sync module - remove redundant wrappers
- [x] Implement ExecutionBase trait for all contexts
- [x] Fix all build errors across workspace
- [x] Apply SOLID, CUPID, GRASP, DRY, KISS, YAGNI principles
- [x] Update README with optimization details

## Phase 12: Iterator System Enhancements ✅
- [x] Advanced iterator combinators (chunks, windows, etc.)
- [x] SIMD-optimized iterators
- [x] Cache-optimized iteration patterns
- [x] Streaming and batching support
- [x] Channel fusion for zero-copy pipelines
- [x] Adaptive execution strategies
- [x] Prefetching and memory optimization
- [x] NUMA-aware iteration

## Phase 11: Zero-Copy Transport ✅
- [x] Memory-mapped ring buffers
- [x] Zero-copy channel implementation
- [x] Shared memory transport
- [x] RDMA-style operations
- [x] Efficient serialization
- [x] Adaptive batching
- [x] Flow control mechanisms

## Phase 10: Unified Transport Layer ✅
- [x] Transport trait abstraction
- [x] In-memory transport
- [x] IPC transport foundation
- [x] Network transport skeleton
- [x] Message routing
- [x] Connection management
- [x] Transport selection logic

## Phase 9: Advanced Scheduler ✅
- [x] NUMA-aware scheduler
- [x] CPU topology detection
- [x] Work migration policies
- [x] Adaptive load balancing
- [x] Priority scheduling
- [x] Deadline scheduling
- [x] Resource quotas

## Phase 8: Metrics System ✅
- [x] Core metrics collection
- [x] Task execution metrics
- [x] Scheduler performance metrics
- [x] Memory usage tracking
- [x] Latency histograms
- [x] Throughput monitoring
- [x] Metric aggregation

## Phase 7: Async Runtime ✅
- [x] Async executor implementation
- [x] Future polling mechanism
- [x] Async task spawning
- [x] Timer implementation
- [x] I/O reactor integration
- [x] Async synchronization primitives

## Phase 6: Synchronization Primitives ✅
- [x] Fast mutex implementation
- [x] Reader-writer locks
- [x] Condition variables
- [x] Barriers
- [x] Semaphores
- [x] Atomic operations
- [x] Lock-free data structures

## Phase 5: Coroutine Support ✅
- [x] Coroutine trait definition
- [x] Yield mechanism
- [x] Coroutine scheduler
- [x] State management
- [x] Coroutine handles
- [x] Integration with task system

## Phase 4: Error Handling ✅
- [x] Error type hierarchy
- [x] Result types
- [x] Error propagation
- [x] Panic handling
- [x] Error recovery
- [x] Diagnostic information

## Phase 3: Memory Pool ✅
- [x] Object pool implementation
- [x] Arena allocator
- [x] Memory recycling
- [x] Cache-aligned allocation
- [x] NUMA-aware allocation
- [x] Memory statistics

## Phase 2: Work-Stealing Scheduler ✅
- [x] Chase-Lev deque implementation
- [x] Worker thread management
- [x] Task stealing logic
- [x] Load balancing
- [x] Scheduler benchmarks

## Phase 1: Core Architecture ✅
- [x] Task abstraction
- [x] Executor trait
- [x] Basic scheduler interface
- [x] Thread pool implementation
- [x] Basic task spawning

## Next Steps
- [x] Comprehensive test suite (39+ core tests passing)
- [x] Example applications (basic_usage, async_timer working)
- [x] Documentation improvements (SSOT and consolidation notes)
- [ ] Performance benchmarks validation
- [ ] API stabilization  
- [ ] Production readiness review
- [x] SSOT consolidation: zero-copy communication primitives live under
      `moirai_core::communication`
- [x] Iterator windows/chunks consolidated under `moirai_iter::windows`
- [x] Placeholder cleanup: replaced stubs with explicit unsupported errors or working code
- [x] Zero-copy send returns value on failure to prevent data loss
- [x] **Critical Infrastructure**: Fixed executor to actually run tasks (was completely broken)
