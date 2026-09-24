# Moirai Development Checklist

**Target**: Unreleased

## MOI-WEBVIEW-PREVIEW-CAPTURE-TIMEOUT-2026-09-24 - Complete a preview capture under parallel hosts [patch] [verification] - todo

- **Outcome:** `installed_runtime_captures_rendered_preview` passes in every
  `--run-ignored all` run of `moirai-pal --features webview2`.
- **Evidence:** on origin/main `aeb2fb0c` (WebView2 155) it times out in 6 of
  15 full parallel runs with `TimedOut: WebView2 callback did not complete
  before the finite deadline` from `capture_preview_png`. It passes run alone.
  The capture starts right after host creation without waiting for
  `NavigationCompleted`, in a hidden window.
- **Next step:** find out whether the `CapturePreview` completion is held back
  (for example while the navigation is pending, or for a hidden document
  while sibling hosts share the user-data folder) or dropped, then fix it in
  the host or in the test's synchronization.
- **Acceptance:** 30 consecutive full ignored-test runs have zero capture
  timeouts, and the 30 s wait is unchanged.
- **Scope:** `moirai-pal/src/windows/webview/{host/view.rs,tests.rs}`.

## MOI-MPMC-NOTIFY-FENCE-COST-2026-09-24 - Recover the per-push notifier fence [patch] [perf] - todo

- **Outcome:** the bounded `MpmcChannel` and `HybridSender` notifiers stop
  paying a `SeqCst` fence on pushes that no parked receiver can depend on,
  without a reachable lost wakeup.
- **Context:** the fix for the bounded round-trip hang (PR #451) restored the
  fence on every push: the removed elision read the consumer cursor before
  publishing, and a receiver can drain the items ahead of the push in that
  window and park. The elision's commit (`ae5bcedc`) recorded the fence as a
  13-64% `bounded_channel_matrix/moirai_mpmc` cost at 4 and 8 producers; that
  figure is unreverified against current main.
- **Candidate:** a receiver parks only after seeing `tail == head` post
  registration, spinning while a claimed slot is unpublished; with `SeqCst` on
  the tail claim, the cursor read, and the receiver's tail read, only the push
  that claims the receiver's head position needs the fence. Loom model first,
  extending `DrainRing` in `moirai-core/tests/loom_mpmc_waiter.rs` with a
  second producer.
- **Acceptance:** the extended loom models admit no lost wakeup; the
  4/8-producer bench cells beat the fenced baseline on pinned cores;
  `mpmc_roundtrip_preserves_multiset` passes 1,000 consecutive runs.
- **Scope:** `moirai-utils/src/queue/ring.rs`,
  `moirai-core/src/channel/{mpmc,hybrid}`,
  `moirai-core/src/communication/ring_buffer.rs`.

## MOI-PARKED-POOL-FIRST-REGION-2026-09-17 — Bring a parked pool to full width faster [patch] [perf] — todo

- **Outcome:** a data-parallel region submitted after the workers parked (idle
  past `SPIN_LIMIT`, about 60 µs) runs close to its hot-pool time.
- **Consumer driver:** apollo `APOLLO-REAL-3D-PASS-COUNT`
  (`../apollo/backlog.md#apollo-real-3d-pass-count`). Its 64³ forward z sweep,
  about 66 tasks of 64 KB through `for_each_unit_task_mut_with` on the
  machine-wide pool, takes 21.6-22.4 µs directly after another region and
  53.4-56.6 µs as the first region after about a millisecond of serial work:
  same code and buffers, so about 31-35 µs is waking the pool, above the
  documented ~8 µs per parked worker. Any entry whose caller does serial work
  between calls pays it.
- **Spike:** a bench timing one fixed region hot and after idle gaps of 0.1, 1
  and 10 ms, attributing the after-idle time to unpark order (one wake per
  submitted task vs a batch), per-worker wake cost, and the submitter's own
  share of the tasks.
- **Acceptance:** the wake share recorded per gap; a change (batched or early
  unpark, submitter-side prewake, spin policy) lands only if it cuts the
  after-idle time with idle CPU inside a stated budget, else the finding is
  recorded as falsified.
- **Class:** [patch] [perf]; status: todo; priority: P2; last-update: 2026-09-17.

## MOI-QUEUE-PLANE-SHRINK-2026-09-02 [patch] [perf] — todo

- **Outcome:** A local plane that has drained releases storage it grew past the
  configured initial capacity, so a one-off burst stops setting the retained
  slot count for the life of the process.
- **Finding:** `next_job` drains the injector to exhaustion and planes only
  grow, so the largest burst a worker ever drains fixes its retained slots —
  measured 256 slots for a 200-job burst and 2,048 for 2,000, from a 16-slot
  start. At Apollo's 24 workers and the 128-byte `ScheduledJob` of ADR-036 a
  256-slot plane retains 32,768 bytes per worker against the 16,384 the
  128-slot default provisions.
- **Dependency:** ADR-038 rejected bounding the drain pass: emptying the
  injector is what lets a high-priority job preempt work already queued ahead
  of it, since the injector is one cross-priority FIFO. Shrinking on empty
  leaves that ordering untouched.
- **Compounding cost:** growth retires the old buffer into
  `ChaseLevInner::retired_arrays`, and the executor's planes take the default
  `DeferredReclaim`, documented as retaining retired arrays until the final
  owner or stealer endpoint drops. Every intermediate buffer of a doubling run
  is therefore held alongside the live one, so a 16-to-2,048 growth retains
  2,032 dead slots on top of 2,048 live -- close to twice the peak.
- **Existing instrument:** `moirai-executor/tests/indexed_allocation_contract.rs`
  already probes this with the pointer-identity ledger. `probe_first_growth`
  asserts `growth.direct.retained()` equals the sum over every doubling step of
  `expected_growth_capacities`, with `block_count(...) == 1` per step ("each
  doubling step must retain one direct queue buffer"), and asserts the buffers
  are released only on drop ("dropping the queue must release every direct
  buffer"). The retained-byte model and its oracle therefore already exist;
  this item changes when the release happens, and that test is where the
  change is measured.
- **Sized 2026-09-03; the policy switch is rejected.** `SharedEpochReclaim` was
  the obvious candidate, and reading it settles the question against it.
  `SharedEpochState::enter` is an `AcqRel` `fetch_add` on one shared
  `AtomicUsize`, paired with a `fetch_sub` when the guard drops, and
  `enter_steal_access` takes it in both `ChaseLevStealer::steal` and
  `steal_batch`. Every thief working one victim would contend that single
  counter, so the switch buys reclamation with an atomic read-modify-write on a
  shared line in the steal path — the trade the synchronization ladder exists to
  refuse, and the one `DeferredReclaim` documents itself as avoiding.
- **Benefit, derived from the model `probe_first_growth` already asserts.**
  Retained dead bytes are the sum over every doubling step below the final
  capacity. A worker starting at the 128-slot default and growing to 2,048
  retires 128 + 256 + 512 + 1,024 = 1,920 slots; at the 128-byte
  `ScheduledJob` of ADR-036 that is 245,760 bytes of dead arrays per worker
  beside 262,144 live. Across 24 workers, about 5.9 MB dead. Derived, not
  measured: the per-step model is test-backed, the burst that reaches 2,048 is
  the characterization test's, not a profile of a real workload.
- **Direction instead: reclaim at quiescence.** The epoch counter exists to
  prove no steal is in flight. The scheduler already establishes that at its
  own quiescence boundary — `WorkerQueues` participates in it — so a
  reclamation that runs only there needs no per-operation counter and leaves
  `DeferredReclaim` on the hot path untouched. What must be established before
  this is sized further: that the quiescence condition genuinely precludes an
  in-flight `steal_batch` on a retired array, which is a Loom obligation, not a
  reading.
- **Acceptance oracle:** `retained_local_plane_storage_tracks_burst_size`
  inverts — retained slots return to the configured capacity once the plane
  drains — with priority, steal, saturation, and wake-progress coverage and the
  executor Loom models unchanged.

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
- [x] Remove the obsolete `paste` advisory exception. The current locked WGPU
      graph contains no `paste` advisory after the provider source refresh;
      `RUSTSEC-2023-0071` remains the only active RSA residual.

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
