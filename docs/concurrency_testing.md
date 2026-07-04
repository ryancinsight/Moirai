# Concurrency verification runbook

The full concurrency-safety surface of the scheduler is verified by three test
tiers. This file is the single place that enumerates them and the exact commands
to run them — the loom models in particular are invisible to `cargo test` /
`cargo nextest` (they are `#![cfg(loom)]`-gated and run only under
`RUSTFLAGS="--cfg loom"`), so without this list they rot unnoticed.

The findings each tier protects are logged in [`concurrency_audit.md`](concurrency_audit.md).

## Tier 1 — loom exhaustive-interleaving models (machine-checked)

Each model re-states a lock-free protocol over loom atoms and lets loom enumerate
every thread interleaving. They are `#![cfg(loom)]`-gated: empty in a normal
build (no `loom` dependency pulled in), so they never touch the standard suite.

Convention: run **one model per invocation** with `--test <name>` (a bare
`--cfg loom` build would also compile the non-loom test files against loom's std
shims). Always `--release` (loom state-space search is slow in debug).

| Model | Crate | Verifies |
|-------|-------|----------|
| `loom_wake_handshake` | moirai-executor | park/wake Dekker handshake (`pending_tasks` ↔ idle bitset) never loses a wakeup — all four accesses SeqCst (audit R19) |
| `loom_join_quiescence` | moirai-executor | `join()` quiescence Dekker handshake (`active_workers` ↔ `join_waiters`) never loses a wakeup — SeqCst; guards the R23 fix |
| `loom_lifo_slot` | moirai-executor | per-worker `LifoSlot` hands a job to exactly one taker (pop/steal/replace-push) — no double-`ptr::read` (audit R22) |
| `loom_chase_lev` | moirai-scheduler | Chase–Lev deque push/pop/steal protocol |

Run the whole loom suite:

```sh
# moirai-executor models
for t in loom_wake_handshake loom_join_quiescence loom_lifo_slot; do
  RUSTFLAGS="--cfg loom" cargo test -p moirai-executor --test "$t" --release || exit 1
done
# moirai-scheduler models
RUSTFLAGS="--cfg loom" cargo test -p moirai-scheduler --test loom_chase_lev --release
```

When a lock-free protocol's memory orderings change, update the mirroring model
in the same change (the models pin the production orderings by construction) and
re-run it. A model that no longer matches production is a stale test, not a pass.

## Tier 2 — adversarial / stress tests (empirical, in the standard suite)

Run under the committed nextest timeout (`cargo nextest run -p <crate>`); a 30 s
slow mark or 60 s hang is a defect, never a budget bump.

| Test (moirai-executor `schedule::runtime::tests`) | Guards |
|------|--------|
| `scheduler_scope_nested_saturation_completes` | nested `scope` inside a scoped job completes (no deadlock) — W ∈ {1,2,4} (audit R20) |
| `scheduler_scope_recursive_fork_join_is_sound` | drive-shaped log₂-depth recursive fork-join, analytical sum oracle — the corruption path (audit R20) |
| `scheduler_scope_nested_panic_propagates_and_pool_survives` | a panic in a nested scoped job unwinds only its own job; sibling runs; outer scope completes (audit R21) |
| `scheduler_scope_nested_leaves_scheduler_quiescent` | help-while-waiting re-entrant `execute_job` leaks no `pending`/`active` count (`join()` reaches quiescence) |
| `deque_concurrency` (moirai-scheduler) | Chase–Lev deque under real threads |

## Tier 3 — standard local gate

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo nextest run --workspace          # committed 30 s slow / 60 s kill timeout
cargo test --doc                       # doctests (nextest does not run these)
```

CI note: no `.github/workflows` runs Tier 1 today. Until one exists, run the
loom suite locally whenever a `#[cfg(loom)]`-modelled protocol changes (the
memory orderings in `schedule/runtime/{types,worker,scheduler/core}.rs` and
`moirai-scheduler/src/deque/chase_lev.rs`).
