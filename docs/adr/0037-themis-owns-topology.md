# ADR-037: Themis owns topology; the scheduler reports no worker placement

Status: Accepted

- Date: 2026-09-01
- Change class: [arch] [major]
- Refs: `MOI-THEMIS-TOPOLOGY-DUPLICATION-2026-09-01`,
  `MOI-WORKER-CORE-PREMISE-2026-09-01`, ADR-027, atlas ADR 0002

## Context

`moirai-scheduler::numa::{CpuTopology, NumaNode, CacheLevel}` mirrored
themis's types and answered `distance()` differently: themis selects
id-versus-compact-index by row length, the mirror always indexed
`distances[to_node]` by raw node id, so the two disagreed on a sparse-node
Linux host with real SLIT rows. It also folded `cache_levels().unwrap_or(&[])`,
turning themis's typed absence into "zero cache levels" -- the fabrication
themis's own docs warn against.

Nothing consumed the disagreeing half. `distance`, `adjacent_nodes`,
`cores_in_same_node`, and `cache_levels` had no call site. The only consumer
was scheduler construction, which used `logical_cores` and
`core_to_numa_node` to derive `worker_numa_nodes` from
`core_id = worker_id % logical_cores`. Workers are never bound to processors,
so "worker `i` runs on core `i`" was fiction, and the table it produced was a
placement claim the runtime did not enforce. The `numa_aware` flag, its
builder methods, and the `numa` cargo feature existed only to switch that
derivation on -- a feature that, when enabled, fabricated an answer.

## Decision

Delete the mirror. Themis is the one authority for node distance and cache
levels; a scheduler that wants them asks `themis::CpuTopology` directly, as
`moirai-core`, `moirai-executor`, and `moirai-parallel` already do for worker
counts.

Delete the fabricated derivation. Construction reports no per-worker node
(`None` for every worker). The same-node steal tier stays: it is
value-tested, and it activates for an assignment a caller can vouch for --
today only the injected assignments in the executor's own tests. A future
worker-binding feature makes the premise true and feeds the table honestly;
until then the runtime claims nothing it does not enforce.

Delete `ExecutorConfig::numa_aware`, both `numa_aware(bool)` builder
methods, and the `numa` cargo feature on `moirai`, `moirai-core`,
`moirai-executor`, and `moirai-scheduler`. With the derivation gone the flag
controlled nothing, and a flag that promises NUMA-aware placement while doing
nothing is the same fabricated claim in configuration form.

## Consequences

Breaking, under the Unreleased line:

- `moirai_scheduler::numa::{CpuTopology, NumaNode, CacheLevel}` are gone.
  Callers that need topology use `themis::CpuTopology`; its `NumaNode` carries
  the same processors and distances, and its cache levels are `Option` --
  absence stays absence.
- `ExecutorConfig::numa_aware`, `MoiraiBuilder::numa_aware`, and
  `ExecutorBuilder::numa_aware` are gone. Remove the call; default behaviour is
  unchanged, because the flag's only effect was a table that
  `normalize_worker_numa_nodes` cleared on every single-node host anyway.
- The `numa` cargo feature is gone from every crate and from `full`. Remove it
  from feature lists.

`moirai-scheduler` no longer depends on `themis`; the dependency edge stays
with the crates that actually detect topology. The source-text contract in
`benchmarks/tests/benchmark_contracts` now asserts the three struct names are
absent from the scheduler, alongside the schedulers it already guards against.

ADR-027 (facade NUMA policy reaches scheduler construction) described the
plumbing this removes; it stands as history and is superseded on the point of
the flag.
