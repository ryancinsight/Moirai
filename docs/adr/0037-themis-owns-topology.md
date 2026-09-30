# ADR 0037: Themis owns topology; the scheduler publishes only enforced worker placement

Status: Accepted

- Date: 2026-09-01
- Change class: [arch] [major]
- Refs: `MOI-THEMIS-TOPOLOGY-DUPLICATION-2026-09-01`,
  `MOI-WORKER-CORE-PREMISE-2026-09-01`, ADR-027, atlas ADR 0002, themis ADR 0006
- Revised 2026-09-29: worker binding now exists (`WorkerPlacement::Pinned`); the
  node table is populated only for pinned workers.

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
`core_id = worker_id % logical_cores`. Workers were never bound to
processors, so "worker `i` runs on core `i`" was fiction, and the table it produced was a
placement claim the runtime did not enforce. The `numa_aware` flag, its
builder methods, and the `numa` cargo feature existed only to switch that
derivation on -- a feature that, when enabled, fabricated an answer.

## Decision

Delete the mirror. Themis is the one authority for node distance and cache
levels; a scheduler that wants them asks `themis::CpuTopology` directly, as
`moirai-core`, `moirai-executor`, and `moirai-parallel` already do for worker
counts.

Delete the fabricated derivation. A worker reports a node only when the
runtime enforced its placement: `ExecutorConfig::worker_placement` is
`WorkerPlacement::Unbound` by default, and an unbound worker reports `None`.

`WorkerPlacement::Pinned` makes the premise true. Construction plans
processors from `themis::CpuTopology::detect()` in ascending id order, worker
`i` taking `i % n`; a pool larger than the machine shares processors, and an
undetectable or empty topology is `InvalidConfiguration`, never a silent
fallback. Each worker registers its thread, binds itself with
`themis::bind_current_thread` (fail closed, themis ADR 0006), and publishes
the outcome; a worker whose bind failed returns without taking work.
Construction waits for every outcome and, on any failure, shuts down and joins
the started workers and returns `ExecutorError::WorkerPlacementFailed` for the
lowest-numbered failing worker. The scheduler never escapes with a refused
binding. `worker_numa_nodes[i]` is then the node of worker `i`'s processor,
and still cleared when fewer than two nodes are represented. The same-node
steal tier is value-tested and reads this table.

Moirai owns the failure vocabulary (`PlacementFailure`); `themis::BindError`
is translated in one function and is not part of Moirai's contract. Because
`BindError` is `#[non_exhaustive]`, a variant this version does not know maps
to `PlacementFailure::Unsupported` -- no binding took effect.

Delete `ExecutorConfig::numa_aware`, both `numa_aware(bool)` builder
methods, and the `numa` cargo feature on `moirai`, `moirai-core`,
`moirai-executor`, and `moirai-scheduler`. With the derivation gone the flag
controlled nothing, and a flag that promises NUMA-aware placement while doing
nothing is the same fabricated claim in configuration form.

## Rejected alternatives

**Pin by default.** Binding changes what the OS may do with a process that
shares its host, can fail on cpuset-restricted containers, and lowers
throughput when the pool competes with other pinned processes. The default
stays `Unbound` and claims nothing; pinning is a request that can fail.

**Best-effort pinning that degrades to unbound.** A worker that continued
after a refused bind would leave the published table describing placement
nobody enforces -- the fabrication this ADR removes -- so refusal is an error.

## Consequences

Breaking, under the Unreleased line:

- `ExecutorConfig` has the field `worker_placement`, and `ExecutorError` has the
  variant `WorkerPlacementFailed { worker, processor, cause }`, with the
  `#[non_exhaustive]` cause `PlacementFailure`.

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

Overturning evidence: a host where binding refuses routinely in supported
deployments would argue for a partial-pin policy that publishes only the
workers that bound, with the table keeping `None` for the rest.
