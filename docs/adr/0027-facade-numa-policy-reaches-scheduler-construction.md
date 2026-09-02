# ADR 0027: Facade NUMA policy reaches scheduler construction

Status: Accepted

- Date: 2026-08-14
- Change class: [arch] [minor]
- Refs: `MOI-NUMA-002`, Atlas `ATLAS-MOIRAI-NUMA-095`

### Context

The `numa` feature exposed `MoiraiBuilder::numa_aware(bool)`, and
`ExecutorConfig` already stored the corresponding policy. The facade method
discarded its argument, while `HybridExecutor` always performed topology
detection and built per-worker NUMA assignments. The public control therefore
did not affect the scheduler it described.

### Decision

Keep one scheduler construction implementation with an explicit NUMA policy
parameter. Direct `ThreadScheduler` construction retains topology-aware
behavior. The `HybridExecutor` passes the
configured value when its `numa` feature is active and passes `true` when the
feature is absent, preserving the existing non-NUMA build behavior. The
facade's `numa_aware` method mutates the existing `ExecutorConfig` field, and
the facade `numa` feature enables the matching core and executor features.

NUMA awareness means only per-worker victim-selection locality. Moirai does
not claim topology-directed memory placement; allocation remains owned by the
provider boundary responsible for memory policy.

Construction retains the locality tier only when the worker assignment
contains at least two distinct NUMA nodes. A single represented node cannot
change victim order, so retaining it would execute the same peer scan twice on
a miss. The full-ring pass remains the sole path for absent, partial-one-node,
and single-node topology.

Revision 2026-09-01: clarified the multi-node precondition and normalized
single-node assignments to absence after a source audit found that the original
construction populated every single-node worker with node zero despite the
documented skip behavior. Enforced worker-to-processor binding remains a
separate provider-first correction; topology-derived slots are advisory until
that correction lands.

### Alternatives rejected

1. Keep the no-op setter and document the limitation. Rejected because it
   leaves a public policy that cannot affect runtime behavior.
2. Add a second scheduler constructor or duplicate worker construction.
   Rejected because it forks the scheduler lifecycle and creates policy drift.
3. Implement memory placement in Moirai. Rejected because that crosses the
   scheduler and allocator ownership boundary without a current contract.

### Verification

The facade regression builds enabled and disabled runtimes and asserts the
corresponding `ExecutorConfig` values. The executor runtime regression builds
the scheduler with NUMA disabled and asserts every worker assignment is
unset. A construction-level regression normalizes absent, one-node, and
partially known layouts to absence while preserving an exact two-node layout.
A synchronized three-worker regression proves that a same-node miss reaches a
cross-node victim without depending on platform NUMA detection.
The hosted Rust workspace gate verifies the all-feature construction,
configured Nextest, doctests, Clippy, and rustdoc at the exact merged head.
