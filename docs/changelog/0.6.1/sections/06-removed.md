### Removed

- **`CleanupConfig::cleanup_interval`.** Nothing ever read it: reclamation is
  proportional to task registration rather than to elapsed time, so there is no
  interval to configure. Migration: delete the field from struct literals.

- **`moirai_core::memory::UnifiedRingBuffer`** (ADR-0016). It was a third copy of
  the bounded ring in the crate, serialized per side by a mutex, and
  `UnifiedChannel` was its only consumer — so the type is deleted rather than
  kept beside the canonical cores. Migration: use
  `moirai_utils::queue::LockFreeQueue`, which is the core `UnifiedChannel` now
  holds, or reach the channel itself through `moirai_core::channel::unified`.

- **The scheduler topology mirror and the `numa_aware` flag** (ADR-037).
  `moirai_scheduler::numa::{CpuTopology, NumaNode, CacheLevel}` duplicated
  themis and answered node distance differently; the only consumer derived a
  per-worker NUMA table from `worker_id % logical_cores`, which workers, never
  bound to processors, do not satisfy. The types, the derivation,
  `ExecutorConfig::numa_aware`, both `numa_aware(bool)` builder methods, and
  the `numa` cargo feature are removed. Migration: read topology from
  `themis::CpuTopology`; drop the builder call and the feature. Default
  behaviour is unchanged, since the table was cleared on single-node hosts
  anyway. The same-node steal tier remains for injected assignments.

[Version 0.6.1 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
