### Breaking

- `ExecutorConfig` gains the public field `worker_placement`, so struct
  literals without `..ExecutorConfig::default()` no longer compile, and
  `ExecutorError` gains the variant `WorkerPlacementFailed { worker, processor,
  cause: PlacementFailure }`, so exhaustive matches need an arm. Migration: add
  the field (`WorkerPlacement::Unbound` keeps current behaviour) and handle or
  forward the variant.

- The workspace requires `melinoe` 0.10.0 and `themis-topology` 0.11.0, and both
  appear in public signatures. `moirai_gpu::resident_blocks` and
  `plan_persistent_launch` take `themis::GpuTopology`, and the
  `par_partition_*` functions of `moirai_parallel::melinoe_ext` (also
  re-exported at the `moirai` root under the `melinoe` feature and as
  `moirai::melinoe_ext`) take `melinoe::MelinoeCell` and
  `melinoe::region::WriterShard`; each is now the type from the new provider
  release, so a caller on `themis-topology` 0.10 or `melinoe` 0.9 passes a type
  these functions do not accept. `moirai::initialize()` and
  `moirai_executor::initialize()` register Moirai's executor with `melinoe`
  0.10 only: a crate still calling `melinoe` 0.9's `sync::partition_*`
  directly compiles unchanged and runs on that release's scoped threads
  instead of Moirai's pool, with the same results. Migration: move those
  dependencies to `themis-topology` 0.11 and `melinoe` 0.10.

[Version 0.6.1 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
