# ADR 0041: Hephaestus consumes Moirai; Moirai carries no GPU provider adapter

- Status: Accepted
- Date: 2026-09-04
- Revised: 2026-09-28
- Board item: MOI-GPU-CYCLE-2026-09-28 (Moirai PR 501); first decision
  MOI-GPU-HEPHAESTUS-ROUTE-2026-09-04 (Moirai PR 325)

## Context

Atlas layers the stack foundation → infrastructure → domain. Moirai is
infrastructure; Hephaestus is the GPU provider and consumes Moirai
(`hephaestus-wgpu` and `hephaestus-cuda` depend on `moirai-sync`, and
`hephaestus-cuda`/`hephaestus-host` reach `moirai-runtime` through `leto-ops`).
Atlas ADR 0002 assigns GPU launch shaping — the occupancy planner over a themis
`GpuTopology` and a mnemosyne `KernelResourceBudget` — to Moirai, and device
backends to Hephaestus.

The first decision made `moirai-gpu` a scheduling adapter over
`hephaestus_core::ComputeDevice`: `GpuContext<D>`, `DevicePreferences`,
`GpuTask`/`FunctionGpuTask`/`ConfiguredGpuTask`, the `WgpuContext` and
`CudaContext` provider aliases, and `Moirai::spawn_gpu`. The default
`wgpu-backend` feature, the `cuda-backend` feature, and the `hephaestus-host`
dev-dependencies of `moirai-gpu` and the `moirai` facade closed a cycle: each
Hephaestus provider resolved a git copy of Moirai, so Moirai's own lock carried
twelve `git+https://github.com/ryancinsight/Moirai.git` packages beside the
workspace path packages. When Mnemosyne removed the generic parameter of
`periodic_defragmentation_sweep` (Mnemosyne `191d85e`), advancing Moirai's
Mnemosyne source rebuilt the stale git copy of `moirai-executor`, which failed
with E0107 on every workspace build (Moirai PR 499).

## Decision

Moirai depends on no Hephaestus crate. `moirai-gpu` is the occupancy planner
only: `plan_launch`, `plan_persistent_launch`, `resident_blocks`,
`LaunchShape`, and the `KernelResourceBudget` construction facade (ADR 0039).
Its dependencies are themis and mnemosyne-core.

The adapter layer is deleted, not moved. It had no consumer outside Moirai, and
every operation in it is already expressible without it: a Hephaestus device
operation is a closure over a shared device handle, which
`Moirai::spawn(TaskBuilder::new().build(..))` schedules on the work-stealing
executor with its typed `Result` intact. Provider-owned device errors,
acquisition, transfers, and synchronization stay in Hephaestus, with no CPU
fallback.

A workspace contract test asserts that `Cargo.lock` resolves no git copy of
Moirai, so a reintroduced consumer edge fails the gate at introduction.

## Alternatives rejected

- Keep the adapter over `hephaestus-core` only, dropping the provider and host
  crates: breaks the package cycle but keeps a repository-level edge from
  infrastructure to its consumer and needs a Moirai-owned `ComputeDevice`
  test double to cover it.
- Move the adapter into Hephaestus: `spawn_gpu` would make Hephaestus depend on
  `moirai-runtime` for surface nobody consumes, and couples a device provider
  to one runtime.
- Restore the removed generic parameter in Mnemosyne or pin Mnemosyne back:
  a compatibility shim that leaves the cycle in place.
- A committed `[patch]` unifying the git copies with the workspace paths:
  hides the cycle in the manifest and makes Moirai unconsumable as a git
  dependency.

## Consequences and verification

`moirai-gpu` loses `GpuContext`, `DevicePreferences`, the task types, the
provider aliases, the `wgpu-backend`/`cuda-backend` features, and the
re-exported Hephaestus types; the `moirai` facade loses `spawn_gpu` and the
`create_gpu_context*` methods. This breaks the public API of both crates and
lands in their next major release. The tests of the deleted surface are deleted
with it; the occupancy tests and the `gpu_acceleration` example remain. The
lock loses every Hephaestus, Leto, WGPU, and git Moirai package, and the
workspace's `hephaestus-*` and `eunomia` dependencies are removed.

Revision 2026-09-28: the scheduling adapter is withdrawn because its provider
edges formed the cycle described above; the planner decision stands.
