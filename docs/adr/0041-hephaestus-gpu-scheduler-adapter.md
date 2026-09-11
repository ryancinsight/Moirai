# ADR 0041: Hephaestus GPU scheduler adapter

- Status: Accepted
- Date: 2026-09-04
- Board item: [`MOI-GPU-HEPHAESTUS-ROUTE-2026-09-04`](../backlog.md#moi-gpu-hephaestus-route-2026-09-04)

## Context

`moirai-gpu` currently owns a second WGPU device, buffer, pipeline, and task
implementation. That duplicates the Atlas GPU provider and prevents Moirai
from depending on `hephaestus-wgpu`: the provider currently imports Moirai
runtime helpers. The duplicate layer also carries a fabricated device-memory
estimate, a permanently empty device enumeration result, direct `bytemuck`
layout bounds, and boxed futures on the task path.

The scheduler owns execution admission and work stealing. Hephaestus owns
device acquisition, typed device buffers, kernel dispatch, synchronization,
and provider-specific WGPU/CUDA behavior. Eunomia remains the layout contract.

## Decision

Make `moirai-gpu` a provider-neutral scheduling adapter. `GpuContext<D>` owns a
real `D: hephaestus_core::ComputeDevice`; provider constructors acquire
`hephaestus_wgpu::WgpuDevice` or another Hephaestus device at the integration
boundary. `GpuTask<D>` is generic and synchronous at the device seam, and the
Moirai runtime wraps it in its existing `Task` implementation. The returned
handle is therefore scheduled by the work-stealing executor, while a task's
device operation remains statically dispatched and typed.

The adapter exposes no direct WGPU or vendor device, buffer, pipeline, or
byte-casting API. The provider aliases are explicit acquisition conveniences;
task and transfer operations remain on the generic Hephaestus contract, and
there is no silent CPU fallback. A missing or failed device is a typed error.
The provider dependency direction is corrected first: Hephaestus WGPU may
retain the independent `moirai-sync` substrate, but it cannot import
`moirai-runtime` or `moirai-gpu`.

## Alternatives rejected

- Keep the direct WGPU layer: duplicates the provider and violates the Atlas
  ownership boundary.
- Put the scheduler into Hephaestus: reverses dependency direction and couples
  a device provider to one runtime.
- Preserve boxed futures: adds allocation and dynamic dispatch to a task path
  whose provider operation is synchronous and statically typed.
- Fall back to CPU when device acquisition fails: masks a provider fault.

## Failure modes and verification

Provider acquisition, allocation, dispatch, synchronization, and transfer
errors remain typed through the Hephaestus result. The adapter tests execute a
host reference `ComputeDevice` with a value-sensitive task, verify executor
completion, and `gpu_task_propagates_host_provider_error_through_runtime`
asserts the provider's `LengthMismatch` fields through the scheduler. The
generic task compiles against the provider seam. WGPU and CUDA provider suites
remain the provider-owned device evidence; this crate does not claim hardware
execution without a device.

The public GPU API changes and requires a major migration. In-repository
callers migrate in the same delivery; no compatibility wrapper is retained.

## Revision 2026-09-11

The decision is implemented against the current provider graph. `moirai-gpu`
now contains only the generic `ComputeDevice` context, acquisition preferences,
and synchronous typed task seam; its direct WGPU device, buffer, pipeline,
bytemuck and boxed-future modules were removed. `Moirai::spawn_gpu` submits the
typed operation to the existing work-stealing executor. The host provider tests
cover input-sensitive upload/download, feature rejection, runtime task
completion and typed `LengthMismatch` propagation. WGPU and CUDA compilation use
the same generic seam; hardware and device-specific kernel evidence remain
provider-owned.

The consumer lock pins Hephaestus `ff370517`, whose CUDA implementation owns
the dynamically loaded driver boundary. The activated graph therefore contains
no `cuda-oxide` or CUDA import-library link; a hosted runner without a CUDA
driver can compile the provider graph without attempting `-lcuda`.
