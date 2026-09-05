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

The adapter exposes no WGPU or vendor types, no direct host/device byte-cast
API, and no silent CPU fallback. A missing or failed device is a typed error.
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
completion and error propagation, and compile the generic task against the
provider seam. WGPU and CUDA provider suites remain the provider-owned device
evidence; this crate does not claim hardware execution without a device.

The public GPU API changes and requires a major migration. In-repository
callers migrate in the same delivery; no compatibility wrapper is retained.

## Migration

The migration is delivered as Moirai 0.6.0. Consumers replace concrete GPU
context and task types with `GpuContext<D>` and `GpuTask` implementations,
where `D` is supplied by Hephaestus. The old direct WGPU surface is removed;
callers do not retain an adapter or forwarding wrapper.
