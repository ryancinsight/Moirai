# ADR 0061: Explicit browser WebGPU canvas surface

Status: Proposed

Date: 2026-09-16

Driver: [MOI-WASM-GPU-CANVAS-2026-09-16](../backlog.md#MOI-WASM-GPU-CANVAS-2026-09-16),
[RITK-SNAP-METIS-001](../../ritk/backlog.md#RITK-SNAP-METIS-001)

## Context

The browser PAL currently owns a bounded, borrowed RGBA frame contract and a
2-D `CanvasRenderingContext2d` presenter. RITK's browser viewer uses that
presenter, so every frame is rasterized and uploaded through the 2-D API even
when the browser has a WebGPU adapter. RITK's GPU volume renderer is native-only
and cannot be reused in a browser context because its device lifetime belongs to
the native host.

The provider must add a browser GPU presentation seam without importing DICOM,
viewer state, or a consumer's shader policy. WebGPU is optional browser
capability, so selection must be explicit and a missing adapter must be a typed
error rather than an invisible CPU fallback.

## Decision

Add `moirai_pal::wasm::WebGpuCanvas`. Its async constructor resolves a named
`HTMLCanvasElement`, checks `navigator.gpu`, requests an adapter and device, and
configures a `webgpu` canvas context with the browser's preferred format. The
surface retains only the context, queue, device and validated canvas extent.

`present` accepts the existing borrowed `RgbaFrame`, resizes and reconfigures
only when the validated extent changes, then calls
`GPUQueue.copyExternalImageToTexture` with one `ImageData` source and the
current swap-chain texture. The queue operation is the only GPU upload; the PAL
retains no frame bytes after the call. Context/device/copy failures become
`io::Error` values with their browser operation named.

The existing `WebCanvas` and `CanvasSurface` 2-D path remain unchanged. A
consumer opts into the GPU surface through an explicit constructor and owns the
policy for whether a browser without WebGPU is unsupported, a user-visible
configuration error, or a separately selected 2-D mode. No fallback is hidden
inside the PAL.

The binding uses `js_sys::Reflect` for the descriptor dictionaries and method
calls because `web-sys`'s WebGPU bindings are unstable-gated; the browser
objects remain JavaScript-owned and no unsafe Rust is introduced. The helper
validates every required property and operation at the boundary.

## Alternatives rejected

1. Making `WebCanvas` silently choose 2-D when WebGPU initialization fails
   would hide a capability decision and make performance/presentation evidence
   ambiguous.
2. Adding `wgpu` to `moirai-pal` would duplicate the Atlas Hephaestus provider
   and pull a native device dependency into the browser PAL.
3. Putting WebGPU setup in RITK would duplicate browser binding ownership and
   couple the DICOM consumer to a platform API.
4. Copying a frame into a retained Rust GPU buffer would violate the borrowed
   frame contract and add a second allocation to every animation frame.

## Threat model and limits

The browser page and GPU driver are outside the Rust trust boundary. A page can
remove or replace a canvas, revoke a device, or return a context error; each
operation is checked and surfaced. The provider does not claim device
isolation, driver correctness, color-management equivalence, or physical GPU
execution from a successful JavaScript call. Browser-engine visual evidence
and consumer-level pixel equivalence remain separate RITK gates.

## Verification

The PAL native suite covers the unchanged shared extent/frame validation. The
WASM target compiles the WebGPU bindings and the strict Clippy gate checks the
new module. Browser integration will exercise a real WebGPU adapter, frame
dimensions, non-black RGBA output and device-loss diagnostics in the RITK
consumer; a browser without WebGPU is an explicit unsupported result.
