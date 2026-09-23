# ADR 0063: Recover explicit browser WebGPU surfaces

Status: Accepted

Date: 2026-09-18

Revision: 2026-09-19 — each acquired device now owns a cancellable observer
for `GPUDevice.lost`; observed loss fails presentation until explicit recovery.

Driver: MOI-WASM-GPU-RECOVERY-2026-09-18 (delivered by [Moirai #400](https://github.com/ryancinsight/Moirai/pull/400))

## Context

`WebGpuCanvas` owns a browser `GPUDevice`, queue and configured canvas
context. The browser can revoke the device or invalidate the swap-chain after
initialization. The current surface reports the resulting operation error, but
has no explicit way to acquire a fresh device without reconstructing the
consumer's input listeners and canvas ownership. A consumer that retries
`present` on the stale handles would repeat the failure; a hidden two-dimensional
fallback would change the presentation contract and invalidate GPU evidence.

## Decision

Add an explicit asynchronous `WebGpuCanvas::recreate` operation. It resolves
`navigator.gpu`, requests a new adapter and device, obtains the queue and
preferred format, and replaces the retained device state only after all setup
steps succeed. The configured extent is cleared after a successful replacement
so the next borrowed frame performs one bounded canvas resize and configure.
When setup fails, the old state remains unchanged and the typed browser error
is returned. The operation never selects the two-dimensional presenter.

Each acquired device also owns a browser-local observer for its
`GPUDevice.lost` promise. Once that promise settles, `present` returns
`io::ErrorKind::Other` before configuring or uploading another frame. Recovery
installs a fresh loss state and observer as one device-state replacement, so a
late notification from the prior device cannot mark the replacement as lost.
Dropping the surface or replacing its device cancels the corresponding
observer through `LocalTaskHandle`; a failed acquisition never mutates or
cancels the active state.

`metis-web::CanvasSurface` exposes the same recovery operation for its explicit
GPU variant. It retains the existing DOM input listeners and consumer-owned
state; only the provider GPU handles are replaced. A consumer decides when to
invoke recovery after a surfaced `present` error and must not submit frames
until recovery succeeds.

## Alternatives rejected

1. Reconstructing the whole `CanvasSurface` would discard the input listener
   guards and force every consumer to rebuild its state machine.
2. Retrying `present` against the old device would preserve invalid handles and
   produce an unbounded failure loop.
3. Silently switching to `WebCanvas` would hide a capability transition and
   make GPU/non-GPU evidence incomparable.
4. Retaining a second device beside the active one would keep stale browser
   resources alive and add an unbounded recovery state.

## Threat model and limits

The page, browser and GPU driver are outside the Rust trust boundary. A page
may replace the canvas, deny adapter/device setup or revoke the device between
recovery and the next frame. Every setup operation remains checked and bounded;
the consumer must surface a later `present` error again. Device loss is
asynchronous: a `present` call made before the browser settles `GPUDevice.lost`
and runs the observer microtask can still reach the browser queue. Recovery does
not claim device isolation, driver correctness or physical GPU execution.

## Verification

The provider's native tests continue to cover the shared frame and extent
contract. Native loss-channel tests prove notification visibility and generation
isolation. The standalone `wasm32-unknown-unknown` check and strict library
Clippy compile the recovery path, and the Metis consumer gate compiles the
public adapter method. `CanvasInput` remains a separate retained field in the
consumer surface. Actual device-loss detection and recovered non-black pixels
require consumer evidence from a browser with an adapter that can be
deliberately revoked.
