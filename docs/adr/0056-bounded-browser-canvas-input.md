# ADR 0056: Bounded browser canvas input

Status: Accepted

Date: 2026-09-11

Driver: [MOI-WASM-CANVAS-INPUT-2026-09-11](../backlog.md#MOI-WASM-CANVAS-INPUT-2026-09-11),
[RITK Metis migration](../../ritk/backlog.md#RITK-SNAP-METIS-001)

## Context

Moirai owns the browser DOM boundary and already exposes pointer and wheel
metadata. The metadata currently contains viewport coordinates only. A canvas
consumer must route an event to the exact target surface without importing
`web-sys` or repeating browser layout reads in every application. RITK owns
DICOM parsing, geometry, viewer state and presentation policy; this decision
must remain format-neutral.

## Decision

Extend `PointerMetadata` and `WheelMetadata` with the browser's
target-relative CSS-pixel `offset_x` and `offset_y` values. Preserve viewport
coordinates for consumers that need document-level routing. The values are
snapshots captured from the originating DOM event and do not retain a browser
element or callback. Applications remain responsible for bounded event queues,
gesture policy, coordinate conversion and viewer semantics.

## Alternatives rejected

1. Re-reading `getBoundingClientRect` in each consumer duplicates the browser
   binding and introduces a layout-dependent race between event delivery and
   application handling.
2. Exposing raw `web-sys` events leaks JavaScript ownership and makes callback
   lifetime a consumer concern.
3. Adding canvas, DICOM or viewer types to Moirai would move application
   policy into the platform layer and prevent reuse by non-medical apps.

## Threat model and limits

Coordinates are untrusted browser input. The PAL reports the browser snapshot;
consumers must reject non-finite derived values, out-of-surface coordinates and
oversized event batches at their own trust boundary. `offset_*` is relative to
the event target as defined by the browser and can differ from client
coordinates under nested elements, CSS transforms or device-pixel scaling.
This seam does not claim cross-browser rendering or `isTrusted` parity.

## Verification

The native crate gate remains applicable to the non-WASM surface; the WASM
check and warning-denied Clippy compile the new accessors against `web-sys`.
Metis and RITK consume the accessors through their bounded, format-neutral
canvas event contracts and add browser evidence for target-local routing.
