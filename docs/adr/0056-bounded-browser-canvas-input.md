# ADR 0056: Bounded browser canvas input

Status: Accepted

Date: 2026-09-11

Revised: 2026-11-19 — retain an immutable affine content-box measurement so
captured gestures can preserve their press coordinate basis.

Driver: MOI-WASM-CANVAS-INPUT-2026-09-11 (delivered; see repository history),
[Moirai PR #715](https://github.com/ryancinsight/Moirai/pull/715)

## Context

Moirai owns the browser Document Object Model (DOM) boundary and exposes
pointer and wheel metadata. A canvas consumer must route an event to the exact
target surface without importing `web-sys` or deriving browser geometry in
each application. Live picking needs current layout, while a captured gesture
must retain its press-time coordinate basis if layout changes before release.
RITK owns DICOM parsing, geometry, viewer state and presentation policy; this
decision remains format-neutral.

## Decision

Keep viewport and target-relative CSS-pixel coordinates in event metadata.
Expose content-box measurement as an immutable, owned `ContentBoxGeometry`
snapshot containing the validated two-dimensional affine map and measured
dimensions, without retaining a DOM object. A live point lookup takes a fresh
snapshot; a captured gesture reuses its press snapshot and projects later
client points through that same map.

The platform abstraction layer owns CSS geometry, including nested rotation
and skew. Métis owns snapshot capture lifetime and gesture mechanics. RITK
owns medical measurement and viewer state. Applications remain responsible
for bounded event queues and their own domain semantics.

## Alternatives rejected

1. Re-reading layout during a captured gesture changes its coordinate basis
   after relocation or resize.
2. Exposing raw `web-sys` events leaks JavaScript ownership and makes callback
   lifetime a consumer concern.
3. Deriving a diagonal client-to-canvas scale downstream loses rotation and
   skew already represented by the browser affine map.
4. Adding canvas, DICOM or viewer types to Moirai would move application
   policy into the platform layer and prevent reuse by non-medical apps.

## Threat model and limits

Coordinates are untrusted browser input. Construction rejects non-finite,
singular or unrepresentable geometry, and projection rejects non-finite input
or arithmetic. Finite off-image points remain valid so the consumer can apply
its own capture policy. Three-dimensional transforms, motion paths, CSS zoom
and inaccessible closed-shadow ancestry remain unsupported.

Overturn this decision if a supported layout cannot be represented by the
captured affine model, or measured DOM behavior demonstrates that projection
requires richer browser evidence.

## Verification

Native geometry tests cover identity, scale, rotation, skew, frozen snapshots
and invalid arithmetic. The WebAssembly gate compiles the browser measurement
surface. Provider readiness also requires a Métis consumer candidate and a
real-browser resize and relocation fixture. Moirai PR #323 provides the event
metadata basis; PR #715 carries the snapshot addition.
