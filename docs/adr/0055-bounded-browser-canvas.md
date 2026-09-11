# ADR 0055: Bounded browser canvas

Status: Proposed

Date: 2026-09-11

Driver: [MOI-WASM-CANVAS-2026-09-11](../backlog.md#MOI-WASM-CANVAS-2026-09-11),
[Metis browser](../../metis/backlog.md#METIS-BROWSER-001),
[RITK Metis migration](../../ritk/backlog.md#RITK-SNAP-METIS-001)

## Context

RITK already owns DICOM parsing, pixel interpretation, geometry and viewer
state. Its native Metis path presents a format-neutral `PresentationFrame`, but
its browser path still starts an egui canvas beside Metis's DOM host. Moving
browser canvas calls into Metis would duplicate the browser binding boundary;
putting DICOM or viewer state into Moirai would violate the domain ownership
boundary.

Moirai already owns the WASM document and element handles. The missing seam is
an owned canvas surface that can accept a borrowed RGBA frame, validate its
resource bounds before JavaScript calls, and upload it to an HTML5 canvas.

## Decision

Add `WebCanvas`, `CanvasSize` and `RgbaFrame` to `moirai-pal::wasm`. A document
lookup resolves an element by identifier and rejects missing or non-canvas
elements with typed I/O errors. `CanvasSize::new` checks non-zero dimensions,
the maximum pixel count and multiplication overflow. `RgbaFrame::new` checks
that the borrowed byte slice is exactly four bytes per pixel and stays below
the provider byte bound.

`WebCanvas::present` resizes the canvas to the frame dimensions and submits one
`ImageData` object through the browser's 2-D context. The Rust API borrows the
frame and owns no pixel or callback storage after the call; the browser copy at
the Web API boundary is documented as an unavoidable platform transfer. The
provider contains the only `web-sys` canvas bindings. Metis consumes this seam
for generic presentation, and RITK supplies frames after its DICOM pipeline has
finished; no DICOM type or medical policy crosses the boundary.

## Alternatives rejected

1. Keeping eframe/egui as the browser presentation path leaves Metis without a
   browser rendering contract and retains a second event and canvas runtime.
2. Importing `web-sys` in Metis or RITK duplicates callback and browser binding
   ownership that already belongs to Moirai PAL.
3. Passing an owned `Vec<u8>` or retaining the last frame in the provider makes
   browser memory grow with application policy and defeats the borrowed frame
   contract.
4. Adding DICOM or viewer objects to the PAL would move domain semantics into
   infrastructure and make the provider unusable by non-medical applications.

## Threat model and limits

Frame dimensions and bytes are application inputs at a browser boundary. The
provider rejects zero, overflowing, mismatched and oversized frames before
allocating `ImageData`; the fixed limit bounds one upload but does not replace
Metis or RITK application memory policy. Browser canvas security, origin policy,
color management and GPU acceleration remain browser behavior. A successful
upload proves only that the provider submitted the validated frame; visual
correctness is established by a browser capture in the consumer integration.

## Verification

Pure validation tests cover zero dimensions, pixel-count overflow, exact RGBA
length, oversized frames and boundary acceptance. The PAL package must pass
warning-denied native Clippy and the `wasm32-unknown-unknown` check. Metis and
RITK add consumer tests and an engine-labelled visual capture after adopting
the published provider revision.
