# ADR 0048: Browser text input contract

Status: Proposed

Date: 2026-09-07

Driver: [MOI-WASM-DOM-TEXT-2026-09-07](../backlog.md#MOI-WASM-DOM-TEXT-2026-09-07),
[Metis text](../../metis/backlog.md#METIS-TEXT-001).

## Context

Metis needs DOM text editing before the native window path exists. The current
Moirai DOM seam exposes scalar input values and event lifetimes, but it does not
carry selection ranges or composition updates. Each consumer would otherwise
import `web-sys`, repeat browser exception handling and create a second callback
ownership model. Browser text is untrusted and its value, event data and locale
must be bounded before application state allocates from them.

## Decision

Moirai's WASM DOM seam will expose text-control values and selection snapshots
for `HTMLInputElement` and `HTMLTextAreaElement`. Selection offsets retain the
browser's UTF-16 code-unit convention and carry a closed direction enum. The
seam will expose `InputEvent` metadata (`inputType`, optional data,
`isComposing`) together with the target value and selection, and
`CompositionEvent` data and locale. Values and metadata are copied into owned
Rust strings only after provider bounds are checked; unsupported targets return
`None`, and browser exceptions return typed I/O errors. A setter restores a
validated selection through the browser element.

The provider owns browser bindings and callback teardown. Metis owns the
editing state machine, grapheme policy, undo/clipboard behavior, line layout,
and rendering. Native IME event production and assistive-technology behavior
remain host evidence, not claims of the WASM provider.

## Alternatives

Leaking `web-sys` types would duplicate the browser binding and lifetime
surface in every Atlas consumer. Treating browser UTF-16 offsets as Rust byte
indices would corrupt selections around combining marks and emoji. Returning
only the final value would lose preedit and cancellation transitions. A
third-party text wrapper would introduce another callback owner and runtime
dependency instead of extending Moirai's existing DOM substrate.

## Threat model and limits

The page and browser event stream are untrusted. Provider bounds cap value,
event-data, input-type and locale allocations; invalid metadata fails closed.
UTF-16 offsets are transport coordinates and are not grapheme boundaries. The
seam does not claim Unicode grapheme segmentation, bidi layout, fallback-font
metrics, native IME delivery, clipboard permissions or OS accessibility.

## Verification

Native tests cover text-bound validation, direction parsing and invalid UTF-16
range ordering. The WASM check and warning-denied Clippy verify the generated
`web-sys` feature surface and the owned DOM API. Metis adds policy tests and a
browser editing trace; synthetic browser events are marked untrusted when the
automation cannot provide an OS IME.
