# ADR 0007: WebAssembly Browser Event-Loop Integration

Status: Accepted

**Date**: 2026-05-25
**Revision**: 2026-09-07

Revision note: the browser host seam now includes owned DOM elements, form
control values, checked and disabled control state, modal dialog lifecycle,
focus, pointer capture, pointer metadata, wheel metadata, bounded file-drop
metadata and event listeners. Metis consumes
these handles without importing `web-sys`; the listener guard removes the
callback before releasing its JavaScript closure.

## Context

Moirai's WebAssembly target runs on a browser JavaScript event-loop thread.
Browser Web API handles are thread-affine, callback closures must have an
owned lifetime, and incoming data must not create unbounded Rust memory. The
native reactor contract therefore cannot be copied to the browser by adding a
`Send + Sync` bound or by starting a blocking driver thread.

## Decision

The browser reactor remains cooperative. Browser callbacks enqueue bounded
readiness events and wake the exact Rust future waiting for a WebSocket
message. The native `Reactor` implementation keeps `Send + Sync`; the WASM
definition is thread-affine and has the same registration, polling and wake
roles without promising cross-thread transfer.

The delivered WebSocket slice has one owning `WebSocketConnection` per
descriptor. It retains the open, message, close and error callbacks, detaches
them in `Drop`, and closes the browser socket. Messages are decoded from text,
`ArrayBuffer`, or typed-array values only after the configured byte limit is
checked. Each connection has a finite message queue and the reactor has a
finite readiness queue. Oversize messages, queue exhaustion, poisoned state,
and browser errors become terminal typed I/O errors; no data is silently
dropped. One `WebSocketReceive` future may be pending at a time, its waker is
updated when an executor supplies a replacement, and dropping it unregisters
the waiter. Closing or failing a connection wakes a pending receive and
removes its queued readiness events.

The same ownership rule now covers `FileReader`: callback closures are held by
an RAII guard, handlers are cleared and an in-flight read is aborted on drop,
and no callback uses `Closure::forget`.

Browser deadlines use the same ownership boundary. `WebTimer` retains the
`setTimeout` callback and its JavaScript window handle, clears the handle on
drop, and clamps durations to the browser's signed 32-bit millisecond range.
Metis can therefore race a bounded receive against a bounded deadline without
leaving a timer callback or a WebSocket waiter after cancellation.

Moirai also owns the narrow DOM boundary used by Atlas WASM applications.
`WebDocument` and `WebElement` wrap the current document, trusted markup,
text/attribute updates, input/select values, checked checkbox/radio state,
disabled button/input/select state, modal dialog lifecycle, focus control and
child insertion. Pointer events expose a `PointerMetadata` snapshot containing
the browser `pointerId`, normalized device type, viewport coordinates, button
state, modifier keys and primary-pointer marker. Elements own the
`setPointerCapture`, `hasPointerCapture` and `releasePointerCapture` calls with
typed invalid-input errors when the browser rejects a request.
Wheel events expose a `WheelMetadata` snapshot with three deltas, their
pixel/line/page unit, viewport coordinates and modifier keys. The event seam
returns no metadata for unrelated event kinds, so application policy can keep
scroll and gesture handling explicit.
Drag events expose a bounded `DropMetadata` snapshot with CSS-pixel coordinates
and validated `DroppedFile` records. File counts, names, media types and byte
sizes are checked before allocation; file bytes and filesystem paths remain
outside the DOM seam.
`WebEventListener` owns one callback registration and removes it in `Drop`;
`WebEvent` exposes only the target/value/pointer/drop-metadata/default-action
operations needed by an application, so browser bindings do not leak into
Metis domain code. `spawn_local` routes application futures to the browser
event loop without creating a second executor.

## Rejected alternatives

The prior callback implementation used `Closure::forget`, discarded message
payloads, and returned an empty readiness poll. That shape could appear to
work while leaking JavaScript roots and losing application input. A blocking
native driver or an unbounded event/message queue would violate browser
thread-affinity or the memory bound, so neither is a compatibility path.

## Verification

The state machine has native value tests for zero limits, message ordering and
close, oversize and queue-overflow terminal errors, waiter cancellation,
single-waiter rejection, producer wakeup, and executor-waker replacement.
`moirai-pal` compiles for `wasm32-unknown-unknown` and passes warning-denied
Clippy for both WASM and the native library against merged Mnemosyne backend
`2eb49c1`. The historical bounded WebSocket slice recorded 39/39 native PAL
tests; the current configured Nextest run passes 47/47, including checked and
disabled DOM-state coverage. The DOM surface is compile-checked on WASM and
Metis's browser traces exercise checked controls, disabled lifecycle, modal
dialog open/close and focus restoration. Metis `43dd7c7` exercises pointer ID
`1`, provider-backed capture verification and release on the pointer surface.
Metis `f15a6fa` consumes the metadata snapshot and its input-sensitive browser
trace records mouse coordinates, changed and held buttons, Shift modifier
state and primary-pointer state at the same viewport.
Metis `ad00f9d` consumes the `WheelMetadata` snapshot from provider revision
`f634b3a802ec0355da22f111ed01067d2435c5cb`; its WASM build and warning-denied
Clippy pass, and the in-app browser trace records input-sensitive vertical and
horizontal pixel deltas with viewport coordinates. The CUA scroll action is
automation-generated and does not expose the browser `isTrusted` flag, so the
trace does not claim physical-wheel or cross-engine parity.
The provider's file-drop validation helpers have native value tests for bounded
names and media types plus finite, representable byte sizes; the DOM binding is
compile-checked with the `DragEvent`, `DataTransfer` and `FileList` features.

## Residuals

The cooperative task executor, Web Worker scheduling, browser `fetch` and
general browser network facade, and a headless Chromium/Firefox/WebKit trace
remain open work. The native `net` module is intentionally not compiled for
WASM until those browser APIs have an owned bounded contract. This ADR does not
claim full Tokio or Tauri browser parity.
