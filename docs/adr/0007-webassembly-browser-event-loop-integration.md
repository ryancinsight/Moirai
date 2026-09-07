# ADR 0007: WebAssembly Browser Event-Loop Integration

Status: Accepted

**Date**: 2026-05-25
**Revision**: 2026-09-06

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
`2eb49c1`. The configured Nextest run passes 39/39 native PAL tests. A real
browser trace is still required before Metis adopts this provider contract.

## Residuals

The cooperative task executor, Web Worker scheduling, browser `fetch` and
general browser network facade, and a headless Chromium/Firefox/WebKit trace
remain open work. The native `net` module is intentionally not compiled for
WASM until those browser APIs have an owned bounded contract. This ADR does not
claim full Tokio or Tauri browser parity.
