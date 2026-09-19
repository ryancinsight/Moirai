# ADR 0065: Per-monitor DPI awareness for native windows

Status: Accepted

Date: 2026-09-19

Driver: [MOI-WINDOW-DPI-2026-09-19](../backlog.md#MOI-WINDOW-DPI-2026-09-19),
[Metis desktop](../../metis/backlog.md#METIS-DESKTOP-001)

## Context

The Win32 window provider already translates `WM_DPICHANGED` into a bounded
`WindowEvent::DpiChanged`, but a window created under the process's ambient DPI
context is not guaranteed to receive truthful per-monitor transitions. Metis
uses the event to map physical geometry, text and hit testing. A synthetic
message test proves translation only; it cannot establish that a real move
between monitors produces the event.

## Decision

Each `NativeWindow` enters
`DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2` on its creating thread before
class registration and HWND creation. The previous thread context is retained
in a thread-affine guard for the complete `NativeWindow` lifetime and restored
when the window is dropped. Nested windows therefore restore contexts in LIFO
order, and creation failures restore the prior context through the guard's
destructor. Child hosts created while the window is live inherit the same
per-monitor-aware context.

The provider does not synthesize DPI events or infer a transition when the host
has one monitor or equal-scale displays. A physical-monitor capture must report
the actual `WM_DPICHANGED` value; an unavailable transition remains an explicit
evidence limit.

## Alternatives

Process-wide `SetProcessDpiAwarenessContext` was rejected because the PAL does
not own the caller's other windows or initialization order. Leaving the
ambient context unchanged was rejected because it makes the provider's DPI
event contract host-dependent. Injecting `WM_DPICHANGED` was rejected because
it would make visual evidence indistinguishable from a real monitor move.

## Safety and failure behavior

The guard calls only the Windows system binding and stores the returned opaque
context without dereferencing it. An invalid previous context returns the
native last-error value and prevents window creation. Drop restores a valid
previous context and ignores the unreportable restoration result; the window
has already been synchronously destroyed, and no Rust invariant depends on a
successful context restoration. The guard's thread affinity follows the
thread-owned `NativeWindow` contract.

## Verification

The Windows lifecycle test creates a real hidden HWND and asserts that its
window awareness context equals per-monitor-v2 before posting the existing DPI
message sequence. The provider's native tests, warning-denied Clippy, format,
documentation and cross-target checks remain required. The test does not claim
that the host has two physical monitors; Metis V05 supplies that host-level
capture when such a transition is available.
