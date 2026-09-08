# ADR 0049: Win32 window provider

Status: Accepted

Date: 2026-09-08

Revision 2026-09-08: add a bounded message-queue wait so native consumers can
run an event-driven loop without busy polling or an unbounded blocking call;
retained events are drained before the operating-system wait. Add bounded
Windows IME composition phases and surface retrieval failures through the
provider result.

Driver: [MOI-WINDOW-WIN32-2026-09-08](../backlog.md#MOI-WINDOW-WIN32-2026-09-08),
[Metis desktop](../../metis/backlog.md#METIS-DESKTOP-001)

## Context

Moirai's PAL owns native readiness reactors and the browser DOM boundary, but it
has no native window or operating-system event producer. Atlas applications
currently receive `PlatformEvent` values only from their caller. Adding a GUI
toolkit would duplicate Moirai's scheduler and process ownership and would not
provide a common bounded frame and authority boundary for Metis.

## Decision

Add a Windows-only `moirai_pal::windows::window` provider using the existing
Windows system-binding dependency. `NativeWindow::new` validates a bounded
`WindowConfig`, registers one process-local window class, creates an HWND and
retains callback state in an owned allocation. The public handle is thread-bound
because Win32 window procedures and message queues are thread-affine.

`NativeWindow::poll_events` removes only messages addressed to its HWND and
returns a bounded batch. The window procedure translates `WM_CLOSE`, focus
changes, left-button and movement messages, `WM_KEYDOWN`, `WM_CHAR`, `WM_SIZE`
and `WM_DPICHANGED` into the provider's value events. `WM_NCDESTROY` reports
completed destruction. Surrogate pairs are combined before text delivery.
`WM_IME_STARTCOMPOSITION`, `WM_IME_COMPOSITION` and `WM_IME_ENDCOMPOSITION`
become `TextComposition` events carrying start, preedit, commit or cancel
phases. IME strings are read through the existing Windows system binding,
capped at `MAX_COMPOSITION_UNITS` and validated as UTF-16 before they enter the
event queue. An active composition is canceled when a composition message has
no string flag or when the composition ends, so an empty update cannot leave a
stale preedit value in a consumer.
`present_argb8888` validates
the dimensions and exact pixel count, reuses the retained vector when possible,
and invalidates the client area. `WM_PAINT` uses a top-down 32-bit DIB and
`StretchDIBits` to repaint the retained frame. `Drop` calls `DestroyWindow` only
for a live handle; callback state is cleared at `WM_NCDESTROY` and remains owned
by the Rust handle until destruction returns.

`NativeWindow::wait_events` is the event-driven companion to `poll_events`. It
first drains retained events (including constructor lifecycle state and queue
overflow) so readiness is not lost, then waits on the owning thread's queue for
a caller-supplied finite duration capped at 30 seconds. A timeout returns an
empty batch; the provider never sleeps or spins on behalf of the consumer.

The provider contains no process, filesystem, network, WebView or authorization
policy. Those capabilities remain in Moirai's existing lifecycle APIs or in the
consumer's host policy. The class and event limits are constants so a consumer
cannot turn message traffic or pixel dimensions into unbounded allocation.

## Alternatives

Winit, tao, egui, GPUI, Iced and Tauri were rejected because they introduce a
second runtime and do not extend the existing Moirai PAL contract. A global
window map was rejected because it adds shared mutable state and cross-window
identity races. Calling Win32 from each Atlas consumer was rejected because it
duplicates unsafe callback ownership and makes teardown inconsistent. Keeping
application-supplied events only would leave the native host gap open.

## Safety and failure behavior

Every Win32 call is isolated in this module and its `SAFETY` comment states the
pointer, lifetime and thread assumptions. The callback never unwinds across the
system ABI. Null callback state, invalid UTF-16 and invalid client dimensions
fall back to the default window procedure or a bounded diagnostic event; they
do not dereference an unproven pointer. Frame presentation retains no borrowed
pointer after the call returns. A failed creation or repaint returns an `io::Error`
with the native error code; no silent fallback renderer is selected.

This increment is Windows-only and supplies a software frame surface and native
IME event production. WebView2 COM hosting, accessibility providers, OS
permission enforcement and macOS/Linux implementations require separate
providers and runtime evidence. Application editing policy still owns how a
consumer displays or commits the composition phases.

## Verification

Unit tests cover value-level decoding, composition bounds and queue limits. A
Windows test creates a real window, pumps creation and destruction messages,
posts representative messages including IME start/end, verifies
lifecycle/resize/DPI/text values and presents a bounded frame; a second native
test verifies retained initial readiness, rejects an overlong wait, proves a
posted input wakes the finite wait and observes an empty queue afterward. The
tests do not claim a particular installed IME or CJK keyboard journey.
The PAL package must pass warning-denied Clippy and native tests on Windows;
cross-target library checks verify that non-Windows and WASM builds do not
compile the provider. Miri cannot execute Win32 calls, so the FFI path is
covered by the Windows host test and the documented pointer/lifetime boundary.
