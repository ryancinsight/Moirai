# 0089 — Report native pointer capture loss

Status: Accepted

## Context

Windows can take mouse capture away from a window without delivering the
button release. A consumer that sees only button and motion messages can
continue a drag after the operating system has ended it.
[WM_CAPTURECHANGED](https://learn.microsoft.com/en-us/windows/win32/inputdev/wm-capturechanged)

## Decision

Emit `WindowEvent::PointerCaptureLost` when Windows revokes held capture.
Keep a deliberate release on the final button's up event represented by that
button event. On `WM_CANCELMODE`, clear the stored capture state before calling
the default window procedure. Its capture-release notification can reenter
the callback without emitting a second cancellation.
[WM_CANCELMODE](https://learn.microsoft.com/en-us/windows/win32/winmsg/wm-cancelmode)

## Alternatives

Treating capture loss as keyboard focus loss changes unrelated state. A
synthetic button release can commit a gesture whose release never arrived.

## Migration

This is a [major] Rust API change because `WindowEvent` is exhaustive.
Consumers add a `PointerCaptureLost` match arm
that abandons the held pointer gesture. RITK translates it into its
presentation event stream. The Metis Python adapter emits
`pointer_capture_lost`.

The Windows regression cases exercise capture transfer to a peer window and
cancel mode. Each checks that capture changes and that the event is queued once.

[Provider PR](https://github.com/ryancinsight/Moirai/pull/704)
