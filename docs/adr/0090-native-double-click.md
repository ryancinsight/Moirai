# ADR 0090: Report native pointer double-clicks

Status: Accepted

## Context

GUI clients use double-click to finish shapes and activate items. The Win32
window queue retains down and up events but carries no timestamps. A reader
cannot infer click spacing or operating system timing from events drained in a
batch.

## Decision

Enable `CS_DBLCLKS` when registering the Win32 class. Treat the button double-click
message as the second button press for capture and deliver the corresponding
`PointerDown` followed by `WindowEvent::PointerDoubleClick`. Deliver the second
`PointerUp` through the regular release path. Windows owns double-click timing
and spatial tolerance. Applications decide what the marker does.

The marker is additive at the event stream level and changes the exhaustive
Rust enum. Existing consumers retain their down and up handlers. Add an arm for
the marker when matching every `WindowEvent`. A consumer that handles the
marker instead of its second click must consume the corresponding release
without performing the default click action again.

## Alternatives

A timer started when the application drains the queue sees processing time
rather than the physical click interval. Adding timestamps would still require
emulating the platform's click interpretation. Use the classification Windows
already supplies.

## Verification

The native-window lifecycle test posts the Windows down, up, double-click, up
sequence and checks coordinates and the five translated events in order. A
consumer regression checks that completing an annotation does not reuse the
trailing release as another click.

[Microsoft documents when the window receives `WM_LBUTTONDBLCLK`.](https://learn.microsoft.com/en-us/windows/win32/inputdev/wm-lbuttondblclk)
