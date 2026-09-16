# ADR 0060: Browser event trust provenance

Status: Accepted

Date: 2026-09-15

Driver: [MOI-WASM-DOM-TRUST-2026-09-15](../backlog.md#MOI-WASM-DOM-TRUST-2026-09-15),
[Metis browser](../../metis/backlog.md#METIS-BROWSER-001),
[RITK Metis migration](../../ritk/backlog.md#RITK-SNAP-METIS-001)

## Context

Moirai owns the browser event boundary, while Metis and RITK consume its
format-neutral input snapshots. The browser exposes whether an event was
trusted through `Event.isTrusted`, but the current metadata snapshots discard
that bit. A consumer cannot therefore distinguish a user-mediated event from a
script-created event without importing `web-sys` and retaining a second DOM
binding.

## Decision

Capture `Event.isTrusted` once in `WebEvent` and copy the value into
`PointerMetadata`, `WheelMetadata` and `KeyboardMetadata`. Each metadata type
exposes an `is_trusted` accessor. Moirai reports the browser snapshot and does
not decide whether an application should accept it; Metis carries the value
through its canvas event contract, and an owning application such as RITK
applies its input policy at the consumer boundary.

The value is a snapshot, not a retained browser event or callback. It is
therefore independent of listener lifetime and remains format-neutral. The
native PAL has no corresponding browser provenance bit and is unchanged.

## Alternatives rejected

1. Importing `web-sys` in Metis or RITK duplicates the browser boundary and
   makes the ownership and lifetime of DOM events ambiguous.
2. Treating every browser event as trusted removes a security signal and lets
   synthetic script events reach applications that require user-mediated
   input.
3. Having Moirai reject untrusted events would put application policy in the
   platform layer and prevent non-interactive consumers from observing the
   browser snapshot for their own policy.

## Threat model and limits

Script-created DOM events are the direct threat: a page or injected script can
dispatch an event whose `isTrusted` value is false. Consumers that require
user-mediated input must reject false values before changing state. A true
value is only the browser's provenance signal; it does not prove a physical
human, a secure browser session, an operating-system permission decision or
the authenticity of WebDriver/automation infrastructure. Cross-engine and
native accessibility evidence remains a consumer-level verification task.

## Verification

The metadata accessors are compiled against the locked `web-sys` event binding
by the WASM check and warning-denied Clippy; native PAL tests and Rustdoc cover
the unchanged non-WASM surface. Metis and RITK add value-semantic tests for
preserving and enforcing the snapshot. The existing hosted browser trace keeps
its independent `event.isTrusted` observation; agreement between that trace and
the provider value is evidence of the browser path, not proof of physical
input.
