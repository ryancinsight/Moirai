# ADR 0064: Bounded browser text clipboard

Status: Accepted

Date: 2026-09-19

Driver: [MOI-WASM-CLIPBOARD-2026-09-19](../backlog.md#MOI-WASM-CLIPBOARD-2026-09-19),
[Metis integration](../../metis/backlog.md#METIS-INTEGRATION-001)

## Context

Metis already lets browser text controls use the browser's native clipboard
and history operations, but a consumer-owned command or toolbar cannot observe
or write clipboard text through the Moirai provider boundary. Calling
`navigator.clipboard` directly in a consumer would import `web-sys`, duplicate
secure-context checks and bypass the provider's text-size policy. A clipboard
API also crosses a browser permission and user-activation boundary, so a
successful Rust call cannot imply that the browser will permit the operation.

## Decision

`moirai-pal::wasm` exposes `WebDocument::clipboard`, which resolves the
document window's `navigator.clipboard` property and returns a `WebClipboard`
handle. The provider exposes asynchronous `read_text` and `write_text`
operations. Both operations use the existing 1 MiB UTF-8 text bound before
retaining or submitting application data. A missing property, including an
insecure context, returns `Unsupported`; a rejected browser promise returns an
`Other` I/O error with no browser payload copied into the diagnostic. The
provider retains only the browser API handle and never exposes a native
clipboard handle or filesystem path.

Metis owns user-facing clipboard commands, status text, and any content policy.
The provider does not read clipboard data on mount, grant permissions, or
invent a fallback clipboard implementation. Consumers must invoke writes from
an interaction that satisfies the browser's transient user-activation policy
and surface a rejected operation to the user.

## Alternatives rejected

1. Calling `web-sys` from Metis would duplicate the PAL boundary and make
   browser capability checks drift between consumers.
2. Reading clipboard text during mount would violate user-activation and
   permission expectations and would expose user content without an action.
3. Retaining an unbounded string or returning raw `JsValue` would bypass the
   existing text validation contract and leak browser representation details.
4. Falling back to a hidden textarea or an in-memory string would report a
   successful operation without changing the operating-system clipboard.

## Threat model and limits

The page, browser, permission store and operating-system clipboard remain
outside the Rust trust boundary. A browser may deny either operation, return a
non-text value, revoke permission, or require a user gesture. The provider
surfaces those outcomes and bounds the text it accepts; it does not prove page
identity, clipboard confidentiality, cross-engine parity, or native-host
accessibility. Native clipboard providers and consumer content redaction are
separate work.

## Verification

The existing text-validation tests continue to prove the 1 MiB UTF-8 bound.
The WASM target check compiles `WebDocument::clipboard`, `WebClipboard` and
both Promise conversions against the pinned `web-sys` API; native PAL tests
remain independent of browser bindings. Metis must add a user-activated
consumer journey and record browser permission results before claiming
clipboard acceptance.
