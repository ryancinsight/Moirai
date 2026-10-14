# ADR 0078: Private native startup observation

Status: Accepted

Date: 2026-10-14

Change class: [patch]

Target: workspace 0.9.1. Acceptance of this source contract does not establish
actual Windows observation or repair the consumer startup failure.

## Decision

The native PAL owns bounded, passive WebView2 startup observations behind both
an explicit unstable fixture selection and a dedicated diagnostic tracing
filter. The consumer supplies its exact configured URI and selected DOM nodes;
PAL does not own application identifiers. This adds no supported public API.

Native resource MIME/status, lifecycle/process failures and actual CDP
Runtime/Log events accompany a single sequential selected DOM chain. The
[source-adjacent contract](../../moirai-pal/src/windows/webview/host/diagnostics/design.md)
owns byte/count quotas, completion ordering, point-observation limits, Weak
callback lifetimes and teardown. Registration and activation share the original
navigation deadline. Ordinary applications install no diagnostic observers.

## Alternatives and failure law

Desktop frames do not establish browser DOM contents. Runtime exception events
alone miss handled Rust failures that update status nodes and return normally.
Injected JavaScript, replacing Ready, changing page permissions, extending a
startup deadline and assuming missing callbacks mean success are rejected.

The private callback implementation admits native UTF16/UTF8 replies before
copying; the SDK convenience completion converts the entire reply first.
No bound is claimed for Chromium allocations before the callback. Oversized,
malformed, stale, missing, exhausted and incomplete observations remain distinct.
Every token removal is attempted before the existing controller close.

## Verification

Portable real-value tests cover closed JSON admission, exact encoded quotas,
request identity and cancellation. SDK compilation validates actual interfaces.
A real installed-runtime fixture must retain known DOM values, a missing node,
resource metadata and teardown under the unchanged wait. Consumer attribution
requires a subsequent actual Windows run; no product repair follows from the
previous blank desktop frame and successful navigation alone.
