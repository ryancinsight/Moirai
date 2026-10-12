# ADR 0068: Apple WebView application host

- Status: Accepted
- Date: 2026-10-11
- Change class: [minor]

## Contract and ownership

Moirai owns native application lifecycle, WebKit handles and packaged resource
delivery. Metis owns manifests, compilation, bundling and application behavior.
One HTML/CSS/Rust-WASM bundle runs in browsers and installed Apple applications.
Native OS callbacks stay on the main thread. AppKit and UIKit supply lifecycle;
the resource and WebView configuration implementation is shared.

The new `moirai-webview` crate starts at 0.1.0 and adds no API changes to the
existing crates. A separate process-cleanup fix advances their patch version
to 0.7.1. It uses Apple's framework bindings and structured tracing;
it does not depend on Metis or domain crates.

## Invariants

For a canonical resource root R and requested relative path p, delivery requires
canonical(R/p).starts_with(R), a regular file and a bounded file length. URL
authority must equal `metis://app`. Invalid percent encodings, traversal,
symlink escapes and non-packaged navigation fail. MIME types distinguish WASM,
JavaScript, CSS and HTML. Resource delivery never contacts a network endpoint.
Metadata checks reject non-regular files before opening and validate the opened
descriptor. Unix nonblocking/no-follow flags refuse replacement of the leaf
with a symlink and avoid waiting on a replaced FIFO. The read stops after the
public byte bound plus one, which detects growth beyond the admitted limit.
Packaged parent directories are owner-controlled inputs; this is not an OS
filesystem sandbox contract.
The original page CSP remains in force. Native application state never grants
page code a filesystem or process capability.

Successful reads expose `NSHTTPURLResponse` status 200, the resource MIME in
`Content-Type`, and its exact byte length in `Content-Length`. The response
constructor rejects lengths outside Foundation's signed ABI before any task
callback. A plain `NSURLResponse` supplies bytes without a Fetch success status;
wasm-bindgen 0.2.129 rejects that response before binding application events.
Patch 0.1.1 corrects this transport contract in the shared WebView implementation
without changing the frontend, CSP or allowed resource origin. Apple's
[HTTPURLResponse](https://developer.apple.com/documentation/foundation/httpurlresponse)
owns the native status/header representation.

The event-loop host owns its WebView and delegates until application exit.
UIKit owns rotation, keyboard and suspend/resume presentation; no mobile child
process is spawned. A shared Rust-WASM application retains its own state within
that WebView. Cold relaunch persistence remains an application contract.

## Alternatives and verification

Opening Safari alone does not provide an installed host. Reimplementing DOM
behavior separately in Swift creates a second application. A loopback server
adds network lifecycle to a packaged application. WKURLSchemeHandler instead
serves the same assets through one constrained resource implementation.

Tests cover actual filesystem contents, MIME types, URL authority, escaped
traversal, symlink escape, real sockets/FIFOs and inclusive size bounds. At the 64 MiB limit,
byte(i) = i mod 256 defines the complete content oracle; a larger file is
rejected. New tests capture arrange/act/assert spans for failure diagnosis.
The Apple CI job compiles each ABI and runs the real resource/process tests
with a 60-second execution limit; browser PAL compilation checks the
descriptor-free cancellation contract. Native acceptance journeys must load
the real WASM, submit a name, inspect the exact Rust greeting, and capture the
visible application. iOS device compilation is distinct from device signing
and installation. Browser journeys inspect the same computed greeting.
Native Foundation tests assert status, parsed MIME, URL and lengths, including
empty resources and the resource bound. The Metis macOS journey reproduces an
empty result before the correction and exact Rust greetings for normal,
Unicode and empty names afterward.
