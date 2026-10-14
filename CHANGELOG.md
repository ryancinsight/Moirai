# Changelog

All notable changes to the Moirai concurrency library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.2] - Unreleased

### Fixed

- [patch] Check the actual WebView2 object's callable methods instead of
  requiring this realm's DOM `EventTarget` prototype identity. Cache the real
  receiver and methods; retain exact listener ownership through successful
  removal, including compensation after a registration exception. Missing,
  noncallable and throwing operations preserve member context. No supported
  public signature or message limit changes, and no JavaScript bridge is added.

Fourteen actual Chrome contracts pass, including seven new real-object checks.
Strict WASM and Windows diagnostics and the original Rust/WASM producer pass.
The new installed-runtime fixture requires real bidirectional Unicode/identity
values, inclusive payload bounds, close/remount, and actual profile cleanup.
Its execution and native consumer acceptance remain required; ordinary Chrome
and compilation do not establish installed bridge delivery.

## [0.9.1] - Unreleased

### Fixed

- [patch] An explicitly selected private verification diagnostic observes native
  WebView2 resource MIME/status, lifecycle/process errors, CDP exceptions/logs
  and bounded selected DOM status without evaluating JavaScript. Registration
  and enable completions share the original navigation deadline; callback
  teardown and incomplete requests remain observable. This adds no supported
  public API and leaves ordinary application observation disabled.

The protected provider run passes 31 Windows policy/frame cases and six
installed-runtime fixtures, including selected native DOM and owned shutdown.
The subsequent consumer trace identifies the nominal bridge cast rejection;
this instrumentation diagnoses that failure without repairing it.

## [0.9.0] - Unreleased

### Added

- [minor] `wasm::WebViewBridge` and its owned message listener post/receive actual
  WebView2 JSON under the existing shared native/browser message limits.
  Explicit listener close and counted failed-cleanup guards preserve callback
  lifetime when the browser refuses teardown.
- [minor] `wasm::WebPermissions` supplies real geolocation, camera, microphone
  and notification requests with typed native outcomes, finite deadlines,
  one unsettled slot per kind/document and owned media leases. Snapshots retain
  cancelled settlement guards and zero-or-two geolocation dispatchers; late
  media acquisition stops every track when publication has been cancelled.
- [minor] DOM listener explicit close and actual active/retained counts expose
  failed teardown ownership. `wasm::WebNumber::decimal` uses the native
  ECMAScript formatter; `WebViewBridge::operation_wait_limit` exposes the shared
  30-second provider bound.
- [minor] `ProcessSpec::working_directory` owns an absolute child directory
  validated before process resources are created, with native OS failures
  retained and no process-global directory mutation.
- [minor] `WebViewConfig::with_user_data_folder` admits a fresh caller-owned
  local profile, checks the runtime's effective directory and waits for the
  actual normal browser-process exit during close. Directory cleanup remains
  caller-owned (ADR 0077).

### Fixed

- [patch] Native WebView2 permission callbacks mark DENY decisions as nonsaved, so a
  fresh profile can produce independent repeated-request denial evidence.
- [patch] Closed window and WebView polling drain the actual final destruction
  event without dispatching through a destroyed HWND. The native callback domain
  moves to its own leaf; the window owner remains below 500 lines.

Real Windows page/permission/profile-exit acceptance and consented positive
grant/late-acquisition cleanup remain pending; static compilation does not
establish those runtime outcomes.

## [0.8.3] - Unreleased

### Fixed

- WebView2 capture admission uses the exact decoded Base64 length, including
  terminal padding, before allocation. Valid captures of exactly 64 MiB remain
  admitted; excess decoded bytes, invalid alphabet, padding and terminal bits
  remain errors.
- The `webview2` production feature enables its optional PNG decoder instead
  of relying on a development dependency to validate captured frame bytes.

## [0.8.2] - Unreleased

### Added

- `moirai-webview` 0.2.0 validates host-owned exact HTTPS destinations from
  `MoiraiExternalLinks` bundle metadata and hands admitted popup links to the
  operating system browser. AppKit and UIKit share the same bounded Rust
  policy; remote content never replaces the packaged frame (ADR 0074).

### Fixed

- Interpret bodyless Fetch statuses before payload admission. A 304's
  representation Content-Length does not consume received-body budget;
  absent body streams also return exact empty bytes. Actual payload bounds,
  deadline, redirect and cancellation contracts remain unchanged (ADR 0072).

## [0.8.1] - Unreleased

### Fixed

- Bind the remote task reply listener before dispatching work. A fast reply
  cannot race listener creation; an occupied reply endpoint dispatches no work
  (ADR 0071). Message and connection budgets remain unchanged.
- Decode actual stream read dictionaries by validating the object, completion
  flag and byte chunk. Web IDL dictionaries have no class constructor;
  instanceof rejects every real Fetch result and prevents the Rust HTTP page
  from completing its readiness probe.

## [0.8.0] - Unreleased

### Added

- Bounded browser `WebHttp` Fetch, GET/POST methods and complete response bytes;
  request/body budgets, redirect suppression, omitted credentials and one deadline.
  Dropping a request aborts Fetch and cancels/releases its owned stream reader.
- Owned document listeners and read-only `WebLocation`, element ancestor,
  attribute and text observations let Metis move browser policy into Rust.

### Changed

- File slices and HTTP bodies share one statically dispatched stream reader;
  filled-slice and complete-response termination retain their distinct contracts.
- Browser DOM acquisition, events and page observations live in leaf modules
  below 500 lines; public names remain unchanged.

### Fixed

- Borrowing scheduler scopes share the scheduler's wait primitive, eliminating
  Darwin's two lazy native allocations per indexed fan-out. Pending predicates
  and completion lifetimes remain independent; existing zero-allocation and
  computed-value contracts retain their workload and thresholds (ADR 0070).
- Shared-memory verification uses unique PID/sequence names within Darwin's
  31-byte POSIX limit. Real exact-limit reads and excess-name errors are checked
  without changing queue capacities, payloads or concurrency assertions.
- Python bindings use PyO3 0.29's build-mode selection: Cargo tests link the
  interpreter, while maturin sets `PYO3_BUILD_EXTENSION_MODULE` and supplies
  extension linker arguments. An unconditional legacy feature suppresses the
  interpreter link even for the workspace test executable on macOS.
- Browser-local reactor ownership uses `Rc`; native reactor ownership retains
  `Arc`. WASM/native warning-denying PAL checks pass.
- WebGPU device construction follows the state declaration order.

## [Unreleased]

### Added

- Independent `moirai-webview` 0.1.0 crate: one packaged resource policy and
  WKWebView implementation with AppKit/UIKit lifecycle adapters (ADR 0068).
  Resource tests inspect real bytes, MIME, traversal, symlink escape, real
  sockets/FIFOs and inclusive limits. Metadata and descriptor checks reject
  special files; Unix nonblocking/no-follow open flags preserve that boundary.
- Apple CI compiles macOS, device and simulator ABIs, runs real resource and
  contained-process tests within 60 seconds, and checks the browser PAL.
- Resource reads and native navigation use the same origin policy, including
  non-Apple library builds with warning-denying diagnostics.

## [moirai-webview 0.1.2] - Unreleased

### Fixed

- Install Rust-owned AppKit application, Edit and Window menus. Command-Q
  now invokes native termination while the WebView holds focus; Close,
  Minimize, Zoom and full-screen actions use the native responder chain.
  Standard window style flags and page authority remain unchanged (ADR 0073).
- Emit structured native window lifecycle events with actual geometry, state
  and standard-button properties, excluding page content.

## [moirai-webview 0.1.1] - Unreleased

### Fixed

- Packaged Apple resources return `NSHTTPURLResponse` status 200 with exact
  `Content-Type` and `Content-Length` headers. A plain `NSURLResponse` exposes
  Fetch status zero, which wasm-bindgen 0.2.129 rejects before application
  initialization. AppKit and UIKit share the corrected response constructor.
- Native Foundation regressions check status, MIME, URL and exact lengths for
  HTML, JavaScript, WASM, empty files and the inclusive resource bound;
  unrepresentable signed lengths fail before response delivery.

## [0.7.2] - Unreleased

### Fixed

- Unix single-instance reception reads complete messages after peer shutdown.
  A monotonic message deadline and nonblocking reads avoid Darwin's rejection
  of socket timeout changes on a closed peer; truncation and stalls remain errors.
- Native receive-cancellation verification drives actual readiness when a
  replacement future is pending, then checks the exact payload and retirement
  of both registrations instead of assuming synchronous datagram delivery.
- macOS contained-process cleanup checks live group membership after Darwin's
  zombie-only `killpg` EPERM. Exited status remains stable; denied live members
  and incomplete kernel queries remain errors. Real leader/descendant fixtures
  exercise the cleanup law.
- Targets without native descriptors reject a cancellation owner instead of
  discarding ownership in waker registration; WASM builds emit no unused-owner
  diagnostic.

## [0.7.0] - 2026-10-04

### Changed

- Release the workspace as 0.7.0 on the mnemosyne-memory 0.9.0 requirement.
  Mnemosyne published 0.9.0 and its git default branch moved with it, so any
  fresh resolution of `moirai-core`'s `mnemosyne-memory ^0.8` requirement
  against the git source fails — breaking `cargo update` in the release gate
  and in every downstream placeholder build (leto's Crates.io Release). The
  requirements move to the 0.9 line; `mnemosyne-memory-core` stays on 0.3.0.

### Breaking

- The optional Mnemosyne integration now requires mnemosyne-memory 0.9.0;
  consumers must update their dependency graph to the 0.9 provider line.

## Historical workspace releases

- [[0.6.1] - 2026-10-02](docs/changelog/0.6.1/index.md)
- [[0.5.0] - 2026-08-11](docs/changelog/0.5.0/index.md)
- [[0.6.0] - 2026-09-07](docs/changelog/0.6.0/index.md)
- [[0.4.0] - 2026-07-17](docs/changelog/0.4.0/index.md)
- [[0.3.1] - 2026-07-15](docs/changelog/0.3.1/index.md)
- [[0.3.0] - 2026-07-15](docs/changelog/0.3.0/index.md)
- [[0.2.0] - 2026-05-24](docs/changelog/0.2.0/index.md)
- [[0.1.0] - 2024-09-04](docs/changelog/0.1.0/index.md)
