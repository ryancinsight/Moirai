# Changelog

All notable changes to the Moirai concurrency library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0] - Unreleased

### Added

- [patch] Add a separately built, checksum-verified read-only recovery observer
  for per-space ACK/CRYPTO, actual flight, flow credit and timer state (ADR 0082).
  Preserve canonical transport/locks and the original CUBIC promotion gate.
  Validate all 81 corrected local policy pairs and retain the rejected off-timer
  PTO-selector capture. The complete Linux state study retains 105/108 CUBIC/
  NewReno, 108/108 BBR and all 30 competing experiments; the unchanged pristine
  CUBIC job independently reproduces 105/108 at the same source tree.
- [patch] Retain a real maximum-workload handshake-loss experiment and isolated
  recovery candidate: 17/27 baseline versus 27/27 candidate complete exchanges,
  plus 23 native candidate contract tests. Correlate all six Linux early failures
  with queued Handshake CRYPTO and Data-only PTOs. Keep the patch experimental,
  the canonical dependency unchanged and the original promotion gate enforced.

- [patch] Archive and independently audit the complete October 14 recovery trace
  study and independent original CUBIC gate. Preserve 105/108 CUBIC/NewReno,
  108/108 experimental BBR and all competing outcomes; distinguish early
  server-handshake failures from established congestion/pacing/flow-credit
  pressure. Repeat the complete study on security-patched 140d60cf: six early
  server-handshake failures and 15 shared-queue pair deadlines remain. Retain
  unsuccessful/canceled inputs and keep promotion blocked.
- [patch] Add opt-in load-probe aggregation of pinned Quinn recovery/PTO events,
  handshake/admission/exchange phases and actual shared timer-driver wake
  lateness (ADR 0080). Drain server diagnostics concurrently with client work;
  retain the full original workloads, deadlines and CUBIC promotion gate.
- [patch] Forward only the public trace opt-in inside owned Linux namespaces;
  an authenticated two-peer preflight rejects missing child observations before
  full measurement. Retain the unsuccessful environment-filtered study.
- [minor] Bounded stock congestion-policy selection retains CUBIC as default;
  NewReno and explicitly experimental BBR permit matched measurements through
  the same authenticated transport. ADR 0079 preserves the original full-matrix
  promotion gate and adds shared-queue evidence collection without changing
  deadlines, flow-control windows, socket buffers or security contracts.
- [patch] Archive unsuccessful stock-policy experiments and correct measured
  observer/namespace-cleanup races. Verify kernel process absence before
  accepting a raced kill; retain real cleanup errors. Log originating probe
  failures before endpoint drain without emitting successful partial reports.
- [patch] Complete a fresh original-domain comparison after collector correction:
  CUBIC and NewReno each verify 105/108 pairs; experimental BBR verifies 108/108.
  Archive all matched fairness/latency/queue evidence and unsuccessful earlier
  experiments. BBR's mixed-policy tradeoff does not justify a default change;
  the original CUBIC promotion gate and MOI-QUIC-RECOVERY-001 remain open.

- [minor] PAL-owned applied UDP buffer/broadcast/multicast configuration and
  effective kernel queue readback, shared by async UDP and QUIC. QUIC queue
  requests leave TLS, stream windows and finite deadlines intact (ADR 0081).

- [patch] Preserve the complete 108-pair controlled CI matrix and independent
  verification: 105 exact-byte successes, three maximum-frame/16-stream loss
  deadlines and zero kernel UDP errors. Keep the measured recovery limit and
  failed performance gate explicit; diagnostic/full artifacts have distinct names.
- [minor] Passive connection metrics expose wire, recovery, congestion-window,
  path-RTT and MTU observations without exposing Quinn types. Controlled probes
  retain verified progress, namespace UDP/socket counters and every failure;
  diagnostics can select one declared case while normal CI retains all 108 pairs.
- [patch] Native load probe and Linux/Windows transport gates validate mutual TLS,
  complete bytes and acknowledged FIN; isolated kernel profiles retain exact
  RTT/goodput samples, loss counters and source/runner provenance.

- `WebSocketPeerClosed` carries a validated close code/reason as an I/O cause;
  native session dispatch identifies clean closure without comparing diagnostics.
- `moirai-quic` 0.1.0: TLS 1.3 native QUIC with explicit trust roots, optional
  mutual TLS, validated ALPN, reliable multiplexed streams, bounded connection,
  stream/handshake/frame admission and finite operation deadlines (ADR 0075).
- Native facade `quic` module in default and minimal builds; no QUIC feature
  flag exists. `Moirai` implements the fallible core
  `TaskSpawner` trait so protocol drivers share the application's scheduler.
- PAL shared cancellable UDP readiness, native socket adoption and batched I/O
  pollers; the timer's maximum admitted duration is authoritative for protocols.

### Changed

- [minor] Reconcile current main 0.9.2 and classify the pending QUIC additions
  and pre-1.0 UDP migration under workspace 0.10.0; moirai-quic remains 0.1.0.
  Preserve main's accepted ADR 0078 and number the applied UDP decision 0081.
- Quinn owns recovery/congestion and delegates task, timer and socket driving to
  Moirai. Its Tokio synchronization primitives introduce no Tokio runtime;
  QUIC's upstream rustls/Retry bridge links ring native code statically.
- Writable readiness remains cached until actual backpressure, avoiding one
  register/cancel cycle per packet. Shared notification retirement occurs under
  the registration lock so late subscribers cannot be stranded.

### Breaking

- [major] The pre-1.0 UDP configuration moves from the async crate to PAL.
  Replace `moirai_async::net::types::UdpConfig` imports with
  `moirai_async::net::UdpConfig`; use `buffers.receive`/`buffers.send` with
  `NonZeroUsize` and an explicit multicast group/interface instead of the
  former ignored `buffer_size` and boolean. Native clients update the same
  change; no compatibility facade retains the old type owner (ADR 0081).

- Replace `UdpConfig.buffer_size` with independent `buffers.receive` and
  `buffers.send` requests using `NonZeroUsize`. Replace the ignored multicast
  boolean with an explicit `MulticastConfig` group/interface variant. Import
  the PAL-owned configuration from `moirai_async::net`, rather than its former
  `types` module. Absent buffer requests now explicitly retain kernel defaults.

- Unix/Windows facade builds require ring's C/assembly build tools, including
  `--no-default-features`. Remove `--features quic` from draft-increment commands;
  QUIC and TLS validation remain part of the native transport contract.

### Fixed

- [patch] Require Rustls 0.23.46 in the shared TLS dependency and standalone
  QUIC provider after RUSTSEC-2026-0346/0347/0348/0349. Preserve caller trust,
  TLS/ALPN, early-data policy and public vocabulary; refresh both locks and
  repeat native/security/full-matrix validation without advisory suppressions.
  All applicable non-performance CI jobs pass; the independent original CUBIC
  gate still fails the same three deadlines. Preserve preceding-stack evidence.
- [patch] Resolve the independent fuzz lock with the timer driver's regular
  tracing dependency edge; preserve every package identity and strict locked
  resolution. Diagnostic tests retain full span trees on failure.
- [patch] Compile shared WebView limits only for opted-in Windows WebView2,
  browser consumers or portable tests. Default/minimal Windows transport
  builds retain strict diagnostics without unused optional-provider state.
- [patch] The embedded benchmark source inventory includes both socket-test
  leaves after the TCP/UDP split; loopback, readiness, backpressure and
  cancellation assertions retain their full computed-value requirements.
- Partial framing aborts on cancellation, failure and deadline. FIN
  acknowledgement completes queued output before connection teardown.
- Spawner rejection retains its original error across Quinn's infallible runtime
  ABI; message/handshake allocation respects caller budgets.
- [minor] `moirai-pal`: bounded Windows text clipboard provider.

## [0.9.3] - Unreleased

### Changed

- [patch] Accept Mnemosyne-memory 0.10 through the canonical Git source.
  Allocator calls remain private and their Rust implementation is unchanged;
  the public GPU budget remains on mnemosyne-memory-core 0.3. All workspace
  versions and internal requirements advance together. No scheduling, bridge,
  feature or test implementation changes. The API gate compares all 16 library
  surfaces against each exact locked base/head graph and participates in the
  verification verdict. Proc-macro-only targets remain compiled by the full gate.
- [patch] Require patched rustls 0.23.46 and refresh the workspace/fuzz locks
  for [RUSTSEC-2026-0346](https://rustsec.org/advisories/RUSTSEC-2026-0346.html),
  [0347](https://rustsec.org/advisories/RUSTSEC-2026-0347.html),
  [0348](https://rustsec.org/advisories/RUSTSEC-2026-0348.html) and
  [0349](https://rustsec.org/advisories/RUSTSEC-2026-0349.html). No advisory
  exception is added. Other registry records retain the base graph.

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
The [installed Windows fixture](https://github.com/ryancinsight/Moirai/actions/runs/38539086879/job/115769121797)
passes real bidirectional Unicode/identity values, inclusive payload bounds,
close/remount, and actual profile cleanup in 8.155 seconds. Its document owns a
packaged SVG icon; missing resources and browser errors still reject acceptance.
Metis adoption and native consumer journeys remain separate requirements.

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
