# Release version contract

This compatibility document is embedded by the benchmark-contract test. It
contains no execution state; active work lives in [`../backlog.md`](../backlog.md)
and unowned risk in [`../gap_audit.md`](../gap_audit.md).

**Target Version**: 0.11.0 (workspace); the native QUIC release cohort is
moirai-async 0.12.0, moirai-quinn 0.1.0, moirai-quic 0.2.0, and facade 0.12.0.

- [x] [minor] Implement unbounded MPSC, persistent polling and native ReadBuffer;
  corrected-source strict async patch API 223 pass/30 checker skips and 34
  actual-source Miri tests pass after the payload-panic wake repair.
- [x] [arch] Implement the licensed versioned Quinn adaptation; the prior
  exact head passes 192 native package tests and default/minimal facade
  sessions. The conformance repair passes 36 local driver tests, including all
  three stress cases, the 128-event fairness regression, checked conversion
  behavior and two read-buffer rejection regressions. The cast-gate repair
  still needs committed exact-head qualification (ADR 0084).
- [x] [minor] Independently resolve source-consumer default/minimal normal/build
  graphs without Tokio; no path/patch result is registry admission.
- [x] [arch] Owner accepted ADR 0084's versioned public migration design;
  no license/security, performance or release gate was waived.
- [ ] [arch] Same-reviewer approval of the changed final exact head, full
  hosted/platform/security and original 108-pair performance gates before
  merge/publication. The archived-provenance controller suite passes 26 cases;
  the discontinued async-std dependency is removed, with first-spawn executor
  configuration under regression coverage. The local Atlas conformance ratchet
  now passes on the preceding committed candidate; the subsequent Atlas cast
  gate identified 37 imported-driver casts and its checked-conversion repair
  requires a new exact-head run. The optional FIPS license gate remains failed
  under the unchanged policy.
- [ ] [minor] Ordered archive/publication and clean registry-consumer admission;
  notification prerequisite delivery and every historical hold remain explicit.

- [ ] [patch] moirai-async 0.11.1: notification future creation/broadcast and
  independent-permit correctness; exact-source validation and review required.
  This package increment does not qualify QUIC promotion or native Tokio removal.

- [x] [minor] Integrate current main's 0.9.2 WebView fixes; classify pending native QUIC additions and the documented pre-1.0 UDP migration under workspace 0.10.0.
- [patch] The private allocator edge accepts Mnemosyne-memory 0.10; public
  mnemosyne-memory-core 0.3 budget types and all Rust implementations remain unchanged.

- [x] [patch] Fourteen real Chrome contracts, strict WASM/Windows compilation, and 1,310 native workspace tests.
- [x] [patch] Checked native method admission, registration compensation and exact callback lifetime.
- [x] [patch] Compiled Rust/WASM installed-WebView2 roundtrip, payload bounds, close/remount and profile cleanup ([actual Windows run](https://github.com/ryancinsight/Moirai/actions/runs/38539086879/job/115769121797)).

## Apple local increment

- [x] [patch] Darwin terminal leader cleanup; real descendant and status oracles.
- [x] [patch] Descriptor-free cancellation-owner rejection; WASM diagnostics.
- [x] [minor] moirai-webview 0.1.0 resource and native lifecycle implementation.
- [x] [patch] Real-file byte/MIME, traversal, symlink, socket/FIFO and inclusive-bound tests.
- [x] [patch] Apple ABI, resource, process and browser-PAL CI coverage.
- [x] [patch] Closed-peer instance messages, exact payload bounds and live-stall deadlines.
- [x] [patch] Exact capture-byte admission and production Windows PNG wiring; full computed boundary, production compilation and warning-denying checks pass.
- [ ] [patch] Continuous installed WebView2 preview acceptance on the new Windows CI job.
- [ ] [minor] Native input, lifecycle and device evidence from the active backlog.
- [ ] [minor] Provider publication and consumer standalone locks before release.
