# Release version contract

This compatibility document is embedded by the benchmark-contract test. It
contains no execution state; active work lives in [`../backlog.md`](../backlog.md)
and unowned risk in [`../gap_audit.md`](../gap_audit.md).

**Target Version**: moirai-async 0.12.0; moirai-quinn 0.1.0;
moirai-quic 0.2.0; facade 0.12.0; unchanged workspace 0.11.0.

- [x] [minor] Implement unbounded MPSC, persistent polling and native ReadBuffer;
  strict async patch API 223 pass/30 checker skips and 33 actual-source Miri tests pass.
- [x] [arch] Implement the licensed versioned Quinn adaptation; native package
  tests pass 191 cases, upstream driver tests pass all 25 cases with ignored
  stress cases executed, and default/minimal facade sessions pass (ADR 0084).
- [x] [minor] Independently resolve source-consumer default/minimal normal/build
  graphs without Tokio; no path/patch result is registry admission.
- [ ] [arch] Independent exact-source review, full required hosted/platform/
  security and original 108-pair performance gates before merge/publication.
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
