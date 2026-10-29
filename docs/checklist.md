# Release version contract

This compatibility document is embedded by the benchmark-contract test. It
contains no execution state; active work lives in [`../backlog.md`](../backlog.md)
and unowned risk in [`../gap_audit.md`](../gap_audit.md).

**Target Version**: 0.11.4 (workspace); moirai-quic 0.1.0; moirai-webview 0.2.0

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
