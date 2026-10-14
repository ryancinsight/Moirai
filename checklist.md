# Moirai transport checklist

Active target: workspace 0.9.0; moirai-quic 0.1.0. Phase: Closure.

- [x] [minor] Specify native QUIC ownership, trust, ALPN, resource and cancellation contracts in ADR 0075.
- [x] [minor] Implement shared cancellable socket readiness and one scheduler/timer/UDP runtime boundary.
- [x] [minor] Implement authenticated independent streams, admission, complete framing, error retention and acknowledged FIN.
- [x] [minor] Require QUIC in every native facade build; default 187-test suite and minimal authenticated session pass, browser facade compilation excludes native QUIC, and CI checks both native modes.
- [x] [minor] Use validated typed WebSocket closure with code/reason; keep truncation distinct and avoid outgoing close allocations. Include the new provider in shared compatibility gates.
- [x] [patch] Verify real UDP loss/reordering, arbitrary bytes, malformed frames, exact limits, TLS rejection, stream isolation, backpressure and spawner failure.
- [x] [patch] Prove the readiness retirement regression fails against the preceding implementation and passes with the registration-lock transition.
- [x] [patch] Check native API compatibility against 0.8.2: 892 applicable checks pass under the stronger patch compatibility override.
- [x] [patch] Verify the canonical provider lock without path patches; warning-denying Clippy, formatting and all 187 HTTP/PAL/facade/QUIC tests pass.
- [x] [patch] Verify Metis against the published Git provider without path patches: 176 backend/IPC/core tests, strict diagnostics and native workspace/all-target checks pass.
- [x] [patch] Reconcile current main without losing capture-boundary fixes; review trust, ownership, framing, replay and cleanup contracts.
- [x] [patch] Specify and implement byte-verified, mutually authenticated process-pair measurements; all 27 local full-domain cases pass.
- [x] [patch] Verify native QUIC on Linux and Windows at published revision 54568557; both 193-test suites, strict diagnostics and minimal authenticated facade sessions pass. The full Linux workspace passes 1,301 tests and 702 minimal-feature tests.
- [x] [patch] Measure and archive all 108 controlled process pairs at source tree 60db5f8c: 105 exact-byte successes and three unchanged 60-second loss-case failures. Independently recalculate rates/percentiles and verify both kernel queue-byte deltas; the performance gate remains failed.
- [x] [minor] Expose passive recovery metrics and preserve all profile outcomes, verified progress, namespace UDP/socket counters and both peer diagnostics; 17 focused real tests and strict diagnostics pass.
- [x] [minor] Replace ignored UDP options with one PAL-owned applied buffer/broadcast/multicast contract; share native queue requests and readback with QUIC (ADR 0078). The ignored-buffer regression fails before the change and passes afterward; all 338 native transport/async tests and strict diagnostics pass.
- [x] [patch] Rebase on main 7828f5df, retaining PR 617 capture decoding and PR 619 page ownership. Initial preservation audit matches 47 transport/evidence paths and 64 main code paths; the subsequent private Windows module-gate correction follows its actual consumer features. both canonical locks resolve without changing external dependency identities.
- [x] [patch] Repair the socket-test embedded inventory without changing assertions; the full macOS workspace passes 1,329 tests with nine existing ignored probes, and the 397-test native integration suite and minimal facade session pass.
- [x] [major] Classify the existing PAL-owned UDP configuration migration against current main and document it under Breaking; four native API surfaces pass, while async configuration reports the intentional type-owner change.
- [x] [patch] Compare five native API surfaces with the previous QUIC head; all 980 applicable checks pass without a new API regression.
- [x] [patch] Reproduce the fresh Windows native job's six unused-WebView diagnostics with the installed MSVC target; align the shared-limits module with the actual optional WebView2/browser/test consumers and retain warning-denying default/minimal gates.
- [ ] [minor] Complete protected default-branch promotion before release.

Next increment: evaluate the recorded recovery-policy limit with matched full-domain and competing-flow measurements, then complete protected review/promotion for [provider PR 618](https://github.com/ryancinsight/Moirai/pull/618) and its dependent Metis change.
