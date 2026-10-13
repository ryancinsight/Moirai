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
- [ ] [patch] Collect Linux/Windows transport and controlled netem evidence from the committed review revision.
- [ ] [minor] Complete protected default-branch promotion before release.

Next increment: review [provider PR 618](https://github.com/ryancinsight/Moirai/pull/618) and its dependent Metis change, then collect Windows/Linux transport evidence before release.
