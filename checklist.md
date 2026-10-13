# Moirai transport checklist

Active target: workspace 0.9.0; moirai-quic 0.1.0. Phase: Closure.

- [x] [minor] Specify native QUIC ownership, trust, ALPN, resource and cancellation contracts in ADR 0075.
- [x] [minor] Implement shared cancellable socket readiness and one scheduler/timer/UDP runtime boundary.
- [x] [minor] Implement authenticated independent streams, admission, complete framing, error retention and acknowledged FIN.
- [x] [patch] Verify real UDP loss/reordering, arbitrary bytes, malformed frames, exact limits, TLS rejection, stream isolation, backpressure and spawner failure.
- [x] [patch] Prove the readiness retirement regression fails against the preceding implementation and passes with the registration-lock transition.
- [x] [patch] Check native API compatibility against 0.8.2: 669 applicable checks pass under the stronger patch compatibility override.
- [x] [patch] Verify the canonical provider lock without path patches; warning-denying Clippy, formatting and all 129 PAL/facade/QUIC tests pass.
- [ ] [patch] Complete the canonical Metis Git lock gates after provider review publication.
- [ ] [minor] Complete review/default-branch promotion and Windows/Linux native transport evidence before release.

Next increment: publish the verified provider revision, adopt it in the Metis Git lock and run canonical integration gates.
