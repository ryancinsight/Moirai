# Moirai gap audit

This register contains unresolved risks with no active implementation. A risk
moves to [backlog.md](backlog.md) when its re-open trigger fires.

## Native QUIC transport — 2026-10-13 [minor]

ADR 0079 defines matched stock CUBIC, NewReno and experimental-BBR evaluation.
Policy selection preserves the original CUBIC default and all security/resource
contracts. The independent original 108-pair promotion job remains unchanged.
The first comparison records 105/108 CUBIC, 104/108 NewReno and 107/108
experimental-BBR complete reports; two failures come from collector cleanup
and obscure those pair outcomes. They remain failed experiments. All three
maximum loss cases fail under CUBIC/NewReno and complete under experimental
BBR, whose mixed-policy sampled fairness and larger queues prevent default
selection. The collector joins its observer before namespace cleanup and checks
actual absence after a raced kill; a fresh complete matrix is required. All
30 competing experiments retain zero UDP-error deltas. One unshaped CUBIC pair
records two receive-buffer drops and still verifies all bytes; controlled loss
failures retain zero error deltas. No root-cause closure follows from that
unshaped observation. The recovery gap and protected promotion gate remain open.

Fresh Linux native validation passes at 03d640b8. The fresh Windows QUIC job
fails six unused-WebView diagnostics because the shared-limits module compiles
without its optional WebView2 consumer. Installed MSVC cross-compilation
reproduces the same failure. The corrected module predicate is
`(Windows && WebView2) || Browser || Test`; default Windows diagnostics pass
without suppression. Default, minimal and WebView2-enabled Windows PAL
cross-compilation and strict browser PAL diagnostics pass. The complete native
workspace still passes 1,329 tests with its nine existing ignored probes.
Actual Windows WebView2/capture execution passes at 03d640b8; the final native
QUIC runtime must still pass at the corrected revision.

Current-main integration uses 7828f5df and preserves PR 617 capture admission
and PNG validation plus PR 619 page mechanisms. The transport/evidence audit
compares 47 paths with the previous PR head; all match, allowing only the UDP
ADR comment renumbering. Of 64 main code paths, 63 retain exact bytes and the
private PAL module gate follows its actual optional consumers. The socket-test split
requires its two leaf fixtures in the embedded benchmark inventory; the
unmodified contract fails without them and the complete corrected workspace
passes 1,329 tests (nine existing ignored probes). The native integration suite
passes 397 tests with no skips; minimal authenticated facade exchange passes.
Both canonical locks resolve and preserve every external dependency identity.
Current-main API comparison passes 979 of 980 applicable checks and reports the
existing async UDP type-owner migration. CHANGELOG.md records the pre-1.0 break;
this is not a backward-compatibility pass. Comparison with the previous QUIC
head passes all 980 applicable checks across five native surfaces. Local
validation runs on macOS; fresh
Windows/Linux runtime and protected default-branch promotion remain gates.
Historical platform and controlled-profile evidence below retains its source
revisions and does not certify the reconciled tree.

ADR 0078 closes ignored UDP configuration: positive, native-range-checked
receive/send requests apply through PAL, typed multicast joins the specified
group/interface, and readback reports kernel queue values. The original ignored
buffer regression fails with a 786,896-byte effective queue instead of the
requested 4,096 bytes; the corrected implementation matches a separate native
reference. All 338 transport/async tests and strict diagnostics pass locally,
including exact IPv4/IPv6 unicast, empty datagrams, IPv4 multicast, native option
failure cleanup and configured authenticated QUIC with acknowledged FIN. IPv6
multicast covers group/family validation and real invalid-interface failure;
positive IPv6 multicast delivery remains unmeasured. Queue sizes do not change
protocol windows or deadlines and supply no evidence to close the recovery limit.

Moirai 0.9.0 includes moirai-quic 0.1.0 in every native facade build, without
a QUIC feature flag. The default suite passes 187 tests; the minimal facade
completes authenticated frame exchange and acknowledged teardown. Browser
facade compilation excludes native QUIC. CI checks both native modes. The provider owns
Moirai task/timer/PAL driving around Quinn, TLS 1.3 trust and ALPN, bounded
connections/streams/handshakes/framing and terminal cancellation. Reliable binary
sessions multiplex over UDP; HTTP/3/WebTransport mapping remains outside this
slice. Quinn uses Tokio sync primitives without its runtime and links ring's
native code. Shared readiness retires notifications under the registration lock;
legacy UDP allocates no fan-out state until the shared poller API is used.

All 187 HTTP/PAL/facade/QUIC tests pass against the canonical Git lock without path
patches; native warning-denying Clippy and changed-source formatting pass.
Native loopback checks cover full byte domains, finite loss and pathological
perpetual loss, certificate/name/ALPN/mutual-TLS rejection, truncation, resource
limits, stopped spawners, FIN acknowledgement and sibling isolation. The earlier
readiness implementation deadlocks under the new late-subscriber regression;
the corrected transition passes. Compatibility runs 892 applicable checks with
the stricter patch override against 0.8.2. Metis passes 176 real backend/IPC/core
tests and native workspace checks against the published Git provider with no
path patches. The two-process mutual-TLS CLI passes on IPv4 and IPv6. Canonical provider promotion and
controlled-profile collection remain active gates. Both hosted Linux and Windows
pass strict diagnostics, 193 socket/HTTP/facade/load tests and the minimal native
session at published revision 54568557. The [Linux workspace gate](https://github.com/ryancinsight/Moirai/actions/runs/38459278652)
passes 1,301 tests (nine existing ignored probes) and 702 minimal-feature tests.
The reconciled provider also passes all 27
full-domain local process-pair probes. No wide-area performance or
migration result is inferred from loopback correctness.

Controlled CI measures all 108 process pairs at one source tree: 105 validate
exact bytes; all three maximum-frame 16-stream loss cases exceed 60 seconds.
All three diagnostic repetitions reproduce that limit with continuing verified
progress and zero namespace UDP errors. Recovery snapshots identify contracting
CUBIC windows. [Archived evidence](moirai-quic/measurements/controlled/2026-10-13/README.md)
retains source, queues, samples and both peer traces; the active recovery-policy
evaluation belongs to MOI-QUIC-RECOVERY-001 in backlog.md. No deadline or workload
adjustment establishes a pass.


## Rust page provider acceptance — 2026-10-13 [minor]

Actual PR 619 Windows CI reports native permission origin separately from the
full page-message URL. Two same-document geolocation calls yield one native
DENY and two browser errors; they do not prove independent callback evidence.
The repaired fixture navigates to a second packaged document after real first
settlement under the same profile/origin/controller, retaining the two actual
DENY/two error requirement. Actual Windows
[job 115537873626](https://github.com/ryancinsight/Moirai/actions/runs/38467480001/job/115537873626)
passes at [reviewed source](https://github.com/ryancinsight/Moirai/commit/0d42510b50cec901852124ddbdcd0727632ebcb9), including actual browser exit and
caller-owned profile cleanup. Metis Rust page handoff/four-kind acceptance and
consented ordinary-browser positive cleanup remain pending.
That source passes 25 native policy/destruction tests and four actual capture
fixtures, including two 100-host cases. WebView2 is `153.0.4234.48`, SDK bindings
are `0.39.1`, and the repeated two-document fixture takes 7.410 seconds.

Workspace 0.9.0 implements the actual WebView2 browser bridge, explicit callback
teardown/counts, browser Number formatting, real permission requests and owned
child working directories. Fresh-profile native admission checks caller-owned
empty local storage, nonsaved DENY decisions and the actual browser exit before
caller cleanup. [ADR 0077](docs/adr/0077-rust-page-provider-ownership.md) owns
the selected contracts and its source-adjacent design records failure behavior.

Eleven portable permission state/identity tests pass, including all four busy
slots, cancelled guards, exact media ownership sums, deadline caps, stale
generations and the entire decimal-string identity domain. Strict WASM and
native PAL Clippy pass. Shared message/directory tests and Windows provider
diagnostics cover their own source boundaries; the installed repeated-denial
fixture is compiled but has no local Windows execution evidence.

The complete native PAL/transport suite passes 148 tests without skips. Seven
actual Chrome tests verify JSON, DOM callback and Number contracts without
ignored or filtered tests; they do not establish native chrome.webview handoff.
Locked Rustdoc comparisons pass 196 additive-API checks for each of native
transport, WASM PAL and Windows PAL. Actual final-destruction and nonsaved
repeated-denial fixtures are mandatory Windows runtime gates; the source-bound
actual Windows job above now passes the provider capture/repeated-denial/cleanup
scope. Consumer Rust bridge acceptance remains separate.

Actual native Rust-page handoff, four-kind
consumer acceptance and consented positive/late media cleanup remain open in
the two active acceptance items. Two geolocation dispatchers remain realm-owned
after the first request; up to four genuine settlement guards remain visible
after cancellation. A cleared watch may never callback, so its guard can remain
busy until document destruction. A failed DOM/bridge callback removal retains
its counted owner and prevents further admission. Neither portable tests nor
successful compilation establish native events, granted device cleanup or
complete provider-private allocation accounting.

## WebView2 capture boundary reconciliation — 2026-10-13 [patch]

Workspace 0.8.3 corrects the inclusive decoded-byte law from PR
[505](https://github.com/ryancinsight/Moirai/pull/505): `L = 3q - padding`.
The unchanged decoder fails three value tests; the correction passes all five,
including complete nonconstant payloads at 64 MiB minus two, minus one and
exactly 64 MiB, and rejects the first excess byte before output allocation.
The production `webview2` feature enables its PNG validator. Actual Windows
library compilation and warning-denying Windows Clippy pass without relying
on development dependency resolution. The portable/Apple provider suite passes
157 tests in 2.029 seconds; one existing runtime-dependent probe is skipped.
The full workspace passes 1,278 tests in 7.217 seconds with nine pre-existing
ignored probes, including the embedded release version contract.
Native Clippy, browser PAL compilation and warning-denying Rustdoc pass.
The new Windows job runs the existing installed preview tests; hosted runtime
results remain a merge gate. The 30-run parallel-host stability item remains
open. This byte law does not bound CDP response or decoder-private allocations.
[ADR 0076](docs/adr/0076-webview2-capture-byte-boundary.md) owns the proof.

## Apple external link evidence — 2026-10-13 [minor]

The starter's new-window logo has no native popup handler before
moirai-webview 0.2.0. The provider now owns exact HTTPS validation, signed
bundle authority and one OS handoff in WKUIDelegate, with no remote WebView.
Twenty policy/resource/Foundation tests and 196 compatibility checks pass.
The real native preview records one successful Metis link handoff. Metis PR
[501](https://github.com/ryancinsight/metis/pull/501) records canonical consumer
bundles and the user-authorized Safari destination check. Remaining physical
mobile and native lifecycle acceptance remain consumer gates. LinkActivated
does not prove event trust (ADR 0074).

## Stream dictionary regression — 2026-10-12 [patch]

The real Metis HTTP page fails at Fetch stream decoding because a Web IDL
read-result dictionary has no runtime class constructor. Workspace 0.8.1
checks object shape and validates completion/chunk fields instead of using
instanceof. The visible pre-fix readiness error is the red oracle; live
success, byte bounds, deadline and cancellation remain the consumer gate.

## Browser HTTP ownership — 2026-10-12 [minor]

The PAL supplies bounded Fetch and document observations in 0.8.0 so Metis can
remove its JavaScript protocol/authority duplicate. All 1,257 native workspace tests pass within 60 seconds, with nine pre-existing
ignored probes; the PAL passes native/WASM warning-denying checks. Live consumer Fetch, stream-bound, deadline and cancellation evidence remains
the active board gate.
The WASM public surface passes all 196 applicable cargo-semver-checks checks
against 0.7.2 using compiler-produced rustdoc JSON (the automatic generator
fails at Rust 1.97 WASM target probing). The provider does not bypass browser
CORS, CSP or native permission policy.

## Apple packaged Fetch status — 2026-10-12 [patch]

The native Metis Greet failure exposes a missing success status in the shared
WebView response. `NSURLResponse` delivers the WASM bytes, but wasm-bindgen
0.2.129 rejects its Fetch status zero. moirai-webview 0.1.1 returns an actual
`NSHTTPURLResponse` with status 200 and exact MIME/length headers. The regression
fails against the previous response; eight real resource/response tests pass
after correction. Native macOS normal, Unicode and empty-name greetings pass.
Remaining Apple lifecycle evidence stays in MOI-APPLE-HOST-EVIDENCE-001.

<a id="MOI-GAP-STRUCTURE-001"></a>
## MOI-GAP-STRUCTURE-001 — Historical module-size findings are stale
- risk: The 2024 assessment reported 18 modules above 400 lines, but later splits make that count unfit for current planning.
- evidence: The retired historical board carried ISSUE-003 without a current-revision census or a DoR-scoped target.
- re-open trigger: A fresh structural scan finds a file above 500 lines containing more than one operation family.

<a id="ISSUE-226"></a>
## ISSUE-226 — Lazy priority-plane allocation remains unproved
- risk: Lazy construction could remove the last retained queue-plane allocation but requires safe late stealer publication.
- evidence: The delivered per-plane capacity policy reduced measured retained bytes from 1,572,864 to 540,672 at 24 workers; the remaining proposal lacks its required Loom model.
- re-open trigger: A measured consumer retention budget justifies the remaining saving and funds the stealer-publication model plus paired benchmark.

<a id="ISSUE-132"></a>
## ISSUE-132 — Rayon compatibility is intentionally bounded
- risk: A future consumer may require indexed cardinality, collect/unzip targets, interleave/step, logical-output blocks, numeric or fallible terminals, positions/windows, serial-inner flattening, mutation, intersperse, equality-checked zip, partition mapping, or sorting outside the audited subset.
- evidence: Existing transforms, reducers, slicing, sorting, mutation, and fallible terminals have value and comparison coverage; full Rayon API parity is not claimed.
- re-open trigger: A named consumer requires a missing surface that can retain a dedicated Moirai boundary and same-run comparison.

<a id="MOI-GAP-HTTP2-001"></a>
## MOI-GAP-HTTP2-001 — HTTP/2 has no current consumer
- risk: A future consumer may require multiplexed HTTP/2 behavior beyond the bounded HTTP/1.1 transport.
- evidence: The runtime roadmap makes HTTP/2 conditional while select ergonomics, IPv6, and graceful shutdown remain active work in `MOI-TOKIO-PARITY-STAGE-B1`.
- re-open trigger: A named consumer supplies an HTTP/2 acceptance workflow and resource-bound contract.

<a id="MOI-GAP-GPU-COSCHEDULING-001"></a>
## MOI-GAP-GPU-COSCHEDULING-001 — Device queue co-scheduling is provider-dependent
- risk: Moirai exposes accelerator route and occupancy metadata but does not own persistent device queues or hardware scheduling.
- evidence: ADR 0041 (revised by Moirai PR 501) keeps Hephaestus out of Moirai's dependency graph; stream/queue consumption and persistent kernels remain provider work.
- re-open trigger: Hephaestus exposes a stable queue/persistent-kernel contract and a consumer supplies hardware acceptance evidence.

<a id="MOI-AUDIT-PM-007"></a>
## MOI-AUDIT-PM-007 — PM fact ownership needs a structural audit
- risk: The earlier dead-checkout audit could not determine mechanically whether each live PM fact has one owner.
- evidence: The finding was held for manual review without an item scope or sampled duplication evidence.
- re-open trigger: A fresh board/ADR/doc scan identifies a duplicated live fact and names both authoritative candidates.

<a id="MOI-ADAPTIVE-THRESHOLD-PREMISE-2026-08-31"></a>
## MOI-ADAPTIVE-THRESHOLD-PREMISE-2026-08-31 — Adaptive dispatch lacks a body-cost model
- risk: One element-count threshold cannot select correctly across cheap and compute-heavy bodies; changing it can trade one measured regime for an unmeasured consumer regression.
- evidence: Retained measurements place the one-multiply crossover at 4,096–8,192 elements, `sqrt` plus `ln_1p` at 512–1,024, and chained fused operations below 512.
- re-open trigger: A consumer supplies a representative body-cost distribution or a body-cost-aware policy is specified without per-element dispatch.

<a id="MOI-GAP-ROADMAP-001"></a>
## MOI-GAP-ROADMAP-001 — Historical readiness goals lack acceptance contracts
- risk: API stabilization, production readiness, platform expansion, extended examples, tooling integration, and continuous performance work are aspirations rather than executable items.
- evidence: The retired checklists named these topics without scope, dependencies, value oracles, or concrete next steps.
- re-open trigger: A named consumer or measured defect supplies a DoR-complete acceptance workflow for one topic.

<a id="MOI-GAP-CONSUS-TRANSPORT-001"></a>
## MOI-GAP-CONSUS-TRANSPORT-001 — Object-storage adoption remains consumer-owned
- risk: Consus has not recorded a current comparative MinIO/toxiproxy result or the eventual default flip away from its legacy Tokio/rusoto path in this repository.
- evidence: Atlas ADR 0045 owns the cross-repository decision; Moirai already supplies the store-agnostic TLS/HTTP transport and must not own S3 protocol work.
- re-open trigger: The Consus owner requests a Moirai transport correction exposed by its paired benchmark or contract tests.

<a id="MOI-GAP-SLAB-GENERATION-32BIT-001"></a>
## MOI-GAP-SLAB-GENERATION-32BIT-001 — Slab free-list generation is 16 bits on 32-bit targets
- risk: On wasm32-with-atomics, armv7, and i686 the packed free-list head keeps a 16-bit generation, so 65,536 successful CASes inside one thread's load-to-CAS window can reinstall an occupied slot as the free-list head (ABA); 64-bit targets need 2^32.
- evidence: Read of `moirai-core/src/pool/slab.rs` packing and the `insert`/`remove` CAS sites; the interleaving is derived, not reproduced.
- re-open trigger: A 32-bit target becomes a supported deployment for `SlabAllocator`, or a pause-hook test on `i686-unknown-linux-gnu` reproduces the reinstall; the remedy is a `u32` index plus `u32` generation in an `AtomicU64`, as `LockFreeStack` does.

<a id="MOI-GAP-CHASE-LEV-EXACTLY-ONCE-001"></a>
## MOI-GAP-CHASE-LEV-EXACTLY-ONCE-001 — One unreproduced exactly-once failure predates the slot-claim fix
- risk: `chase_lev_exactly_once_high_thief_contention` failed once on 2026-07-27 in a full workspace run under about 11 concurrent compiles, assertion text not captured; the test has no timing dependence, so the failure was a lost, duplicated, or out-of-range consume.
- evidence: Commit 12abfacbc5a7f0aa34a953c8983cb4abaa51c65a re-reads `bottom` after the steal claim, and `moirai-scheduler/tests/loom_chase_lev_slot_claim.rs` shows the fence-free pop without that re-read takes an item twice; about 75 reruns of the test family at 5-way concurrency and a full run on main did not reproduce the failure, so the link to that defect is unproven.
- re-open trigger: Any `chase_lev_exactly_once_*` failure; capture the assertion text first, which separates a duplicated consume from a lost one.

<a id="MOI-GAP-CHASE-LEV-LOOM-COVERAGE-001"></a>
## MOI-GAP-CHASE-LEV-LOOM-COVERAGE-001 — Loom checks the steal gate, not the deque's indices
- risk: Production `top`, `bottom`, the array pointer, and slot states use `std` atomics, so loom sees only the steal gate (compiled in by path). The owner/thief fence protocol is checked by a separate protocol copy in `tests/loom_chase_lev*.rs`, not the production code. The gate's admission, backoff, and drain stay SeqCst because weaker orderings exhaust the gate model's branch budget.
- evidence: `moirai-scheduler/src/deque/chase_lev/inner.rs` and `storage.rs` import `std::sync::atomic` unconditionally. Under Acquire admission or drain, or Relaxed backoff, `tests/loom_chase_lev_resize_gate.rs` exceeds loom's branch cap or the 60 s budget, while the SeqCst model closes in under 2 s. Relaxing the claim release fails three gate tests on an assertion; relaxing the guard exit passes all seven, so loom does not witness that edge.
- re-open trigger: The deque's atomics become loom-instrumented under `cfg(loom)` and an integration test drives the public `ChaseLevDeque`, or the gate model's retry path is restated so it closes under Acquire.

## Apple host evidence — 2026-10-11

`moirai-webview` 0.1.0 owns shared packaged resources and WebKit, with separate
AppKit/UIKit lifecycles. The 39 transport and six resource tests pass. A
Darwin zombie-only group cleanup fix preserves exited status and terminates
real contained descendants; it does not mask live-member permission failures.
Native macOS and iOS compilation, simulator launches and packaged WASM delivery
are separate from observed DOM/input behavior. The Mac lock prevents native UI evidence; device
signing, physical execution, permission suites and plugin capability closure
remain open under MOI-APPLE-HOST-EVIDENCE-001. No universal Apple sandbox claim
or completed mobile parity claim follows from these builds.

The iPad simulator also reports duplicate Apple accessibility classes from
WebCore/WebKit. Application resource delivery succeeds; the interactive
accessibility journey remains pending. No valid device-signing identity exists
on this Mac, and the physical iPhone reports unavailable.

<a id="MOI-GAP-ASYNC-STATE-LOOM-001"></a>
## MOI-GAP-ASYNC-STATE-LOOM-001 — The async task state machine has no loom model of the production code
- risk: Lifecycle, future, and result cells in `moirai-executor/src/hybrid/async_state/` are `UnsafeCell` accesses justified by the POLLING and QUEUED states excluding other accessors. Cancel, poll, and completion interleavings are argued in `// Safety:` comments but never model-checked against production code.
- evidence: `completion.rs` and `polling.rs` create `&mut` to the lifecycle, future-present, and result cells; `tests/loom_task_waiters.rs` and `tests/loom_queue_handoff.rs` model their own protocol copies, not `async_state`; cancellation appears only in `async_state/tests/shutdown.rs`, which runs without an interleaving model.
- re-open trigger: A loom model drives the production `AsyncFutureState` through poll, cancel, and completion racing on two threads and asserts exactly-once future drop and result publication.

<a id="MOI-GAP-UNSAFE-SAFETY-COMMENTS-001"></a>
## MOI-GAP-UNSAFE-SAFETY-COMMENTS-001 — Production `unsafe` blocks lack a checked SAFETY convention
- risk: The workspace does not deny `clippy::undocumented_unsafe_blocks`, so the `// SAFETY:` convention is prose only. A four-line lookback reports 108 of 488 production `unsafe` blocks without an adjacent SAFETY comment. That count is an overestimate, since many sites carry the comment two or three lines above a `let` that wraps the block.
- evidence: 53 of the 108 are in `moirai-pal`, mostly WebView2 and Win32 COM calls (`windows/webview/host/*.rs`, `windows/window/*.rs`) that repeat one invariant per call. `moirai-executor/src/hybrid/async_state/` writes `// Safety:` in lowercase, unlike the `// SAFETY:` convention. The `moirai-parallel` `Vec::from_raw_parts` sites rely on `MaybeUninit<R>` sharing `R`'s layout, and that argument is correct.
- re-open trigger: Per-crate `undocumented_unsafe_blocks = "deny"` lands, with the COM call sites consolidated behind one safe wrapper per interface and the `Safety:` spelling normalized.

## Rust browser HTTP consumer evidence — 2026-10-12

Metis PR [493](https://github.com/ryancinsight/metis/pull/493) merges at
`aa04a4a9fdbc3d2a6f502ade18e2cf1c58c5de73`, consuming Moirai 0.8.2
`6c7f39dff127f1accfc02ede4a5d59cb34ec9cc5`. The local in-app browser
verifies every response byte at lengths 0, 1 and 16,384; excess declared and
streamed payloads reject. Bodyless 204/205/304 preserve status and zero body
bytes, including a 304 representation length above the received-body budget.
Exact Unicode POST, no followed redirect, header/body deadlines and peer EOF
after Fetch-task drop pass. HTTP Stop releases four listeners; workbench
stop/remount restores 40 application listeners while its three shell handles
remain. Pending-reset isolation leaves the new mount unchanged.

The bounded HTTP consumer evidence item is complete. These observations do
not count provider-private allocations or establish cross-engine/hardware
equivalence; broader networking and permission acceptance remains open.
