# Moirai backlog

<a id="moirai-stable-qualification"></a>
## MOIRAI-STABLE-QUALIFICATION — moirai stable-release qualification [patch] — todo
- priority: verification; milestone: M-async-io-transport; basis: `c24b7ad3e2c1`.
- Outcome: Finish the controlled transport evidence and reconcile the public WebView package. Follow the [release scope](docs/adr/0088-release-scope-1-0.md) and [milestone exits](backlog/milestones.md). Planning targets for `moirai-runtime` are 0.12.1 if Runtime requires a compatible fix, then 0.13.0, then 1.0.0. Verification alone can advance the current version.
- Representative manifest `moirai-runtime 0.12.0`. Target 1.0.0. Index coverage at review: 18/19 publish-enabled workspace versions.
- [x] [patch] Release [moirai-crypto 0.12.1](https://crates.io/crates/moirai-crypto/0.12.1) and verify the unwaived dependency audit, archive provenance, six RSA signature schemes and independent TLS consumer at the supported Rust floor. Downstream adoption is tracked in [Metis #589](https://github.com/ryancinsight/metis/pull/589) and [RITK #1107](https://github.com/ryancinsight/ritk/pull/1107).
- [ ] [patch] Follow the existing QUIC evidence work ([#716](https://github.com/ryancinsight/moirai/pull/716) and [#703](https://github.com/ryancinsight/moirai/pull/703)). Replay loss, cancellation, bounded buffering and shutdown tests under the declared native hosts. Record the measured recovery and fairness results.
- [ ] [patch] Establish the distribution plan for moirai-webview 0.12.0. Verify the normalized published dependency graph or document its owner-approved disposition before counting the public cohort.
- [ ] [major] Close the core execution and async transport exits against the actual runtime, with network and process tests in Metis and a clean independent consumer.
- [ ] [patch] **0.12.1 if Runtime requires a compatible fix.** Qualify runtime cancellation and shutdown on the existing runtime line. Record outstanding handles and verify joined tasks finish or return the documented cancellation outcome. Crypto 0.12.1 has its own receipt. [Source and baseline](gap_audit.md#incremental-release-baseline).
- [ ] [minor] **0.13.0.** Exercise QUIC loss recovery with a bounded send buffer on declared native hosts. Check ordered payload bytes, cancellation, fairness and shutdown. Resolve WebView distribution separately from the runtime cohort.
- [ ] [major] **1.0.0-beta.1 → 1.0.0-rc.1 → 1.0.0.** Freeze the supported surface at beta. A clean transport caller reproduces the controlled recovery and drains outstanding work on the candidate graph. Repeat the applicable exits on the exact RC before stable promotion. Attach exact-revision evidence for the local exits and this member’s [Atlas qualification criteria](https://github.com/ryancinsight/atlas/blob/17b0124a7548ad81e817284afc5cb115c1d1bc78/docs/parity/repositories.json).

This is the single source of truth for active Moirai work. Items are ordered by
priority (`correctness`, `architecture`, `verification`, `tightening`, then
`feature`) and dependency. Hosting records claims and review state; this file
contains only `todo` and `blocked` work. Risks without an active implementation
belong in [gap_audit.md](gap_audit.md).

<a id="MOI-QUIC-PROMOTION-001"></a>
## MOI-QUIC-PROMOTION-001 — Promote verified native QUIC [minor]
- status: todo
- priority: feature
- milestone: M-async-io-transport
- outcome: Moirai 0.11.0 and moirai-quic 0.1.0 are available at the canonical provider revision; Metis resolves its Git lock from that published source. (0.10.0 was taken by main's clipboard release while this PR was in flight.)
- acceptance: ADR 0075, real UDP interoperability/loss/bounds/cancellation tests, clean native diagnostics, compatibility checks and canonical Git lock pass before promotion. Windows/Linux transport evidence remains a platform gate.
- scope: provider publication and consumer revision adoption; no credentials or trust-store configuration.
- current increment: native QUIC, shared readiness, fallible facade spawning, bounded framing and fresh-key Metis sessions are implemented in [PR 618](https://github.com/ryancinsight/Moirai/pull/618). QUIC is required in native facade builds: 187 default tests, the minimal authenticated session, browser facade compilation, strict diagnostics and 892 applicable compatibility checks pass. CI checks default/minimal native sessions. Metis's prior 176-test canonical verification passes; the dependent consumer review adopts the updated provider Git pin. Default-branch promotion remains pending.
- next step: close the measured CUBIC loss-profile limit below and complete required review/checks before protected provider/consumer promotion. Both hosted native platform gates pass; the initial controlled baseline and full loss diagnosis are [archived](moirai-quic/measurements/controlled/2026-10-13/README.md).
- [patch] Windows integration correction: the [this commit](https://github.com/ryancinsight/Moirai/commit/03d640b890af232616a9dc85ffbddc38b3d8d4f8) native job compiles shared WebView limits without their optional consumer and fails six strict diagnostics. The installed MSVC target reproduces that failure; the module now compiles iff Windows WebView2, browser code or portable tests consume it. Default/minimal native CI retains strict diagnostics.
- [patch] Main integration: the provider rebases onto 7828f5df and retains the complete PR 617/619 implementation. Canonical locks, 397 native tests, the minimal session and all 1,329 workspace tests pass on macOS; nine existing ignored probes remain. The embedded inventory includes the unchanged stream/datagram fixture leaves.
- [major] Compatibility classification: current-main comparison reports the existing async UDP type-owner migration; CHANGELOG.md and ADR 0081 define the pre-1.0 migration. Fresh Windows/Linux results for this reconciled source and protected review remain promotion gates.
- UDP increment: ADR 0081 replaces ignored queue/multicast options with one applied PAL contract used by async UDP and QUIC. All 338 native transport/async tests and strict diagnostics pass locally; kernel-default queues and the full controlled performance contract remain intact.

<a id="MOI-QUIC-RECOVERY-001"></a>
## MOI-QUIC-RECOVERY-001 — Evaluate congestion policy under measured loss [minor]
- status: todo
- priority: architecture
- milestone: M-async-io-transport
- outcome: The chosen production recovery policy supports the admitted message/stream domain with measured throughput, latency and competing-flow behavior.
- acceptance: retain the same complete payloads, message counts, deadlines and real byte validation; compare supported congestion policies on matched profiles and document fairness, queue growth and failure bounds before changing defaults. Preserve one canonical transport API and a bounded policy abstraction if variation is required.
- evidence: The 16-stream, 128-message, 65,560-byte 1% loss case exceeds 60 seconds in all three repetitions. Verified progress continues and all namespace UDP error deltas are zero; CUBIC recovery windows contract. No socket-buffer or spawner bypass closes this limit.
- [patch] Diagnostic evidence complete: the [October 14 study](moirai-quic/measurements/recovery/2026-10-14/README.md) independently verifies all 108 original pairs per policy, all 30 competing experiments and actual traces in both peers at security-patched [this commit](https://github.com/ryancinsight/Moirai/commit/140d60cf43e152b816ccd8d4ff5ac12a0e53dfdf). CUBIC/NewReno remain 105/108; experimental BBR remains 108/108. Six early server handshakes remain unestablished while clients switch from one Handshake-space PTO to repeated Data-space PTOs. Established-session congestion, pacing and flow-credit notifications coexist. BBR records mixed-policy sampled Jain indices 0.557124–0.581728 and a sampled 10,313,912-byte congestion window; preceding [this commit](https://github.com/ryancinsight/Moirai/commit/733afc9fcce3b70b4ceef512914e83f6e6142841) records 133,608,526 bytes. Upstream startup-growth PR 2798 remains unmerged. No default change is accepted.
- [patch] ADR 0082 implements a separately built, checksum-verified read-only engine observer. Canonical transport/locks and the independent original gate remain unchanged. All 81 corrected local mutual-TLS process pairs retain exact bytes, cross-space flight sums, cross-peer handshake CRYPTO offsets and actual frame/timer/credit state. Retain the rejected off-timer PTO-selector capture; all six matched Linux shards complete at [this commit](https://github.com/ryancinsight/Moirai/commit/03dcc4a31d7f793e4cb1cb7d96ff74f47033946c) with independent full-domain/state auditing: CUBIC/NewReno 105/108, BBR 108/108, and 39/60 competing pairs complete. The pristine original gate reproduces 105/108 at the same source tree.
- [patch] Focused handshake evidence: all six early Linux competing failures retain 634 queued client Handshake CRYPTO bytes, zero Handshake flight and Data PTOs; the server consumes 2246/2880 bytes. A real maximum-workload single-datagram-loss experiment completes 17/27 baseline versus 27/27 with a focused experimental recovery patch; all 23 native QUIC tests pass against the isolated candidate. The patch remains an artifact, not the canonical dependency. Its full Linux policy/fairness/latency/queue matrix and a reviewed supported production path remain required. Established throughput remains separate.
- next step: measure the focused handshake candidate across the full Linux domain and competing flows and establish a supported reviewed production path before adoption. Continue the distinct established-session throughput investigation using the archived flight/flow-credit/pacing timelines. Retain all complete, unsuccessful and canceled studies; preserve original deadlines and the independent full CUBIC promotion gate. No tolerance, window enlargement or unreviewed upstream fix establishes acceptance.
- [patch] Security prerequisite verified: required CI reports four Rustls 0.23.45 advisories (RUSTSEC-2026-0346/0347/0348/0349). The shared dependency and independently consumed QUIC provider require released 0.23.46; both locks update only Rustls's external identity. The fresh complete policy/shared-queue study and independent original CUBIC gate retain separate source inventories. Every applicable non-performance CI job passes, including Linux/Windows, workspace, supply-chain and lockfile integrity; the original CUBIC performance gate still fails.

<a id="MOI-WEBVIEW-PREVIEW-CAPTURE-TIMEOUT-2026-09-24"></a>
## MOI-WEBVIEW-PREVIEW-CAPTURE-TIMEOUT-2026-09-24 — Stabilize preview capture under parallel hosts
- status: todo
- priority: correctness
- milestone: M-core-execution
- needs: none
- outcome: The six default-profile WebView2 tests (four preview, mapped-folder, packaged-bridge) complete in every full parallel `--run-ignored all` run.
- acceptance: Thirty consecutive runs in each of `--no-default-features --features webview2` (CI set) and `--features webview2` have zero `TimedOut` from host creation, frame waits or capture in those tests and none past the 30 s SLOW mark, every deadline unchanged.
- evidence (base `48bacb85`): every nextest process shares one `<exe>.WebView2` browser. CI set, judge: re-show tests failed 1/30 (47 `TimedOut` frame waits at `preview.rs:107`), test wall time 31.3-52.4 s, 2-4 SLOW per run; mapped-folder failed 29/30 (passed only the fresh-profile run 1), alone 0.66-5.3 s. Here a fresh profile per run still failed it 5/5. Three concurrent processes, equal CPU load: shared folder gives re-show 44.1-49.2 s and mapped-folder 5/5 `TimedOut`; distinct folders 27.3-29.7 s and 5/5 pass. Evidence (`webview2-default-profile` test group, max-threads 1): CI set 29/30 clean, mapped-folder 30/30 in 0.8-6.3 s, re-show median 17.4/17.8 s, SLOW in 1/30. Run 9: all 100 immediate-re-show host creations `TimedOut` (`preview.rs:128`) straight after the 100-host settled test, then the start `Barrier`, waiting for 101 parties of which 100 had panicked, never opened and hung until the 60 s kill (a deadlock; the bounded start line now ends it with the failing host's own panic); unrelated tests ran 2-3x slower in that run. Default set: 30/30 clean, zero `TimedOut`, no SLOW; re-show 10.4-18.1 s, mapped-folder 0.8-4.5 s.
- evidence (spike on `0f3bf9c3`): 12 back-to-back pairs of the settled and immediate re-show tests, each its own process on the default folder, passed 24/24 in 11.1-16.8 s with zero `TimedOut`; `msedgewebview2.exe` processes of the folder were counted at each launch: 0-5 in 12 of the 13 launch counts read (runs 1-6 and the last launch) and 40 in the other (run 6, immediate test). The counts of runs 7-11 were not kept, run 12's first launch was neither read nor recorded as not kept, and a launch-time count cannot separate the prior run's browsers from the new run's own, so the 40 is not attributed to either. The prior-browser-exit hypothesis is neither reproduced nor refuted: the stall has appeared only inside the full parallel run. [PR 697](https://github.com/ryancinsight/Moirai/pull/697) head, 30 full runs per feature set: all six grouped tests passed 60/60, zero `TimedOut`, none SLOW, mapped-folder 0.84-1.89 s; the immediate re-show test took 11.2-29.7 s, one CI-set run (73 s in all) reaching 29.689 s, 0.3 s under the SLOW mark with the same deadlines, consistent with the base rate (one slow run).
- evidence (series at `c3d3ed84`, [PR 700](https://github.com/ryancinsight/Moirai/pull/700)): 30 full runs per set from one archived binary at a fixed path, so the default folder persists across runs; CI set 30/30 clean (immediate re-show 12.4-20.4 s, mapped-folder 1.0-4.7 s); `--features webview2` 29/30: run 6 immediate re-show 33.9 s, SLOW at 30 s, zero `TimedOut`, the whole run 99 s against a 48 s median (median of the 30 wall-time rows of the `--features webview2` per-run table in the PR 700 description) with non-WebView2 tests 2.7x slower than the series median. Run 6 started with no process of the folder (58 of 60 starts had none; runs 14 and 27 saw a prior run's processes still exiting, gone at the kill, and passed), so a lingering prior browser does not explain it; with the 60/60 above that is one slow run in 120.
- scope: `.config/nextest.toml`; `moirai-pal/src/windows/webview/{host,pump.rs,tests.rs,tests/preview.rs,tests/preview/}`.
- next step: Per-host wait spans (`host_creation`, `webview2_environment_wait`, `webview2_controller_wait`, `webview2_navigation_wait`, `navigation_event_wait`, `start_line_wait`, `webview2_frame_wait`, `webview2_capture_wait`) now print the slowest host of each from both re-show tests; repeat the 30-run series per set with `--success-output final` and read each run's `slowest wait` events, so a run that ends before the 60 s kill names its slow wait (a killed run prints no report). The item closes only when the acceptance holds: a run past 30 s fails its clause, so close once the cause is fixed and a series passes every clause.
- links: released [PR 626](https://github.com/ryancinsight/Moirai/pull/626) (`refs/pull/626/head`) holds 5 unlanded commits under this ID; its merge-tree landed-work proof against `48bacb85` conflicts in 5 files.
- basis: `c3d3ed845906e802cc7f9bbd7c719de6021bd8f3`

<a id="MOI-MPMC-NOTIFY-FENCE-COST-2026-09-24"></a>
## MOI-MPMC-NOTIFY-FENCE-COST-2026-09-24 — Remove unnecessary notifier fences
- status: blocked
- priority: correctness
- milestone: M-core-execution
- needs: none
- outcome: Bounded MPMC and hybrid pushes avoid a `SeqCst` fence when no parked receiver depends on it without admitting a lost wakeup.
- acceptance: The two-producer Loom model admits no lost wakeup, pinned-core 4/8-producer rows measured on isolated cores beat the fenced baseline, and the multiset regression passes 1,000 consecutive runs. The loaded-host measurement below leaves the 4/8-producer clause unsupported, not falsified.
- evidence: [PR 698](https://github.com/ryancinsight/Moirai/pull/698) (closed unmerged, `refs/pull/698/head`) A/B-timed the fence-free upper bound in 12 cells (wall time; host about 65% busy; interval widths up to about 54 points): 11 of 12 intervals include zero, 9 of 12 median deltas are gains (best -7.5%, E-cores p8_c4096, CI [-45.5, +8.1]%), and one interval resolves a gain (E-cores p4_c1, -2.9%, CI [-5.8, -0.1]%). PR 698's reading that the receiver and the tail CAS bind is a hypothesis; no profile was taken. Derived there, not model-checked: the head-only notify gate strands a second parked receiver. `HybridSender` is single-producer. Loom positive models exceed the 600 s budget.
- scope: queue ring, MPMC/hybrid channel notifiers, communication ring buffer, Loom model, and paired benchmark.
- blocker: The measurement instrument: wall time on a loaded host cannot resolve the 4/8-producer clause, and instruction counters would not show a fence's latency.
- re-open trigger: A fence-free A/B on isolated cores (affinity, lowered contender priority) or under a host bench lease resolves the 4/8-producer rows, or the oracle is rewritten to a simplification judged by a terminating Loom model plus the multiset regression.
- next step: Re-run the paired benchmark on isolated cores; if a gain resolves, extend `DrainRing` with a second producer and prove the receiver-head condition before changing production ordering.
- basis: `cd3c569e791acbe21527b4cc88a1cc562ddae6c4`

<a id="MOI-PAL-TIMER-FANOUT-001"></a>
## MOI-PAL-TIMER-FANOUT-001 — Share native timer workers
- status: todo
- priority: tightening
- milestone: M-async-io-transport
- needs: none
- outcome: Native PAL timers share bounded worker resources across deadline fanout.
- acceptance: Concurrent pending timers use a constant worker count and preserve the existing deadline, cancellation and waker replacement outcomes under native runtime tests.
- evidence: `TimerState::spawn_sleeper` in `moirai-pal/src/timer.rs` spawns a detached thread per pending timer. The async layer already owns a shared native heap driver in `moirai-async/src/timer/driver.rs`.
- scope: PAL native timer driver, async timer ownership and existing PAL timer contracts.
- next step: Measure worker and retained registration counts over pending fanout, then determine a shared owner without a PAL-to-async dependency.
- basis: `b0e0dd95daf8f6982dea3f60b6ef3c6016bc6e0c`

<a id="MOI-NATIVE-REACTOR-001"></a>
## MOI-NATIVE-REACTOR-001 — Complete native readiness and completion backends
- status: todo
- priority: architecture
- milestone: M-async-io-transport
- outcome: Native file and socket operations reach task wakers through their operating system readiness or completion mechanism without a busy poll.
- acceptance: Windows pins overlapped operations, binds handles once, and maps completions to wakers without thread contention or heap allocation in the poll loop; Linux/BSD register edge-triggered interests, wake exact tasks, and translate hangup/error flags to typed I/O errors.
- scope: Windows completion backend, epoll/kqueue readiness, descriptor registration, and typed event translation.
- next step: Slice 2 of ADR 0067 (Proposed; slice 1, `windows/afd`, is delivered): a `Reactor` implementation over `AfdPort` with a growable slot table replaces `WsaPollReactor`, `SocketLease`, the `POLLNVAL` generation handling, and the connect re-probe in one change. Then slice 3 removes the 10 ms idle tick and the per-iteration `Vec` on every backend; slice 4 (native Windows file I/O) needs its own ADR.
- blocker for the Linux/BSD clause: the acceptance says edge-triggered, ADR 0014 decision 3 records level-triggered plus one-shot dispatch as required by register-after-`WouldBlock`; respecify the clause (judgment tier) before Unix work.
- basis: `3bf9b07808fd2326033f308eacbc2aa4325013fc`

<a id="MOI-WASM-COOPERATIVE-EXECUTOR-001"></a>
## MOI-WASM-COOPERATIVE-EXECUTOR-001 — Bound browser executor turns
- status: todo
- priority: architecture
- milestone: M-core-execution
- outcome: Browser tasks run cooperatively without blocking rendering or input handling.
- acceptance: Immediate work enters through microtasks, no executor turn blocks the main thread beyond 16 milliseconds per frame, budget exhaustion yields through a browser macro-task, and a trace proves task progress while frame and input callbacks continue.
- scope: WASM executor dispatch and browser event-loop scheduling; no native driver thread.
- next step: Specify the turn budget and cancellation ownership against ADR 0007, then add the trace before implementation.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-WASM-WORKERS-001"></a>
## MOI-WASM-WORKERS-001 — Schedule shared-memory work across Web Workers
- status: todo
- priority: architecture
- milestone: M-core-execution
- outcome: Thread-capable browsers can execute Moirai work through a bounded worker pool without spinning the main thread.
- acceptance: Workers share the admitted memory instance, route bounded messages or ring entries, steal through atomics, and park with `Atomics.wait`/`notify`; unsupported browsers return a typed capability result.
- scope: Web Worker lifecycle, shared-memory routing, work stealing, parking, and capability detection.
- next step: Define the worker lifecycle and memory-isolation threat model before selecting the routing representation.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-FUTURES-IO-COMPAT-001"></a>
## MOI-FUTURES-IO-COMPAT-001 — Move the futures-io wrappers into the compatibility module
- status: todo
- priority: architecture
- milestone: M-async-io-transport
- outcome: One wrapper family in `moirai-async::io::compat` maps Moirai I/O onto every external trait family it supports, and `moirai-tls` consumes it.
- acceptance: `ToFuturesIo` and `ToMoiraiIo` leave `moirai-tls`; the replacement wrappers are `#[repr(transparent)]`, forward the task context and vectored writes, and pass the duplex, wake, EOF, and large-transfer scenarios of `io/compat/tests.rs`.
- scope: `moirai-tls/src/lib.rs` wrappers, `moirai-async/src/io/compat.rs`, their callers, and ADR 0006.
- next step: Search the workspace for `ToFuturesIo` and `ToMoiraiIo` callers, then extend the compat module with the `futures::io` traits and migrate them.
- basis: `3bf9b07808fd2326033f308eacbc2aa4325013fc`

<a id="MOI-WASM-PROMISE-FUTURE-001"></a>
## MOI-WASM-PROMISE-FUTURE-001 — Own Promise-to-Future callback lifetimes
- status: todo
- priority: architecture
- milestone: M-async-io-transport
- outcome: JavaScript Promise resolution and rejection wake one Rust future without leaking callbacks or requiring cross-thread access.
- acceptance: An owned callback guard maps resolve/reject to typed results, updates the current waker, cancels registration on drop, and passes lifecycle tests for completion, rejection, replacement, and cancellation.
- scope: general Promise bridge and bounded event handoff; existing specialized file/timer ownership remains authoritative.
- next step: Extract the shared ownership contract from delivered specialized bridges and prove one general state machine.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-WEBVIEW-RUST-PROVIDER-ACCEPTANCE-2026-10-13"></a>
## MOI-WEBVIEW-RUST-PROVIDER-ACCEPTANCE-2026-10-13 — Verify Rust page provider ownership [minor]
- status: todo
- priority: verification
- milestone: M-core-execution
- outcome: Real Windows WebView2 verifies Rust bridge handoff, repeated native DENY and fresh-profile browser exit with current source/tool/module identities.
- acceptance: Run the installed repeated geolocation-denial fixture and actual four-kind consumer probe; every required native event and browser outcome matches. Verify explicit listener close, bounded reported guards through stop/remount, actual BrowserProcessExited and subsequent caller-owned directory cleanup. All diagnostics, existing tests and minor API compatibility gates pass within their unchanged budgets.
- scope: `moirai-pal/src/wasm/{webview,permissions}`, DOM event owners, native WebView2 profile/shutdown and the consumer-owned Rust page. Metis owns schema, page policy and packaged acceptance.
- next step: Publish the provider through normal CI, execute the installed Windows fixture, then exercise the consumer's native readiness/stop protocol and capture its exact resource counts. A compiled fixture or screenshot alone does not establish runtime acceptance.
- basis: `17a680ffebda11bf344fd06138dfc58c73c745be`; [ADR 0077](docs/adr/0077-rust-page-provider-ownership.md).

<a id="MOI-BROWSER-PERMISSION-POSITIVE-2026-10-13"></a>
## MOI-BROWSER-PERMISSION-POSITIVE-2026-10-13 — Verify consented grant and late cleanup [minor]
- status: blocked
- priority: verification
- outcome: Real granted geolocation, camera/microphone leases and notification values establish browser resource cleanup under completion, deadline and cancellation.
- acceptance: With explicit human consent and real available devices, verify every acquired track stops on lease drop and late success after cancellation; verify watch clearing, stale callback isolation, notification granted/denied/default and busy admission while native settlement remains outstanding. Record actual guard/dispatcher counts through stop/remount; an ignored geolocation completion can remain retained until document destruction.
- scope: browser permission provider and a consented ordinary-browser fixture. Native WebView2 DENY policy remains the host contract.
- blocker: No consented positive-grant/device fixture is recorded. Permission requests and native outcome data cannot be replaced with fabricated grants or emulated streams.
- re-open trigger: Explicit fixture consent and real device/browser prerequisites are available.
- basis: `17a680ffebda11bf344fd06138dfc58c73c745be`; [ADR 0077](docs/adr/0077-rust-page-provider-ownership.md).

<a id="ISSUE-004"></a>
## ISSUE-004 — Establish current workspace coverage
- status: todo
- priority: verification
- milestone: M-core-execution
- outcome: Moirai has a current line and branch coverage baseline and a concrete list of uncovered correctness-critical paths.
- acceptance: The configured coverage tool runs against the complete workspace test suite, records reproducible results, and files value-semantic tests for uncovered public behavior without weakening assertions.
- scope: coverage configuration, test execution, and resulting focused tests; no percentage claim from the historical 50-test snapshot.
- next step: Run the current workspace coverage command at this basis and classify uncovered changed and trust-boundary code.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-SPAWN-GLOBAL-MUTEX-REVIEW-001"></a>
## MOI-SPAWN-GLOBAL-MUTEX-REVIEW-001 — Review the sharded task registry decision
- status: todo
- priority: verification
- milestone: M-core-execution
- outcome: The landed sharded registry either receives independent approval or a forward correction grounded in ADR 0005 and current measurements.
- acceptance: Review the single-producer regression, multi-producer scaling, task-scheduling control, dense-block ownership, and cleanup interaction; record a verdict and any correction in ADR 0005.
- scope: landed registry architecture and its measurement contracts; no history rewrite.
- next step: The independent review is done and its registry defects are fixed (ADR 0005 carries the corrected bounds); repeat the decisive benchmark rows on deterministic counters or an isolated-core run, then record the verdict in ADR 0005.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-ASYNC-IO-COMPARISON-001"></a>
## MOI-ASYNC-IO-COMPARISON-001 — Verify native and compatibility I/O contracts
- status: todo
- priority: verification
- milestone: M-async-io-transport
- outcome: Native Moirai I/O and its Tokio compatibility path have measured, value-equivalent loopback behavior.
- acceptance: Heavy cancellation tests preserve buffers, registration/unregistration overhead is recorded, and native versus compat TCP loopback rows compare throughput and latency with identical values.
- scope: async network/file integration tests and `async_tcp_comparison` benchmark rows.
- next step: Build the cancellation fixture and paired loopback harness against the current reactor backends.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="ISSUE-005"></a>
## ISSUE-005 — Audit unsafe-code obligations
- status: todo
- priority: verification
- milestone: M-core-execution
- outcome: Every reachable unsafe operation has a current safety argument and the strongest executable check its platform permits.
- acceptance: Inventory current unsafe sites, verify each `SAFETY` obligation against its safe caller boundary, run Miri where supported and sanitizer/targeted substitutes elsewhere, and file any unsound or uncovered unit as a correctness item.
- scope: workspace unsafe blocks, public safe wrappers, FFI/platform boundaries, and their memory-safety tests; no stale 2024 count as a completion claim.
- next step: Generate a current revision inventory by crate and rank reachable trust-boundary sites before reviewing implementations. The scheduled `Miri` job in `.github/workflows/rust-ci.yml` holds the interpreted set; triage what its comment lists as excluded (moirai-async worker threads that outlive their owner, tests over the per-test budget, `shm_open` and socket tests Miri cannot run) and widen the set as sites are reviewed; a survey of utils, sync, scheduler, async and core found no further undefined behavior.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-WASM-HEADLESS-TRACE-001"></a>
## MOI-WASM-HEADLESS-TRACE-001 — Run browser lifecycle verification
- status: todo
- priority: verification
- milestone: M-core-execution
- outcome: Real browser engines verify WASM task progress, wakeups, and callback cleanup.
- acceptance: Headless Chromium, Firefox, and WebKit traces exercise cooperative tasks, event wakes, cancellation, and callback release with bounded artifacts and value assertions.
- scope: WASM browser tests and hosted trace workflow; no synthetic-only success claim.
- next step: Define one engine-neutral lifecycle scenario and make its trace deterministic before adding the matrix.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-PAL-WINDOWS-CI-001"></a>
## MOI-PAL-WINDOWS-CI-001 — Run every Windows PAL test in CI
- status: todo
- priority: verification
- milestone: M-core-execution
- outcome: Hosted Windows CI runs the native dialog and window tests, including pointer capture and capture re-entry, not only the WebView2 and window-destruction subsets.
- acceptance: The `windows-webview.yml` nextest step (or a sibling Windows job) selects `windows::dialog::` and every `windows::window::tests::` module; `--no-tests fail` holds; a deliberately broken capture assertion fails the job on a draft PR before the fix lands; the job stays within its timeout.
- scope: `.github/workflows/windows-webview.yml`, `.config/nextest.toml` test groups for tests that own a message pump or COM apartment; no test deletion or retry.
- next step: Run `cargo nextest list -p moirai-pal --lib --no-default-features --features webview2` on Windows, classify which modules need a serial test group, then widen the filter expression.
- basis: `8050f2daa157911d3f033e2e0f35403bd741b8cd`

<a id="MOI-JUDGE-STATUS-WORKFLOW-001"></a>
## MOI-JUDGE-STATUS-WORKFLOW-001 — Post and require the `judge` status on main
- status: todo
- priority: verification
- milestone: M-core-execution
- needs: none
- outcome: A pull-request workflow posts `judge` `success` on each head of a ready unlabeled PR, nothing on a `judge-required` head, and `failure` on the `labeled` event, reading the live labels from the API; then `judge` is a required context on `main` beside `Verification result`, so a head with a failed or missing `judge` status cannot merge.
- acceptance: The workflow file exists and `actionlint` passes; a ready unlabeled test PR shows a `judge` `success` status created by `github-actions[bot]`; a `judge-required` one shows none until a non-bot status is posted; `gh api repos/ryancinsight/Moirai/branches/main/protection` lists both `Verification result` and `judge` in `required_status_checks.contexts`.
- evidence: PR 700 merged over a `judge` `failure` at its head because the protection requires only `Verification result` (gap entry MOI-GAP-JUDGE-MERGE-GATE-001 in PR 729; judge verdict on PR 729, F3). At the basis `git grep -n -i judge -- .github` prints nothing while `git grep -n -i verification -- .github` prints 3 lines; atlas at `6dfc83ae596a686d84f899b8a6b79cb6e6ace067` ships no such workflow: `git grep -n -i -E 'context=judge|statuses: write|judge-required' -- .github scripts docs` prints nothing there while `git grep -n -E 'github.event.pull_request' -- .github` prints 17 lines.
- scope: a new PR workflow under `.github/workflows/` with `statuses: write`; the `main` branch protection.
- next step: Write the workflow and land it first, since a required context nobody posts blocks every PR. Requiring `judge` is a repository merge-mechanics setting: ask the user, then set it and confirm with the protection API.
- basis: `69bb7822f5d96e0e9f83f00c99a6b820b5539d15`

<a id="MOI-WEBVIEW-WAIT-SPAN-TESTS-001"></a>
## MOI-WEBVIEW-WAIT-SPAN-TESTS-001 — Test every WebView2 wait span, the report drop and the wait bound
- status: todo
- priority: verification
- milestone: M-core-execution
- needs: none
- outcome: Each production `webview2_*_wait` span, the `Report` drop path and the `wait_for` timeout bound have a test that fails when the behavior changes.
- acceptance: (a) A test asserts a recorded `(host, name)` for `webview2_frame_wait`, `webview2_diagnostics_wait` and `webview2_browser_exit_wait`, the three of seven span names that `tests/preview.rs:47-51` does not list, and `each_host_thread_records_under_its_own_index` asserts distinct per-host names; renaming any `debug_span!` fails a test. (b) A test drops `Report` over two waits and asserts the `slowest wait` events by value, and once inside `catch_unwind` of a panicking host under a writer that returns `Err`, asserting the original panic propagates (`.log_internal_errors(false)`, `waits.rs:168`). (c) `wait_for` rejects a timeout above `MAX_WEBVIEW_WAIT_MILLISECONDS` including a sub-millisecond excess, or `pump.rs:26-27` says whole milliseconds (`as_millis`, `pump.rs:36`).
- scope: `moirai-pal/src/windows/webview/pump.rs`, `moirai-pal/src/windows/webview/tests/preview/waits.rs`, `moirai-pal/src/windows/webview/tests/preview.rs`.
- next step: Add the three recorded-name assertions to the installed-runtime tests, then the `Report` drop and unwind unit tests.
- links: judge verdicts on PR 726 (N2, N4, N6, F8).
- basis: `69bb7822f5d96e0e9f83f00c99a6b820b5539d15`

<a id="MOI-PAL-TEST-SUBSCRIBER-001"></a>
## MOI-PAL-TEST-SUBSCRIBER-001 — Replace the per-module test subscribers with one helper
- status: todo
- priority: tightening
- milestone: M-core-execution
- needs: none
- outcome: One test-subscriber helper in `moirai-pal` replaces the module-local copies; clock-less output is a parameter, not a second copy.
- acceptance: `git grep -c with_test_writer -- moirai-pal/src` totals 1 (16 sites in 15 files at the basis); `git grep -c 'subscriber::set_default' -- moirai-pal/src` lists only the helper, the wait recorder and `stderr_trace` in `waits.rs`, and `wasm/test_trace.rs` (20 sites in 18 files at the basis); the test names `cargo nextest list -p moirai-pal` prints are unchanged.
- scope: the 15 files `git grep -l with_test_writer -- moirai-pal/src` lists and `moirai-pal/src/wasm/test_trace.rs`.
- next step: Enumerate the variation the 16 sites show (`without_time` at 5, span events on or off), put one parameterized `#[cfg(test)]` helper at the crate root and migrate every caller in one commit.
- links: judge verdict on PR 726 (F3).
- basis: `69bb7822f5d96e0e9f83f00c99a6b820b5539d15`

<a id="MOI-PR-METADATA-GATE-001"></a>
## MOI-PR-METADATA-GATE-001 — Enforce the PR title and Item rules in the aggregate
- status: todo
- priority: verification
- milestone: M-core-execution
- needs: none
- outcome: A PR whose title is not a conventional-commit subject of at most 50 characters, or whose body names neither an `Item:` line nor an ADR (sweep, compaction and dependency-update PRs excepted), fails `Verification result`.
- acceptance: A test PR with a 51-character title and one whose body has neither reference each fail the aggregate with a message naming the rule; a conforming PR passes; a draft stays refused.
- evidence: `Verification result` (`rust-ci.yml`, job `verdict`) reads only the draft flag and `needs`. At the basis `git grep -n -i -E 'pull_request\.(title|body)|commitlint|Item:' -- .github` prints nothing while `git grep -n 'github.event.pull_request' -- .github` prints 13 lines; atlas at `6dfc83ae596a686d84f899b8a6b79cb6e6ace067` has no title check either (same pattern over `.github scripts`).
- scope: `.github/workflows/rust-ci.yml` (job `verdict`) or a new PR workflow it needs.
- next step: Fix the exemption signal for the three excepted PR classes before enforcing, then add the check.
- links: judge verdict on PR 729 (F4).
- basis: `69bb7822f5d96e0e9f83f00c99a6b820b5539d15`

<a id="MOI-BOARD-PRIORITY-ORDER-001"></a>
## MOI-BOARD-PRIORITY-ORDER-001 — Order the backlog blocks by priority
- status: todo
- priority: verification
- milestone: M-core-execution
- needs: none
- outcome: The item blocks of `backlog.md` run in the order the file header states: `correctness`, `architecture`, `verification`, `tightening`, then `feature`.
- acceptance: `awk '/^- priority: [a-z]+$/{r=($3=="correctness")?0:($3=="architecture")?1:($3=="verification")?2:($3=="tightening")?3:4; if(r<p)d++; p=r} END{print d+0}' backlog.md` prints 0 (6 at the basis); a block stays after its `needs:` prerequisites.
- scope: `backlog.md` block order only; no block text changes.
- next step: Move whole blocks in one commit built from the fetched default branch, after any open PR that edits the same blocks has landed; compare the sorted block texts before and after to show none changed.
- links: judge verdict on PR 727 (F12).
- basis: `69bb7822f5d96e0e9f83f00c99a6b820b5539d15`

<a id="MOI-FORK-JOIN-LATENCY-2026-09-10"></a>
## MOI-FORK-JOIN-LATENCY-2026-09-10 — Attribute fork-join tail latency
- status: todo
- priority: tightening
- milestone: M-core-execution
- outcome: Repeated small fork-joins use the available worker pool without the observed wake-latency tail.
- acceptance: The 64-task microbenchmark separates wake order, per-worker wake cost, caller participation, and spin policy across worker counts; the selected correction reduces p90 without regressing hot-path controls or idle CPU budget.
- scope: indexed fork-join submission, worker park/wake, caller-help policy, Criterion attribution, and consumer-sized confirmation.
- next step: Resume the retained latency distribution after the crash cause is fixed, then file or implement the measured wake-path correction.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="ISSUE-010"></a>
## ISSUE-010 — Attribute remaining result-handoff variance
- status: todo
- priority: tightening
- milestone: M-core-execution
- outcome: Remaining public result-slot and oversized-job variance is attributed to one measured production mechanism.
- acceptance: Profiles and paired Criterion runs identify the mechanism, preserve lifecycle timing and panic semantics, add no lock to the handoff path, and show no significant control-path regression.
- scope: result-slot ownership, oversized inline trampoline handoff, scheduler submission, and existing diagnostic benchmarks; rejected raw-pointer, relaxed-timestamp, and catch-removal variants remain rejected.
- next step: Profile the current public ready and oversized paths and compare code generation at the first divergent component.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-PARKED-POOL-FIRST-REGION-2026-09-17"></a>
## MOI-PARKED-POOL-FIRST-REGION-2026-09-17 — Reduce first-region wake cost
- status: todo
- priority: tightening
- milestone: M-core-execution
- outcome: A data-parallel region submitted after workers park approaches the hot-pool execution time.
- acceptance: A paired benchmark records hot and 0.1/1/10-millisecond idle-gap distributions, attributes wake order, per-worker wake cost, and caller work, and accepts a change only when latency falls within a stated idle-CPU budget.
- scope: worker park/wake policy, submitter participation, and the fixed-region benchmark; no unmeasured spin extension.
- next step: Add the gap-controlled benchmark and profile the first region before selecting batched wake, prewake, or spin-policy changes.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-NUMA-STEAL-BENCH-001"></a>
## MOI-NUMA-STEAL-BENCH-001 — Measure two-pass stealing on multi-socket hardware
- status: blocked
- priority: tightening
- outcome: The two-pass NUMA victim scan has a measured latency and throughput cost under a real cross-node steal storm.
- acceptance: Criterion compares one-pass and current two-pass scans on the same multi-socket host, including no-local-work fallback, and any correction preserves the deterministic cross-node completion test.
- scope: `steal_job` scan order and Criterion instrumentation; no inference from single-socket CI.
- blocker: No multi-socket NUMA machine is available in CI or the current development host.
- re-open trigger: A controlled multi-socket host is available for the paired run.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-WASM-DOM-FILE-SAFARI-2026-09-14"></a>
## MOI-WASM-DOM-FILE-SAFARI-2026-09-14 — Recover bounded WebKit file reads
- status: blocked
- priority: feature
- outcome: WebKit reads a user-selected file through the bounded browser provider or returns a typed provider error.
- acceptance: A hosted Safari chooser run reads the selected public fixture within the 1 MiB per-read bound, releases stream state on completion/error, and leaves native/WASM warning-denied gates green.
- scope: Safari/WebKit selected-file authorization and the existing bounded reader; no DICOM parser, native permission policy, or unbounded fallback.
- blocker: SafariDriver/WebKit denies the first read for a chooser-selected file across `File.arrayBuffer`, sliced `Blob.arrayBuffer`, `FileReader`, and object-URL streaming.
- re-open trigger: The same selected file becomes readable under a corrected browser or runner authorization path.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-TOKIO-PARITY-STAGE-B1"></a>
## MOI-TOKIO-PARITY-STAGE-B1 — Complete bounded local-runtime parity
- status: todo
- priority: feature
- milestone: M-async-io-transport
- outcome: Moirai supplies select ergonomics, IPv6, and a graceful shutdown signal through its bounded local runtime surface.
- acceptance: Each surface has typed cancellation and backpressure behavior, value-semantic integration tests, and comparison coverage against the corresponding Tokio behavior; HTTP/2 remains excluded without a named consumer.
- scope: select-equivalent ergonomics, IPv6 network paths, shutdown signaling, tests, and documentation.
- next step: Specify and deliver the select-equivalent contract as the first independent slice.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-WASM-NETWORK-FACADE-001"></a>
## MOI-WASM-NETWORK-FACADE-001 — Add bounded browser fetch and networking
- status: todo
- priority: feature
- milestone: M-async-io-transport
- outcome: WASM applications use owned bounded browser fetch and network operations without compiling the native network module.
- acceptance: Request, response, redirect, body, queue, deadline, and callback lifetimes are bounded; failures are typed; browser traces cover success, rejection, cancellation, and exhaustion.
- scope: browser `fetch` and the general browser network facade; native sockets remain separate.
- next step: Define the resource-limit and cancellation contract against ADR 0007 before exposing the first fetch operation.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-FS-TEST-GUARD-REMOVAL-001"></a>
## MOI-FS-TEST-GUARD-REMOVAL-001 — Report failed removal in fs test guards
- status: todo
- priority: verification
- milestone: M-core-execution
- outcome: A fs test whose temp entry is missing or cannot be removed when its guard drops fails, and the three guards share one meaning of `create`.
- acceptance: With a guard `Drop` body emptied, the fs suites of `moirai-async` and `moirai-pal` fail (at 8a5bf9c1dfa71da2eaba990412f9ea6e0b63cef8 32 fs tests pass and leave 22 entries); `TestEntry::create`, `TestFile::create` and `TestRoot::create` all create the entry, or the two that only compute a path are renamed; `cargo nextest list --workspace --all-features --run-ignored all` is unchanged (1524 tests).
- scope: `moirai-async/src/fs/tests.rs`, `moirai-pal/src/fs/tests.rs`, `moirai-pal/src/fs/confined/tests.rs`.
- next step: Decide how a guard reports a failed removal without a double panic (assert only when `std::thread::panicking()` is false), then delete the unreachable clearing in `test_path`, merge the two `impl TestFile` blocks, and drop the redundant `create_dir_all` in the FIFO test.
- links: judge verdicts on PR 680 (F1, F3) and PR 679.
- basis: `8a5bf9c1dfa71da2eaba990412f9ea6e0b63cef8`

<a id="MOI-QUIC-EVIDENCE-LOCK-001"></a>
## MOI-QUIC-EVIDENCE-LOCK-001 — Bind the archived QUIC recovery evidence to the canonical lock
- status: todo
- priority: verification
- milestone: M-core-execution
- needs: none
- outcome: The `quic-recovery.yml` step "Verify the original domain and archived numerical evidence" passes at a default-branch commit, so PRs that touch QUIC paths stop inheriting its red.
- acceptance: `python3 -m unittest discover -s scripts/tests -p 'test_quic*.py'` passes on an export of the head commit (PR 695 run 43780503003 reports FAILED failures=1, errors=2), and either every archived `canonical_lock_sha256` equals the SHA-256 of that commit's `Cargo.lock` or the evidence is regenerated by `scripts/quic_observation/prepare.py` and the recorded measurement procedure.
- scope: `scripts/tests/test_quic_handshake.py`, `scripts/tests/test_quic_observation.py`, `scripts/quic_observation/prepare.py`, `moirai-quic/measurements/recovery/2026-10-14/`, `.github/workflows/quic-recovery.yml`.
- evidence: run 43780503003 job 133016363188, step 6: `candidate-native-contracts` records `canonical_lock_sha256` sha256-c9b6224a40f6...3ee7 against current sha256-4426f89823d0...45ab, and two `ValueError: diagnostic build provenance differs from verified source/patch/locks` (`prepare.py:95`; `state-native.json.gz` also records `diagnostic_lock_sha256` sha256-6bca56bf). None of the 67 listed runs of this workflow concluded success, and it runs only on PRs touching its path filter, which omits `Cargo.lock`.
- cause: the archive was written against the PR 618 branch lock sha256-c9b6224a (`140d60cf`, `03dcc4a3`, `9e650516`). Merging main into that branch (`947d1520`, 2026-10-22) changed the lock to sha256-2b7f3fb4; PR 618 landed as `4bb7d2d4` (lock sha256-41410cb4) and `3e709661` later moved it to sha256-4426f898. No first-parent main Cargo.lock state among the 60 most recent equals sha256-c9b6224a. The archive never matched main's lock.
- engine inputs match: quinn-proto 0.11.19 with registry checksum sha256-0e750cca..., and `engine.patch` SHA-256 sha256-d43d5cba..., equal the archive. Only lock content differs (320 insertions and 126 deletions since `140d60cf`, including rustls-webpki 0.103.15 to 0.103.16).
- next step: decide rebind or regenerate. Rebinding the hash asserts measurements that the current lock never produced, so regenerate against the current lock unless the measured dependency set is shown identical. Also decide whether `Cargo.lock` joins the workflow `paths:` so the next lock change fails in its own PR.
- links: [PR 695](https://github.com/ryancinsight/Moirai/pull/695) judge finding F9 (merged as `5422acbc` with this step red).
- basis: `5422acbcf7ff66a8efbaee6258aa9fbad8e24a4e`

<a id="MOI-QUIC-PACKAGE-FIXTURES-001"></a>
## MOI-QUIC-PACKAGE-FIXTURES-001 — Package the test fixtures and README links of the QUIC-reading crates
- status: todo
- priority: verification
- milestone: M-async-io-transport
- needs: none
- outcome: The published archives of `moirai-quic`, `moirai-crypto`, `moirai-tls` and `moirai-runtime` compile their tests and load example, and the `moirai-quic` README links resolve from the packaged file.
- acceptance: For each of the four packages, `cargo package -p <name>` followed by `cargo test --no-run --all-targets` in the unpacked `.crate` succeeds, and every relative `moirai-quic/README.md` link resolves from the archive root. Each package holds its own copy of the PEM files it reads, and a test fails when the copies of one file differ byte for byte.
- evidence: `include_bytes!` reads the four root `tests/fixtures/*.pem` at 15 sites, outside each reading package (the root `tests/` is the unpublished `moirai-tests` package): `moirai-quic` 6 (`src/tests.rs:34,42`, `src/tests/rejection.rs:287,295`, `examples/load/material.rs:10,11`), `moirai-crypto/tests` 3 (`handshake.rs:46,56`, `rsa_signatures.rs:22`), `moirai-tls/tests/handshake.rs` 4 (93,94,102,103), `moirai/tests/quic.rs` 2 (17,25). `cargo package --list` of `moirai-quic`, `moirai-crypto` and `moirai-tls` lists no fixture file; none carries `publish = false`. `moirai-quic/README.md` links `../docs/adr/` at lines 13, 28, 30, 59 and 78.
- scope: `moirai-quic/{Cargo.toml,README.md,src/tests.rs,src/tests/rejection.rs,examples/load/material.rs}`, the readers in `moirai-crypto/tests`, `moirai-tls/tests` and `moirai/tests`, root `tests/fixtures/`, and a new `package` job in `.github/workflows/rust-ci.yml` added to the `verdict` job's `needs`.
- next step: copy, not move, the PEM files and `tests/fixtures/README.md` into each reading crate's `tests/fixtures/` (a package cannot read another package's files), repoint the 15 sites, delete root `tests/fixtures/` once no reader remains, replace the ADR links with absolute repository URLs, and run the package step once on a branch: hypothesis that the unpacked manifest resolves `moirai-core`, `moirai-async` and `moirai-pal` from the registry, so the result depends on published versions.
- basis: `fb4abdb6af2f0b23e26948178dab1bd301f8fe84`

<a id="MOI-RUNTIME-CORE-DUMP-001"></a>
## MOI-RUNTIME-CORE-DUMP-001 — Remove tracked core dumps from the tree
- status: todo
- priority: verification
- milestone: M-core-execution
- needs: none
- outcome: No tracked blob is an ELF core file, and none ships in the moirai-runtime package.
- acceptance: `git grep -l -a -P '\A\x7fELF[\x01\x02][\x01\x02]\x01.{9}\x04\x00' HEAD` (ELF header with `e_type` 4) prints no path, where it prints `moirai/core` and `tests/core` at the basis, so a rename cannot pass; `cargo package --list -p moirai-runtime` prints no line equal to `core`; the same scan runs as a CI or pre-push check that exits nonzero when it prints a path. History rewrite is out of scope.
- evidence: `git ls-tree -r -l origin/main` lists `moirai/core` (64,999,424 bytes) in the `moirai-runtime` package root (`moirai/Cargo.toml` has no `exclude` or `include`) and `tests/core` (46,981,120 bytes) in the unpublished `moirai-tests` package. Both blobs name a Linux x86_64 test binary under `/workspace/target/debug/deps/` (`moirai-1b224ad9dc0761da`, `moirai_tests-c9f92fd14581cdaf`) and were added in `fdfeeed6` and `cadd5dd8`. `.gitignore` has no core-dump entry.
- scope: `moirai/core`, `tests/core`, `.gitignore`, and the CI or pre-push check that runs the scan.
- next step: `git rm moirai/core tests/core`; add the scan as the regrowth guard. Hypothesis: the kernel `core_pattern` default wrote each file into the working directory of a crashed test binary; confirm on the build image. An ignore entry is optional, anchored (`/moirai/core`, `/tests/core`) with its writer annotation; a bare `core` would ignore new files named `core` under `moirai-pal/src/reactor/core/` (gitignore never affects the 6 files already tracked there).
- basis: `fb4abdb6af2f0b23e26948178dab1bd301f8fe84`

<a id="MOI-QUIC-HISTORICAL-AUDIT-001"></a>
## MOI-QUIC-HISTORICAL-AUDIT-001 — Audit archived QUIC measurements as historical-only provenance
- status: blocked
- priority: verification
- milestone: M-core-execution
- needs: MOI-QUIC-EVIDENCE-LOCK-001
- outcome: An explicit utility verifies the immutable provenance of the archived QUIC measurement directories against the inputs recorded at their own revision and labels them historical-only, so an archive recorded against a superseded lock is never read as current-lock evidence.
- acceptance: `scripts/tests/test_quic_historical.py` holds these tests, each of which fails when the behavior it asserts is removed: `test_full_historical_report_reproduces_without_mutation` (returns `{'scope': 'historical-only', 'revision': <sha>}` and leaves the report unchanged); `test_existing_archive_still_rejected_by_current_verifier` (`verify_provenance` raises `ValueError` matching `diagnostic build provenance differs` for `state-native.json.gz`); `test_moving_refs_and_revision_expressions_are_rejected` (`main`, `HEAD`, an 8-character prefix, `<sha>^`, `--help` and `None` each raise `ValueError`); `test_replacement_missing_and_noncommit_objects` (a `git replace` is ignored, a missing object raises `CalledProcessError`, a blob raises `not a commit`); `test_changed_observed_file_and_extra_field_fail` (a changed observed-file hash or an added `current_qualification` field raises `ValueError`); `test_changed_lock_patch_inventory_and_observed_hashes_fail` (each of five hash fields set to zeros, and a removed `changed_files` entry, raises `ValueError`). All of them, and `test_superseded_revision_inputs_verify_and_current_hashes_fail` below, pass on an export of the head commit (no `.git`) under `python3 -m unittest discover -s scripts/tests -p 'test_quic_historical.py' -k reproduces -k rejected -k replacement -k changed -k superseded`.
- acceptance, tests to add (absent from `refs/pull/703/head`): `test_superseded_revision_inputs_verify_and_current_hashes_fail` commits a fixture revision whose `Cargo.lock` differs from the working tree's and asserts the verifier returns `{'scope': 'historical-only', 'revision': <rev>}` for a report hashed at that revision and raises `ValueError` matching `historical input hash mismatch: Cargo.lock` for the working tree's hashes; `test_real_archive_verifies_at_recorded_revision` runs the utility on the `engine_observation` of `moirai-quic/measurements/recovery/2026-10-14/state-native.json.gz` and asserts by value `{'scope': 'historical-only', 'revision': <rev>}` for `03dcc4a31d7f793e4cb1cb7d96ff74f47033946c` and for `9e6505163b700c31cc89cfa5f0bebdd1846e6eba` (identical `Cargo.lock` and `engine.patch` blobs, different commit), and a `ValueError` matching `historical input hash mismatch: Cargo.lock` for the checked-out `HEAD`. That test needs commits an export and a depth-1 checkout lack, so it runs and passes in the `compare` job of `.github/workflows/quic-recovery.yml` with `actions/checkout` set to `fetch-depth: 0` (the job's `unittest discover -p 'test_quic*.py'` step), whose `unittest` summary reports no `skipped`, and `git grep -n -i skip -- scripts/tests/test_quic_historical.py` prints nothing. In that job each of these must fail the suite: replacing `historical.py:37` (`git('show', f'{revision}:{path}')`) with a working-tree read, removing the hash comparison (lines 38-39), and returning a constant revision.
- scope: `scripts/quic_observation/historical.py`, `scripts/tests/test_quic_historical.py`, and `.github/workflows/quic-recovery.yml` (`actions/checkout` at lines 34-37 sets no `fetch-depth` and its `action.yml:74-76` defaults it to 1, so `03dcc4a3`, an ancestor of `fb4abdb6`, is absent from the checkout, derived).
- evidence: `git diff --stat origin/main refs/pull/703/head -- scripts/` shows 2 files, +150 (`historical.py` +44, `test_quic_historical.py` +106); `prepare.py` and `test_quic_observation_portability.py` are already on main (`0e639654`). On an export of `scripts/`, `Cargo.lock` and `moirai-quic/measurements` of `refs/pull/703/head` the six tests pass, and with `verify_historical_provenance` replaced by `return {'scope': 'historical-only', 'revision': revision}` the changed-lock, changed-observed-file, moving-refs and replacement tests fail while the full-report and existing-archive tests pass. `verify_historical_provenance(<engine_observation of the archive>, <a full clone>, <revision>)` from `refs/pull/703/head` (`historical.py:37-39`) returns `{'scope': 'historical-only', 'revision': <revision>}` for `03dcc4a3` and for `9e650516`, which hold identical `Cargo.lock` and `engine.patch` blobs, and raises `historical input hash mismatch: Cargo.lock` for `fb4abdb6`. `quic-recovery.yml:45` is the only runner of `test_quic*.py` under `.github` and `.githooks`.
- blocker: the QUIC recovery workflow fails on the current lock independent of this change; [MOI-QUIC-EVIDENCE-LOCK-001](#MOI-QUIC-EVIDENCE-LOCK-001) records the `canonical_lock_sha256` mismatch.
- re-open trigger: MOI-QUIC-EVIDENCE-LOCK-001 lands.
- next step: restore the branch from the pull ref (`git fetch origin pull/703/head:<branch>`), merge main into it, add the two tests and the `fetch-depth: 0` checkout, and rerun the `test_quic*.py` suite and the recovery checks.
- basis: `fb4abdb6af2f0b23e26948178dab1bd301f8fe84`

<a id="MOI-ASYNC-SEEK-001"></a>
## MOI-ASYNC-SEEK-001 — Define an async seek trait for Moirai I/O
- status: todo
- priority: feature
- milestone: M-async-io-transport
- outcome: Moirai files and in-memory cursors seek through a poll-based trait that maps onto `tokio::io::AsyncSeek`.
- acceptance: `File::seek` is reachable through the trait, the compat wrappers forward `start_seek` and `poll_complete` with the caller's task context, and a seek in flight survives a re-poll under a new waker.
- scope: `moirai-async::io::traits`, `moirai-async::fs::file`, `io/compat.rs`, and ADR 0006.
- next step: Specify the trait against `File`'s pooled seek request and decide whether a seek in flight is cancellation-safe before writing it.
- basis: `3bf9b07808fd2326033f308eacbc2aa4325013fc`

<a id="MOI-SPIN-BACKOFF-CONSOLIDATION-2026-09-30"></a>
## MOI-SPIN-BACKOFF-CONSOLIDATION-2026-09-30 — One spin-then-yield schedule for bounded-wait sites
- status: todo
- priority: tightening
- milestone: M-core-execution
- outcome: The spin-then-yield schedules in `LockFreeQueue::enqueue`, the SPSC ring (`SPSC_BLOCK_SPINS`), the MPMC channel (`backoff_step`), `ResultCell`, and the task-handle waits derive from one schedule type whose spin budget is a const-generic parameter, and the change deletes more lines than it adds.
- acceptance: A unit test pins the schedule (spin rounds, then yield); each site keeps its measured budget; a benchmark smoke run passes; `git grep -n "spin_loop()"` lists only the shared schedule and true lock spins. `enqueue` keeps its current retry count unless a benchmark shows otherwise.
- scope: `moirai-utils/src/queue/ring.rs`, `moirai-utils/src/result_cell.rs`, `moirai-core/src/channel/{spsc/ring.rs,mpmc/block.rs,mpmc/channel.rs}`, `moirai-core/src/task/handle.rs`; not the Chase-Lev `ContentionWait` or the NUMA backoff without a recorded reason.
- next step: Classify the `spin_loop()` sites as bounded-wait schedules or lock spins, then design the one type; the prior attempt is the head of rescue PR 506 (a public `moirai_utils::backoff` with 7 sites, no tests, changed `enqueue` retry behavior, missed the reopen wait) and is a source, not a base.
- basis: `0a9a2a8ffcec2c07101a5475c77fa976c52ede17`

<a id="MOI-REL-061"></a>
## MOI-REL-061 — Publish reusable Rust crates
- status: blocked
- priority: feature
- outcome: Every reusable workspace crate resolves from crates.io and subsequent workspace releases use the OIDC workflow.
- acceptance: Clean archives publish in dependency order, sparse-index resolution passes, and each package's trusted publisher matches the release workflow identity.
- scope: crates.io publication and trusted-publisher registration; no local private key or long-lived CI token.
- blocker: Release/deploy authority and registry-side trusted-publisher registration are not part of this documentation compaction.
- re-open trigger: The user explicitly authorizes the workspace release and the crates.io publisher identities are configured.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-REL-060"></a>
## MOI-REL-060 — Enable PyPI trusted publication
- status: blocked
- priority: feature
- outcome: Tagged cross-platform wheels publish through PyPI's trusted publisher after the existing build, install, import, metadata, and attestation gates pass.
- acceptance: PyPI accepts the repository/environment identity and a release uploads the already verified wheel matrix without a long-lived token.
- scope: PyPI account verification and trusted-publisher registration; wheel implementation is already delivered.
- blocker: PyPI rejects the current account email and the account cannot complete publisher verification.
- re-open trigger: The account has a PyPI-accepted verified email and the user explicitly authorizes a release.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-APPLE-HOST-EVIDENCE-001"></a>
## MOI-APPLE-HOST-EVIDENCE-001 — Apple host runtime evidence [minor]
- status: todo
- priority: verification
- milestone: M-core-execution
- outcome: verify the shared WKWebView resource provider on native macOS, iPhone and iPad.
- acceptance: real Rust/WASM input/output, geometry, scene reconnect and bounded shutdown; permission denial is a separate host contract.
- evidence: six resource byte/MIME/traversal/symlink/socket/FIFO/bound tests pass; Metis macOS/device/simulator apps compile; macOS/iPhone/iPad native hosts deliver all seven assets including real WASM.
- [x] [patch] Correct Fetch status in moirai-webview 0.1.1: eight resource/response tests pass; real macOS Greet returns the exact Rust result for normal, Unicode and empty names.
- [x] [patch] Verify moirai-webview 0.1.2's Rust native menus: focused Command-Q and Command-W exit; actual minimize/restore, Zoom geometry and full-screen journeys pass; ADR 0073 records native values and capture limitations.
- [x] [minor] Add moirai-webview 0.2.0's exact host-authorized HTTPS popup handoff: 20 real policy/resource/Foundation tests and 196 compatibility checks pass; native preview records one successful OS handoff for the Metis logo. Canonical consumer and browser visual evidence follow delivery (ADR 0074).
- blocker: remaining native geometry/lifecycle evidence and physical device provisioning/availability are unset.
- scope: moirai-webview; [ADR 0068](docs/adr/0068-apple-webview-host.md).
