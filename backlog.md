# Moirai backlog

This is the single source of truth for active Moirai work. Items are ordered by
priority (`correctness`, `architecture`, `verification`, `tightening`, then
`feature`) and dependency. Hosting records claims and review state; this file
contains only `todo` and `blocked` work. Risks without an active implementation
belong in [gap_audit.md](gap_audit.md).

<a id="MOI-QUIC-PROMOTION-001"></a>
## MOI-QUIC-PROMOTION-001 — Promote verified native QUIC [minor]
- status: todo
- priority: feature
- outcome: Moirai 0.10.0 and moirai-quic 0.1.0 are available at the canonical provider revision; Metis resolves its Git lock from that published source.
- acceptance: ADR 0075, real UDP interoperability/loss/bounds/cancellation tests, clean native diagnostics, compatibility checks and canonical Git lock pass before promotion. Windows/Linux transport evidence remains a platform gate.
- scope: provider publication and consumer revision adoption; no credentials or trust-store configuration.
- current increment: native QUIC, shared readiness, fallible facade spawning, bounded framing and fresh-key Metis sessions are implemented in [PR 618](https://github.com/ryancinsight/Moirai/pull/618). QUIC is required in native facade builds: 187 default tests, the minimal authenticated session, browser facade compilation, strict diagnostics and 892 applicable compatibility checks pass. CI checks default/minimal native sessions. Metis's prior 176-test canonical verification passes; the dependent consumer review adopts the updated provider Git pin. Default-branch promotion remains pending.
- next step: close the measured CUBIC loss-profile limit below and complete required review/checks before protected provider/consumer promotion. Both hosted native platform gates pass; the initial controlled baseline and full loss diagnosis are [archived](moirai-quic/measurements/controlled/2026-10-13/README.md).
- [patch] Windows integration correction: the 03d640b8 native job compiles shared WebView limits without their optional consumer and fails six strict diagnostics. The installed MSVC target reproduces that failure; the module now compiles iff Windows WebView2, browser code or portable tests consume it. Default/minimal native CI retains strict diagnostics.
- [patch] Main integration: the provider rebases onto 7828f5df and retains the complete PR 617/619 implementation. Canonical locks, 397 native tests, the minimal session and all 1,329 workspace tests pass on macOS; nine existing ignored probes remain. The embedded inventory includes the unchanged stream/datagram fixture leaves.
- [major] Compatibility classification: current-main comparison reports the existing async UDP type-owner migration; CHANGELOG.md and ADR 0081 define the pre-1.0 migration. Fresh Windows/Linux results for this reconciled source and protected review remain promotion gates.
- UDP increment: ADR 0081 replaces ignored queue/multicast options with one applied PAL contract used by async UDP and QUIC. All 338 native transport/async tests and strict diagnostics pass locally; kernel-default queues and the full controlled performance contract remain intact.

<a id="MOI-QUIC-RECOVERY-001"></a>
## MOI-QUIC-RECOVERY-001 — Evaluate congestion policy under measured loss [minor]
- status: in progress
- priority: architecture
- outcome: The chosen production recovery policy supports the admitted message/stream domain with measured throughput, latency and competing-flow behavior.
- acceptance: retain the same complete payloads, message counts, deadlines and real byte validation; compare supported congestion policies on matched profiles and document fairness, queue growth and failure bounds before changing defaults. Preserve one canonical transport API and a bounded policy abstraction if variation is required.
- evidence: The 16-stream, 128-message, 65,560-byte 1% loss case exceeds 60 seconds in all three repetitions. Verified progress continues and all namespace UDP error deltas are zero; CUBIC recovery windows contract. No socket-buffer or spawner bypass closes this limit.
- [patch] Diagnostic evidence complete: the [October 14 study](moirai-quic/measurements/recovery/2026-10-14/README.md) independently verifies all 108 original pairs per policy, all 30 competing experiments and actual traces in both peers. CUBIC/NewReno remain 105/108; experimental BBR remains 108/108. Three early server handshakes remain unestablished while clients switch from one Handshake-space PTO to repeated Data-space PTOs. Established-session congestion, pacing and flow-credit notifications coexist. BBR records mixed-policy sampled Jain indices 0.537422–0.582141 and a sampled 133,608,526-byte congestion window; upstream startup-growth PR 2798 remains unmerged. No default change is accepted.
- next step: complete security-patched original-domain revalidation, then correlate per-space handshake ACK/CRYPTO state with PTO selection and measure established-session flight, flow-credit and pacing timelines before selecting a focused recovery candidate. Retain all complete, unsuccessful and canceled studies; preserve original deadlines and the independent full CUBIC promotion gate. No tolerance, window enlargement or unreviewed upstream fix establishes acceptance.
- [patch] Security prerequisite: required CI reports four Rustls 0.23.45 advisories (RUSTSEC-2026-0346/0347/0348/0349). The shared dependency and independently consumed QUIC provider require released 0.23.46; both locks update only Rustls's external identity. Repeat full measurements and native/security checks before accepting the patched stack; retain the preceding source's evidence without mixing cases.

<a id="MOI-WEBVIEW-PREVIEW-CAPTURE-TIMEOUT-2026-09-24"></a>
## MOI-WEBVIEW-PREVIEW-CAPTURE-TIMEOUT-2026-09-24 — Stabilize preview capture under parallel hosts
- status: todo
- priority: correctness
- outcome: `installed_runtime_captures_rendered_preview` completes in every full parallel ignored-test run.
- acceptance: Thirty consecutive `moirai-pal --features webview2 --run-ignored all` runs have zero capture timeouts with the 30-second deadline unchanged.
- scope: `moirai-pal/src/windows/webview/{host/view.rs,tests.rs}`; navigation, visibility, and shared user-data-folder synchronization.
- next step: Determine whether navigation, hidden-document state, or sibling-host profile sharing holds or drops `CapturePreview` completion.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-MPMC-NOTIFY-FENCE-COST-2026-09-24"></a>
## MOI-MPMC-NOTIFY-FENCE-COST-2026-09-24 — Remove unnecessary notifier fences
- status: todo
- priority: correctness
- outcome: Bounded MPMC and hybrid pushes avoid a `SeqCst` fence when no parked receiver depends on it without admitting a lost wakeup.
- acceptance: The two-producer Loom model admits no lost wakeup, pinned-core 4/8-producer rows beat the fenced baseline, and the multiset regression passes 1,000 consecutive runs.
- scope: queue ring, MPMC/hybrid channel notifiers, communication ring buffer, Loom model, and paired benchmark.
- next step: Extend `DrainRing` with a second producer and prove the receiver-head condition before changing production ordering.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-SEC-077"></a>
## MOI-SEC-077 — Remove the unresolved RSA advisory
- status: blocked
- priority: correctness
- outcome: No attacker-observable service depends on `rsa 0.9.10` while `RUSTSEC-2023-0071` remains unresolved upstream.
- acceptance: Replace or remove RSA signing and verification, remove the cargo-deny exception, and pass the locked advisory, license, ban, and source checks.
- scope: RSA-dependent signing/verification and supply-chain policy; no hidden advisory suppression.
- blocker: The current upstream RSA release has no safe replacement release.
- re-open trigger: A maintained safe implementation is available or the dependent protocol is removed.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-NATIVE-REACTOR-001"></a>
## MOI-NATIVE-REACTOR-001 — Complete native readiness and completion backends
- status: todo
- priority: architecture
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
- outcome: Browser tasks run cooperatively without blocking rendering or input handling.
- acceptance: Immediate work enters through microtasks, no executor turn blocks the main thread beyond 16 milliseconds per frame, budget exhaustion yields through a browser macro-task, and a trace proves task progress while frame and input callbacks continue.
- scope: WASM executor dispatch and browser event-loop scheduling; no native driver thread.
- next step: Specify the turn budget and cancellation ownership against ADR 0007, then add the trace before implementation.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-WASM-WORKERS-001"></a>
## MOI-WASM-WORKERS-001 — Schedule shared-memory work across Web Workers
- status: todo
- priority: architecture
- outcome: Thread-capable browsers can execute Moirai work through a bounded worker pool without spinning the main thread.
- acceptance: Workers share the admitted memory instance, route bounded messages or ring entries, steal through atomics, and park with `Atomics.wait`/`notify`; unsupported browsers return a typed capability result.
- scope: Web Worker lifecycle, shared-memory routing, work stealing, parking, and capability detection.
- next step: Define the worker lifecycle and memory-isolation threat model before selecting the routing representation.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-FUTURES-IO-COMPAT-001"></a>
## MOI-FUTURES-IO-COMPAT-001 — Move the futures-io wrappers into the compatibility module
- status: todo
- priority: architecture
- outcome: One wrapper family in `moirai-async::io::compat` maps Moirai I/O onto every external trait family it supports, and `moirai-tls` consumes it.
- acceptance: `ToFuturesIo` and `ToMoiraiIo` leave `moirai-tls`; the replacement wrappers are `#[repr(transparent)]`, forward the task context and vectored writes, and pass the duplex, wake, EOF, and large-transfer scenarios of `io/compat/tests.rs`.
- scope: `moirai-tls/src/lib.rs` wrappers, `moirai-async/src/io/compat.rs`, their callers, and ADR 0006.
- next step: Search the workspace for `ToFuturesIo` and `ToMoiraiIo` callers, then extend the compat module with the `futures::io` traits and migrate them.
- basis: `3bf9b07808fd2326033f308eacbc2aa4325013fc`

<a id="MOI-WASM-PROMISE-FUTURE-001"></a>
## MOI-WASM-PROMISE-FUTURE-001 — Own Promise-to-Future callback lifetimes
- status: todo
- priority: architecture
- outcome: JavaScript Promise resolution and rejection wake one Rust future without leaking callbacks or requiring cross-thread access.
- acceptance: An owned callback guard maps resolve/reject to typed results, updates the current waker, cancels registration on drop, and passes lifecycle tests for completion, rejection, replacement, and cancellation.
- scope: general Promise bridge and bounded event handoff; existing specialized file/timer ownership remains authoritative.
- next step: Extract the shared ownership contract from delivered specialized bridges and prove one general state machine.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-WEBVIEW-RUST-PROVIDER-ACCEPTANCE-2026-10-13"></a>
## MOI-WEBVIEW-RUST-PROVIDER-ACCEPTANCE-2026-10-13 — Verify Rust page provider ownership [minor]
- status: todo
- priority: verification
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
- outcome: Moirai has a current line and branch coverage baseline and a concrete list of uncovered correctness-critical paths.
- acceptance: The configured coverage tool runs against the complete workspace test suite, records reproducible results, and files value-semantic tests for uncovered public behavior without weakening assertions.
- scope: coverage configuration, test execution, and resulting focused tests; no percentage claim from the historical 50-test snapshot.
- next step: Run the current workspace coverage command at this basis and classify uncovered changed and trust-boundary code.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-SPAWN-GLOBAL-MUTEX-REVIEW-001"></a>
## MOI-SPAWN-GLOBAL-MUTEX-REVIEW-001 — Review the sharded task registry decision
- status: todo
- priority: verification
- outcome: The landed sharded registry either receives independent approval or a forward correction grounded in ADR 0005 and current measurements.
- acceptance: Review the single-producer regression, multi-producer scaling, task-scheduling control, dense-block ownership, and cleanup interaction; record a verdict and any correction in ADR 0005.
- scope: landed registry architecture and its measurement contracts; no history rewrite.
- next step: The independent review is done and its registry defects are fixed (ADR 0005 carries the corrected bounds); repeat the decisive benchmark rows on deterministic counters or an isolated-core run, then record the verdict in ADR 0005.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-ASYNC-IO-COMPARISON-001"></a>
## MOI-ASYNC-IO-COMPARISON-001 — Verify native and compatibility I/O contracts
- status: todo
- priority: verification
- outcome: Native Moirai I/O and its Tokio compatibility path have measured, value-equivalent loopback behavior.
- acceptance: Heavy cancellation tests preserve buffers, registration/unregistration overhead is recorded, and native versus compat TCP loopback rows compare throughput and latency with identical values.
- scope: async network/file integration tests and `async_tcp_comparison` benchmark rows.
- next step: Build the cancellation fixture and paired loopback harness against the current reactor backends.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="ISSUE-005"></a>
## ISSUE-005 — Audit unsafe-code obligations
- status: todo
- priority: verification
- outcome: Every reachable unsafe operation has a current safety argument and the strongest executable check its platform permits.
- acceptance: Inventory current unsafe sites, verify each `SAFETY` obligation against its safe caller boundary, run Miri where supported and sanitizer/targeted substitutes elsewhere, and file any unsound or uncovered unit as a correctness item.
- scope: workspace unsafe blocks, public safe wrappers, FFI/platform boundaries, and their memory-safety tests; no stale 2024 count as a completion claim.
- next step: Generate a current revision inventory by crate and rank reachable trust-boundary sites before reviewing implementations. The scheduled `Miri` job in `.github/workflows/rust-ci.yml` holds the interpreted set; triage what its comment lists as excluded (moirai-async worker threads that outlive their owner, tests over the per-test budget, `shm_open` and socket tests Miri cannot run) and widen the set as sites are reviewed; a survey of utils, sync, scheduler, async and core found no further undefined behavior.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-WASM-HEADLESS-TRACE-001"></a>
## MOI-WASM-HEADLESS-TRACE-001 — Run browser lifecycle verification
- status: todo
- priority: verification
- outcome: Real browser engines verify WASM task progress, wakeups, and callback cleanup.
- acceptance: Headless Chromium, Firefox, and WebKit traces exercise cooperative tasks, event wakes, cancellation, and callback release with bounded artifacts and value assertions.
- scope: WASM browser tests and hosted trace workflow; no synthetic-only success claim.
- next step: Define one engine-neutral lifecycle scenario and make its trace deterministic before adding the matrix.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-FORK-JOIN-LATENCY-2026-09-10"></a>
## MOI-FORK-JOIN-LATENCY-2026-09-10 — Attribute fork-join tail latency
- status: todo
- priority: tightening
- outcome: Repeated small fork-joins use the available worker pool without the observed wake-latency tail.
- acceptance: The 64-task microbenchmark separates wake order, per-worker wake cost, caller participation, and spin policy across worker counts; the selected correction reduces p90 without regressing hot-path controls or idle CPU budget.
- scope: indexed fork-join submission, worker park/wake, caller-help policy, Criterion attribution, and consumer-sized confirmation.
- next step: Resume the retained latency distribution after the crash cause is fixed, then file or implement the measured wake-path correction.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="ISSUE-010"></a>
## ISSUE-010 — Attribute remaining result-handoff variance
- status: todo
- priority: tightening
- outcome: Remaining public result-slot and oversized-job variance is attributed to one measured production mechanism.
- acceptance: Profiles and paired Criterion runs identify the mechanism, preserve lifecycle timing and panic semantics, add no lock to the handoff path, and show no significant control-path regression.
- scope: result-slot ownership, oversized inline trampoline handoff, scheduler submission, and existing diagnostic benchmarks; rejected raw-pointer, relaxed-timestamp, and catch-removal variants remain rejected.
- next step: Profile the current public ready and oversized paths and compare code generation at the first divergent component.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-PARKED-POOL-FIRST-REGION-2026-09-17"></a>
## MOI-PARKED-POOL-FIRST-REGION-2026-09-17 — Reduce first-region wake cost
- status: todo
- priority: tightening
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
- outcome: Moirai supplies select ergonomics, IPv6, and a graceful shutdown signal through its bounded local runtime surface.
- acceptance: Each surface has typed cancellation and backpressure behavior, value-semantic integration tests, and comparison coverage against the corresponding Tokio behavior; HTTP/2 remains excluded without a named consumer.
- scope: select-equivalent ergonomics, IPv6 network paths, shutdown signaling, tests, and documentation.
- next step: Specify and deliver the select-equivalent contract as the first independent slice.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-WASM-NETWORK-FACADE-001"></a>
## MOI-WASM-NETWORK-FACADE-001 — Add bounded browser fetch and networking
- status: todo
- priority: feature
- outcome: WASM applications use owned bounded browser fetch and network operations without compiling the native network module.
- acceptance: Request, response, redirect, body, queue, deadline, and callback lifetimes are bounded; failures are typed; browser traces cover success, rejection, cancellation, and exhaustion.
- scope: browser `fetch` and the general browser network facade; native sockets remain separate.
- next step: Define the resource-limit and cancellation contract against ADR 0007 before exposing the first fetch operation.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

<a id="MOI-ASYNC-SEEK-001"></a>
## MOI-ASYNC-SEEK-001 — Define an async seek trait for Moirai I/O
- status: todo
- priority: feature
- outcome: Moirai files and in-memory cursors seek through a poll-based trait that maps onto `tokio::io::AsyncSeek`.
- acceptance: `File::seek` is reachable through the trait, the compat wrappers forward `start_seek` and `poll_complete` with the caller's task context, and a seek in flight survives a re-poll under a new waker.
- scope: `moirai-async::io::traits`, `moirai-async::fs::file`, `io/compat.rs`, and ADR 0006.
- next step: Specify the trait against `File`'s pooled seek request and decide whether a seek in flight is cancellation-safe before writing it.
- basis: `3bf9b07808fd2326033f308eacbc2aa4325013fc`

<a id="MOI-SPIN-BACKOFF-CONSOLIDATION-2026-09-30"></a>
## MOI-SPIN-BACKOFF-CONSOLIDATION-2026-09-30 — One spin-then-yield schedule for bounded-wait sites
- status: todo
- priority: tightening
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
- outcome: verify the shared WKWebView resource provider on native macOS, iPhone and iPad.
- acceptance: real Rust/WASM input/output, geometry, scene reconnect and bounded shutdown; permission denial is a separate host contract.
- evidence: six resource byte/MIME/traversal/symlink/socket/FIFO/bound tests pass; Metis macOS/device/simulator apps compile; macOS/iPhone/iPad native hosts deliver all seven assets including real WASM.
- [x] [patch] Correct Fetch status in moirai-webview 0.1.1: eight resource/response tests pass; real macOS Greet returns the exact Rust result for normal, Unicode and empty names.
- [x] [patch] Verify moirai-webview 0.1.2's Rust native menus: focused Command-Q and Command-W exit; actual minimize/restore, Zoom geometry and full-screen journeys pass; ADR 0073 records native values and capture limitations.
- [x] [minor] Add moirai-webview 0.2.0's exact host-authorized HTTPS popup handoff: 20 real policy/resource/Foundation tests and 196 compatibility checks pass; native preview records one successful OS handoff for the Metis logo. Canonical consumer and browser visual evidence follow delivery (ADR 0074).
- blocker: remaining native geometry/lifecycle evidence and physical device provisioning/availability are unset.
- scope: moirai-webview; [ADR 0068](docs/adr/0068-apple-webview-host.md).
