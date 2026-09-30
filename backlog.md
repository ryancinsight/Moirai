# Moirai backlog

This is the single source of truth for active Moirai work. Items are ordered by
priority (`correctness`, `architecture`, `verification`, `tightening`, then
`feature`) and dependency. Hosting records claims and review state; this file
contains only `todo` and `blocked` work. Risks without an active implementation
belong in [gap_audit.md](gap_audit.md).

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
- next step: Reconcile ADR 0006 with the current Windows cooperative fallback and specify the first complete backend slice.
- basis: `d352be47a4fdcbf9cd8d27ae917d323db3f52e26`

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

<a id="MOI-TOKIO-IO-COMPAT-001"></a>
## MOI-TOKIO-IO-COMPAT-001 — Complete bidirectional async I/O trait mapping
- status: todo
- priority: architecture
- outcome: Moirai and Tokio I/O types interoperate through transparent typed wrappers without scheduling or allocation in the adapter.
- acceptance: Both wrapper directions implement read/write traits, readiness transitions wake the correct context, layout is transparent, and native/compat behavior is value-equivalent.
- scope: async I/O compatibility wrappers and readiness mapping; Tokio remains a comparison and interoperability dependency.
- next step: Map vectored writes (`poll_write_vectored`, `is_write_vectored`) through both wrappers; read, write, buffered read, flush, shutdown, layout, and waker propagation are delivered and tested.
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
