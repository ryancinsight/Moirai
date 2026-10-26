# ADR 0084: Native Quinn dependency migration

Date: 2026-10-26
Status: Accepted

Decision approval: on 2026-10-26 at 12:07 UTC, the authenticated repository
owner approved the specific pending ADR 0084 design request with a thumbs-up
reaction. The request identified local candidate
`cab5499fa4ec41064a4c191b55911a8dbc852b60` and explicitly covered the
licensed `moirai-quinn` adaptation, native `ReadBuffer`, Tokio-free default and
minimal dependency graphs, and the pre-1.0 public migration from Quinn-owned
read/write error types and direct buffer calls. Shared protocol error identity
remains unchanged. The approval covers this design decision only. It grants no
license or security exception, performance waiver, integration or release
authority. Any implementation change still needs final exact-head review and
the unchanged qualification gates.

The first independent review requested two P2 repairs. The same reviewer
accepted their corrected source on `b50a96cfebf38a99c52b42e816ea3a5340927c25`;
later candidate heads still require exact-revision review and integration.

Decision scope: the authorized native Tokio removal, separate from the held
QUIC recovery experiments and the notification repair delivery owner.
Change class: [minor] for additive async channels; [arch] for a deliberately
versioned Quinn public I/O boundary. No 1.0 admission follows from this work.

## Requirements and ownership

Coeus disallows Tokio anywhere in its native normal dependency graph. Moirai
already owns scheduling, timers and socket readiness. Quinn 0.11.12 still
requires Tokio synchronization and publicly exposes its ReadBuf. Disabling
runtime features therefore does not meet the dependency contract. Preserve
native default and minimal QUIC, supported platforms, stream bounds, security,
all recovery settings and the existing 108-pair performance domain/deadlines.

Notification delivery belongs to task 01a159ec-7017-7725-a554-194472dc20dc;
this lane does not reproduce or overwrite that patch. Historical recovery
task 01a13836-1b78-7d83-93b2-fdef2d3b650c retains its source and acceptance
holds. A failed task-routing read does not authorize adoption of that source.

## Channel decision

Extend the authoritative MPSC channel using explicit bounded and unbounded
mode types. Existing Sender<T>, Receiver<T> and their bounded send/recv
contracts retain the default bounded mode. The unbounded constructor returns
Sender<T, Unbounded> and Receiver<T, Unbounded>; its enqueue method is synchronous and
returns the exact unsent value after close. A distinct immediate-admission name
avoids ambiguous qualified calls to the existing bounded Sender::send API. Capacity is represented as
Some(capacity) or None, never a maximum integer or a blocking-core channel.

Both modes share one FIFO buffer, sender count, closure state and waiter
queues. A receive registration belongs to the receiver across direct
poll_recv calls. A borrowed recv future cancels only a registration it polls.
Dequeuing and waiter replacement occur under the same mutex as enqueueing;
waking and dropping queued payloads occur outside the lock.

Linearization points are insertion/removal/close under that mutex. Per-producer
FIFO follows from insertion order. Accepted values are removed exactly once
by receive or receiver destruction. Last-sender drop closes admission but
retains accepted values until drained. Receiver drop closes admission and
releases buffered values even while senders remain alive. Removing a receive
registration cannot remove a value; cancellation cannot lose a message.
Receiver destruction signals all parked senders after unlocking and before
running any queued payload destructor. A panicking destructor propagates its
panic without bypassing those notifications.

## Native I/O and licensed source boundary

Inspect the exact Quinn package, its public methods, license notices and
release workflow before selecting a versioned first-party adaptation. Reuse
Moirai's initialized-slice I/O traits where they cover the contract; any native
uninitialized buffer contract must preserve initialized-prefix bounds, partial
progress, EOF, reset/error and cancellation semantics. Do not rebrand Tokio,
mask dependencies, remove supported compatibility features or change license
terms. Keep upstream provenance and protocol-engine identity explicit.

The resulting native I/O contract and migration notes must describe every
changed public signature before integration. A local path patch is diagnostic
evidence, never a registry or downstream release receipt. Publish and verify
dependencies in order through their release owners, then qualify a fresh
consumer without path/patch overrides.

## Verification

Run unchanged bounded regression tests plus value-semantic unbounded FIFO,
multi-producer ordering, last-sender drain/closure, receiver-drop payload
return, persistent waker/replacement, cancellation and destructor-reentry
tests. Reproduce the existing buffered-payload retention defect on the exact
baseline. Check additive APIs against the released async package.

Source graph evidence and real runtime evidence are distinct gates. The final
native graph must exclude Tokio without changing Coeus policy. Real default
and minimal facade sessions, stream behavior, supported-platform jobs,
complete original performance domain, packaging/license/RustSec gates and
independent review must all bind to the final exact revision. Test exclusions,
unrun checks and preserved existing recovery holds stay explicit.

## Selected source and release boundary

The implementation imports the licensed Quinn 0.11.12 async driver as the
explicit package `moirai-quinn 0.1.0`, with upstream archive SHA256
`4051e23e9185c255a7e33ef59cdbca87a22d359052eecd22fc6b901fb37d9d11`.
The crate carries unchanged MIT and Apache notices, original file digests and
upstream commit `eaec0db4bcb698f76df736d89938743c88a2ad5f`.
`quinn-proto 0.11.19` and compatible `quinn-udp 0.5.16` retain protocol and UDP
ownership; no experimental recovery patch is imported. The separate UDP 0.6
series is not API-compatible with this upstream driver requirement.

The driver uses actual Moirai NotifyFuture, explicit unbounded MPSC enqueue,
and the existing oneshot recv/poll_recv at the three real connection sites.
Its native buffer signature takes ReadBuffer, and its streams implement
Moirai AsyncRead/AsyncWrite. Optional futures-io, Tokio I/O, Tokio runtime,
Smol and async-std support remains explicit. Upstream unit-test runtimes and
an unmodified upstream wire-reference peer are development dependencies;
they do not establish native normal-graph acceptance.

The `runtime-async-std` compatibility feature spawns on
`async-global-executor`, the executor that `async-std::task::spawn` uses, and
detaches each driver task. This removes the discontinued `async-std` package
from the adapted driver's dependency graph without dropping the runtime
feature. First spawn applies async-std's `ASYNC_STD_THREAD_COUNT` and
`ASYNC_STD_THREAD_NAME` configuration and default worker name, preserving an
already initialized executor. Fresh-process regressions cover configuration,
task lifetime and timer reset; an actual UDP/TLS session has a separate test. The
archived recovery verifier reads the recorded revision's committed lock and
checks its tree, while the live observer continues to use the current lock;
historical experiment archives and values remain unchanged.

The original Tokio unbounded receiver cooperatively returned `Pending` after
128 ready receives in the reviewer's fresh-task oracle. Moirai's channel has
no task-local Tokio budget, so this driver explicitly caps connection events
at 128 per poll. Transmission, timers and application events run after each
batch; continuation wakes outside the connection-state lock. The separate
endpoint event loop already has its own finite work bound. Native scheduling
remains Moirai-owned, and the original timing/performance acceptance remains
open.

The imported driver is also held to Atlas's non-increasing source-conformance
ratchet. Its repair separates datagram conversion from the crate manifest,
denies missing public documentation, makes test observations value-sensitive,
and encodes one-shot incoming, connecting and datagram-send transitions in
single ownership paths. After a bounded send batch, one protocol probe sets
the app-limited state if no work remains or stages one transmit for the next
driver poll. Endpoint rebind retains each earlier socket until the connections
that could still receive there migrate or drain; no more than two retired
sockets are polled per turn. Poisoned state is never resumed as protocol state:
fallible endpoint operations report closure or I/O error and teardown releases
waiters/channels. Real UDP regressions cover multiple connections and repeated
rebinds, alongside the existing all-stress driver suite. This internal repair
does not broaden the approved public API break or waive the unchanged Linux
performance, FIPS license, platform or release gates.

Native source relocation separates impl families without changing protocol
method bodies or dropping original tests. Every imported Rust source file is
below 500 lines. New source identity remains distinct from upstream Quinn;
no package is named or aliased as Tokio, no patch/allowlist masks a dependency.

Moirai async 0.12.0 adds APIs without changing bounded API contracts; a strict
patch-level comparison to published 0.11.0 passes. QUIC 0.2.0 and facade
0.12.0 classify the changed public error-type owner as a pre-1.0 break.
Release order is async prerequisite, licensed driver, QUIC provider, facade.
The shared release/API inventory includes the new package. Normal merge,
publication, licensing/security, platform, full performance and independent
review gates remain mandatory. Automatic main-merge publication is an actual
side effect of the existing release workflow; no merge is authorized by local
validation results alone.
