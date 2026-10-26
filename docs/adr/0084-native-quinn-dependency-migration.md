# ADR 0084: Native Quinn dependency migration

Date: 2026-10-26
Status: Proposed

Exact-head integration and independent review remain pending.

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
