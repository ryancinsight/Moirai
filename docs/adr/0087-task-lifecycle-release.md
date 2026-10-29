# ADR 0087: Task lifecycle release boundary

Status: Accepted

Revision 2026-10-29: automatic publication requires the complete successful
main-branch verification workflow at the identical commit, in addition to
release preflight. A pending or failed run rejects publication; the existing
scheduled release can collect a later successful result without polling.

Revision 2026-10-29: align the unpublished QUIC and WebView packages with the
workspace version and release tag. Their standalone manifest numbers do not
have a prior independent release decision; both registry names remain absent.

Workspace version: 0.12.0

## Context

Awaitable task results, owned cancellation, bounded shutdown calls and autonomous
wait deadlines change the runtime as one dependency-ordered unit. ADRs 0083-0086
specify these contracts. Replacing the unsafe `StateWord` method and adding a
field to the exhaustive `ExecutorMetrics` struct break external implementations
and literals. Cargo's pre-1.0 compatibility boundary therefore requires 0.12.0.

The main-branch publisher runs when manifest versions change. Separate bumped
merges could publish only part of this runtime. The user authorizes the version
change and registry release subject to full release gates.
[PR 650](https://github.com/ryancinsight/Moirai/pull/650) preserves the composition
and its reviewed source branches while verification and release prerequisites
are completed.

## Decision

Release all publishable workspace packages at 0.12.0 in one reviewed composition
and one annotated v0.12.0 source tag. QUIC and WebView inherit that version for
their first publication. Update every internal requirement,
including unpublished Python, benchmark and test consumers. Use the pinned
cargo-release 1.1.7 version step; it performs no upload or tag operation.

For this release, direct registry dependency floors use the versions in the
verified committed lock. These are supported minimums, not claims about the
earliest compatible upstream API. Requirements retain caret compatibility.
Subsequent lock advances do not automatically raise these floors. The minimum
gate resolves the declared direct floors from the committed graph, preserves
every Git provider identity, and checks that graph with Rust 1.95.

The release includes the four lifecycle contracts and their regression tests.
It does not claim Tokio feature, performance or platform parity. Non-yielding
closures remain nonpreemptible, destructor shutdown retains its documented join
contract, and task-wait expiry does not cancel the underlying task. The PAL
thread-per-pending-timer implementation remains a scalability limitation.

Before merge, verify the exact candidate with workspace format, strict Clippy,
nextest, benchmark smoke, doctests, rustdoc, MSRV and supported feature gates.
Run SemVer against the fetched base and published contracts under the declared
breaking classification, the production-cell Loom models and applicable Miri
cases. Release prerequisites include dependency-closed package verification,
minimal-version/MSRV checking, an SBOM and independent reproducible builds.
The strict original Linux QUIC matrix must retain every admitted workload and
its existing deadlines; any error blocks the gate. Historical results do not
certify this candidate.

Retain the existing trusted-publishing workflow selected by Atlas ADR 0068.
Its first-publication restriction must be resolved for QUIC before the complete
runtime closure can publish. Do not infer complete publication from a successful
workflow that skips a package. Verify exact registry versions and installed
artifact smoke tests after upload; no Python registry release is part of this
Rust release.

## Alternatives

A 0.11 patch release misclassifies the public break. Independently bumped merges
can publish incomplete source. Removing the native QUIC dependency or suppressing
its failing workloads changes the production contract instead of establishing
release acceptance. Replacing the publisher introduces an unrelated release
architecture change.

## Consequences

External state implementations follow the task-result migration guide; exhaustive
metrics literals initialize the new field. One protected merge is the release
authority boundary. First publication can still require local registry access;
credentials remain in the Cargo credential store and never enter source or logs.

## Overturning evidence

A candidate gate failure, unresolved publication dependency, incompatible
consumer or differing independent build artifact holds the release until its
cause is corrected. Evidence that either public change is compatible may narrow
the version classification, but does not permit partial source publication.
