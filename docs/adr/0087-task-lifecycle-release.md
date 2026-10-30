# ADR 0087: Task lifecycle release boundary

Status: Accepted

Revision 2026-10-30: preserve the unarmed automatic publisher while integrating
the lifecycle candidate. Compatible private repairs land before the public
0.12 boundary; activation remains a separate reviewed full-cohort change.

## Context

Awaitable task results, owned cancellation, shutdown ownership, and autonomous
wait deadlines form the runtime release scope. ADRs 0083-0086 and 0089 describe
their contracts. Owned completion registration, fallible timer construction,
and the owned I/O error source change public contracts. Cargo's pre-1.0
compatibility boundary therefore requires 0.12.0.

The user authorizes that version change and registry release subject to full
release gates. PR 650 preserves the complete candidate while compatible repairs
and release prerequisites are integrated. The preserved composition does not
itself certify a release or justify merging unrelated changes.

The default-branch publisher runs on every push and daily schedule. Without an
explicit cohort it selects every workspace version missing from crates.io.
PR 654 installs its existing empty-cohort interface before lifecycle or version
changes can trigger automatic publication.

## Decision

Workspace version: 0.12.0

Release every publishable workspace package at that version as one dependency-closed
unit and one matching annotated source tag. QUIC and WebView inherit that version
for their first publication. Internal requirements and the unpublished Python,
benchmark, and test consumers advance with the same source unit. The pinned
cargo-release version step performs no upload or tag operation.

Keep literal empty-cohort input on automatic push and schedule events.
The pinned planning action returns zero before Cargo metadata or registry reads;
its SemVer, authentication, upload, and release-tag jobs then remain skipped.
Keep existing release-event and manual-dispatch paths under their established
authority. This decision does not grant a separate package or Python release.

Integrate compatible private repairs first. Keep public additions and their
native callers at the one 0.12 boundary, with migration documentation and exact
consumer preparation. The executor error enum becomes non-exhaustive;
shutdown-worker creation failures retain their original I/O source. Changes to StateWord, metrics, or no-std capability are
included only when independently required by the delivered contracts; their
presence in the preservation branch is insufficient.

For this candidate, direct registry dependency floors use versions from the
verified lock as supported minimums, retaining caret compatibility. A later lock
advance does not itself raise those floors. The minimum-version gate preserves
Git provider identities and checks the direct-floor graph with Rust 1.95.

Before public integration, verify exact source with format, strict Clippy,
Nextest, benchmark smoke, doctests, rustdoc, MSRV, and supported features.
Compare the fetched base and published contracts under the declared breaking
classification. Run applicable production-cell Loom and Miri cases, consumer
contracts, dependency-closed packaging, minimal versions, an SBOM, and independent
reproducible builds.

The original strict Linux QUIC matrix retains every admitted workload and its
existing deadlines. Any pair failure blocks release. The default branch's
historical known-limit result does not establish 108 successful pairs.

Activate automatic publication only after the exact full candidate and release
cohort pass those gates. Replace the empty input with the complete publishable
workspace cohort in one reviewed change. Interrupted uploads retain the same
candidate, version, cohort, toolchain, target, profile, and archive hashes; only
the pending subset changes. Registry-installed smoke tests and documentation
health cover the entire unit before one source release is finalized.

## Alternatives

A 0.11 patch release misclassifies public breaks. Independently published source
slices can release an incomplete runtime. Removing the QUIC dependency,
suppressing a failing workload, or raising its deadline changes acceptance
instead of establishing it. Replacing the publisher duplicates its existing
owner rather than repairing the required unit semantics.

## Consequences

Private source repairs can land while publication remains unarmed. Publication
does not follow merely from completion of a source merge or a historical green
run. First publication may require the existing local Cargo credential store;
credentials never enter source, evidence, or logs.

The release does not claim Tokio feature, performance, or platform parity.
Non-yielding work remains nonpreemptible, task-wait expiry does not cancel the
underlying task, and thread-per-pending-timer scaling remains a limitation.
[ADR 0090](0090-shared-shutdown-ownership.md) is Accepted and implements shared
teardown ownership and request-only destruction. Its admission, reentrancy,
settlement, and caller deadline oracles must pass before release.

## Overturning evidence

A gate failure, incompatible consumer, incomplete registry unit, changed source
identity, or differing independent artifact holds activation until corrected.
If the pinned empty-cohort plan can reach authentication, upload, or tagging,
publication stays unarmed until its shared owner restores that boundary.
