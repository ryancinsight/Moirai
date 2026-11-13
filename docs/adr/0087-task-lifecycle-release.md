# ADR 0087: Task lifecycle release boundary

Status: Accepted

Revision 2026-10-30: establish the 0.12.0 compatibility boundary with the
required `StateWord` method removal while leaving the pending publisher's cohort
empty. Remaining task-lifecycle changes, consumer preparation, release gates,
tagging, and publication remain separate prerequisites.

Workspace version: 0.12.0

Revision 2026-10-29: install the pending publisher with an empty release cohort
before any task-lifecycle source or version change lands. Activation remains a
separate, full-cohort change after the exact release candidate passes every
release gate.

## Context

The default-branch workflow invokes the version-change publisher on every push
and daily schedule. Without an explicit cohort, that publisher selects every
workspace manifest version missing from crates.io. The empty cohort keeps the
first 0.12.0 public boundary from becoming a publication trigger while the
remaining task-lifecycle composition and its release evidence stay incomplete.

The pinned shared workflow accepts an exact package/version JSON object. Its
pinned planning action treats an empty object as a zero-package plan before it
reads Cargo metadata or registry state. The shared SemVer and publication jobs
run only when that plan reports a nonzero package count.

## Decision

Pass the literal empty object to the pending publisher on push and schedule.
The plan then reports zero packages, so the SemVer, registry authentication,
upload, and release-tag jobs cannot run from those events.

Keep the existing release-event and manual-dispatch paths unchanged. A published
`crate-<package>-v<version>` release validates and publishes that package through
the established trusted-publishing workflow. A manual dispatch validates the
explicit package and version with a dry run; its shared publication job remains
restricted to release events. Neither path gains authority from this decision.

Activate automatic publication only in one later reviewed change that replaces
the empty object with the complete publishable 0.12.0 workspace cohort. That
activation follows the version change and the exact candidate's full native,
SemVer, documentation, feature, supply-chain, package, reproducibility, and
strict QUIC gates. Historical or scoped results do not satisfy that condition.

The workflow remains unarmed. The manifest version records the required public
trait method removal, but this boundary does not deliver the broader
task-lifecycle contracts, authorize an upload, or create a source or registry
tag.

## Alternatives

Leaving workspace discovery enabled until the final composition lands couples
ordinary source integration to registry mutation. Removing the push and schedule
events would discard the established retry path rather than hold it in a tested,
explicit state. Advancing the shared workflow pin is unnecessary because the
current pinned interface already provides the empty-cohort contract.

## Consequences

Task-lifecycle slices and release prerequisites can land without automatic
publication. Push and schedule runs still execute the real pinned plan, but its
zero count prevents any credential request or external mutation. Final release
activation must name all 19 publishable workspace packages at version 0.12.0
and remains reviewable as a single bounded configuration change.

## Overturning evidence

If the pinned plan no longer returns zero before metadata and registry access,
or a zero-count plan can reach authentication, upload, or tagging, publication
stays unarmed until the shared owner restores those properties and this repository
updates to a reviewed pin.
