# ADR 0040: One First-Party Memory Source Identity During Provider Co-Evolution

Status: Accepted

- Date: 2026-09-03
- Change class: [arch] [patch]
- Refs: `MOI-MNEMOSYNE-IDENTITY-2026-09-03`, atlas ADR 0002

## Context

The GPU planner consumes `KernelResourceBudget` from Mnemosyne. Moirai's
planner facade and the provider graph were still selecting Mnemosyne
`03fe32f`, while the current Atlas provider revision is PR #123 at
`7f173751`. Combining those revisions creates duplicate nominal package types
and adds avoidable provider compilation to downstream consumers.

## Decision

Advance both Moirai workspace edges, `mnemosyne-memory-core` and
`mnemosyne-memory`, to `7f173751` while Mnemosyne PR #123 is under review.
Keep the exact revision as a temporary co-evolution pin with a removal
trigger: after the provider change merges to main, remove `rev` and
regenerate the standalone lockfile. The workspace manifest remains the
dependency SSOT; no conversion, path override, or compatibility layer is
added.

## Alternatives rejected

- Keeping `03fe32f` was rejected because it preserves the duplicate source
  identity in the provider graph.
- Adding a downstream budget conversion was rejected because the shared type
  identity is owned by the provider dependency edge.
- A stack-local path override was rejected because it changes standalone and
  published resolution.

## Verification

The standalone lock resolves Mnemosyne `7f173751` and Eunomia `fdbf122`.
Workspace check and warning-denied Clippy pass; Nextest passes 987/987 with
9 expected skips; Loom passes 19/19; the Rust 1.95 floor check and the
no-default-features core suite pass; 22 executable doctests, warning-denied
rustdoc, formatting, and `git diff --check` pass.
