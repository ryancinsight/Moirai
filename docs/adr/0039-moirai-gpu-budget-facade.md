# ADR 0039: Moirai GPU budget facade owns the planner input

Status: Accepted

- Date: 2026-09-03
- Change class: [arch] [minor]
- Refs: `MOI-GPU-BUDGET-IDENTITY-2026-09-03`, atlas ADR 0002

## Context

`moirai-gpu::plan_launch` consumes Mnemosyne's `KernelResourceBudget`. A
downstream provider can resolve Moirai and its direct Mnemosyne dependency
from different source revisions. Rust then treats the two identically named
budget structs as different types, so the provider cannot pass the budget it
constructs to the planner. The failure appears only when a fresh dependency
graph builds the provider and its baseline together; a locally unified Atlas
overlay can hide it.

## Decision

Make `moirai-gpu` the public construction facade for the planner input by
re-exporting `mnemosyne_core::KernelResourceBudget`. The occupancy module
imports that re-export, so the facade and planner share one package-local
source of the type name. Callers construct the budget through `moirai_gpu`
and pass it directly to `plan_launch`.

This is an additive public export. It changes no launch arithmetic, resource
validation, or runtime dispatch. The export is documented with a doctest that
constructs a budget and checks the planner's value result.

## Alternatives rejected

- Keeping the direct Mnemosyne import in every consumer leaves each consumer
  responsible for reproducing the planner's dependency identity and permits
  the source-split failure to recur.
- Adding a conversion helper or duplicate budget wrapper creates a second
  representation and a compatibility seam instead of exposing the canonical
  planner input.

## Verification

The focused `moirai-gpu` check, warning-denied Clippy, native tests, doctests,
warning-denied rustdoc, format gate, and standalone lockfile check cover the
facade export and preserve the existing occupancy arithmetic tests.
