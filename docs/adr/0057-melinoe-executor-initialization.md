# ADR 0057: Melinoe executor initialization

- Status: Accepted
- Date: 2026-09-13
- Driver: [MOI-EXECUTOR-REGISTRATION-ORDER-2026-09-11](../backlog.md#MOI-EXECUTOR-REGISTRATION-ORDER-2026-09-11)

## Context

Melinoe owns a dependency-free, process-global partition slot. Moirai is an
optional scheduler provider and therefore cannot be imported by Melinoe without
reversing the stack dependency. Moirai previously registered its bridge only
inside the lazy `global_arc` initializer. A direct Melinoe partition made before
the first Moirai access consequently used Melinoe's scoped-thread fallback.

The fallback is correct but creates and joins operating-system threads for each
partition. That cost is material for the fine-grained regions for which the
shared Moirai pool exists.

## Decision

`moirai-executor::initialize` is the explicit startup boundary. It builds the
shared executor once and registers the Moirai implementation of Melinoe's
`ParallelExecutor` seam. Calling it again refreshes the registration after a
test or integration uses Melinoe's `clear_parallel_executor` lifecycle hook.
The `moirai` facade re-exports the same operation as `moirai::initialize`.

Moirai's Melinoe extension functions call `initialize` before entering the pool
path, so those wrappers do not depend on an incidental earlier scheduler call.
Applications that call Melinoe's partition functions directly call
`moirai::initialize` (or `moirai_executor::initialize`) during startup.

No constructor crate or platform-specific linker section is used. Such a hook
would add unsafe, target-specific startup behavior and cannot provide the same
contract on WebAssembly. Melinoe remains independent and keeps its fallback;
the provider choice stays at the integration boundary. No size threshold is
added here; `ExecutionPolicy` owns that decision.

## Verification

`moirai-executor/tests/melinoe_registration.rs` clears the Melinoe slot, calls
`moirai_executor::initialize`, then invokes a direct four-shard partition and
checks every returned shard and element. The Melinoe extension tests cover the
parallel, sequential, adaptive, ragged, and empty-region paths. The focused
nextest, Clippy, formatting, doctest, and rustdoc gates run against the same
source revision.

## Residual risk

A raw Melinoe call made before the application initializes a provider still
uses Melinoe's documented fallback. This is intentional: no dependency-free
mechanism can infer an optional scheduler before any integration code executes.
