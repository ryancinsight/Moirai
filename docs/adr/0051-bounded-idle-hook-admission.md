# ADR 0051: Bounded idle-hook admission

Status: Accepted

Date: 2026-09-09

Driver: [MOI-IDLE-HOOK-ADMISSION](../backlog.md#MOI-IDLE-HOOK-ADMISSION),
[Apollo worker admission](../../../apollo/backlog.md#apollo-worker-hook-admission).

Revision 2026-09-09: the fixed registry, public error and caller migration are
implemented. Focused native tests, warning-denied Clippy and the worker
quiescence integration test pass on the pinned Windows toolchain.

## Context

Moirai 0.6 at `3eb9f3ce` accepts an unbounded number of idle hooks but copies
only the first 16 into its execution snapshot. The seventeenth registration
succeeds without ever running. This can silently retain a consumer's worker
scratch. The bounded registry in the older `83aa411` provider rejects overflow;
its scheduler architecture is not needed to correct current admission.

## Decision

Keep 16 process-wide registration slots, matching the existing snapshot budget.
A fixed array behind a mutex owns registrations; registration fills the first
empty slot or returns `IdleHookRegistrationError::CapacityExhausted` without
mutation. Registration, initialization and snapshotting allocate no heap
storage. There is no removal API: the resource lasts for the process lifetime,
and retrying a full registry cannot recover capacity.

Each snapshot executes in insertion order, including duplicate registrations.
Concurrent registration linearizes at the registry mutex. Execution copies the
complete array under that mutex and releases it before any callback. Reentrant
registration affects later snapshots, not the snapshot already executing.
Consumers register once per provider and handle rejection before relying on
automatic reclamation. This is one registry implementation used by both the
public process registry and isolated test fixtures.

The worker insertion point stays unchanged: finish a job, find no work, check
shutdown, exhaust the spin search, run allocator maintenance, run hooks, then
enter the existing idle-bit publication/pending-work/park protocol. Quiescence
belongs to the current compute worker, not the whole pool. Dedicated blocking
workers do not execute these hooks. Registration does not wake parked workers.
A wake during callbacks is handled by the subsequent parking protocol's
pending-work check. Internal re-parking does not repeat hooks. Shutdown can
exit before the hook boundary, so hooks are not guaranteed finalizers.

A callback panic propagates, skips the rest of its snapshot and, on a worker,
unwinds that worker. It cannot poison the registry because the callback holds
no registry lock. Callback authors must avoid panic, recursive hook execution,
and waiting for work that depends on the current worker. Worker replacement,
panic containment and shutdown finalization are outside this admission change.

## Migration and alternatives

This change is [major] [arch]: `register_idle_hook` returns
`Result<(), IdleHookRegistrationError>` instead of `()`. Callers must propagate
or handle the result before treating a hook as installed. The error is exported
through the same executor and facade paths as registration. No compatibility
wrapper, version bump or release is part of this increment. Apollo remains on
its existing provider until its separate admission migration is verified.

Growing the snapshot would only move the silent-loss boundary. An unbounded
snapshot would allocate on the worker idle path and remove the existing
resource budget. Deduplication would change duplicate-registration semantics.
Keeping undocumented insertion order would leave ordered consumer observation
without a contract. A measured requirement beyond 16 independent provider
registrations can reopen the capacity decision as a separate reviewed change;
it does not justify accepting callbacks that cannot run.

## Verification and limits

Local registry fixtures test empty execution, all 16 slots, the rejected
seventeenth registration, unchanged ordered values after rejection, duplicates,
reentrant publication and the exact executed prefix when a callback panics.
These fixtures do not mutate the process registry. The public worker test uses
the existing 128 chunks of 256 values and two reuse epochs. Bounded channel
events report each owner's released scratch element count, checksum and zero
remaining capacity; per-owner work accounting is the independent oracle.
Events remain observable even if reclamation finishes before the parallel call
returns. The prior five-second deadline remains unchanged; no polling sleeps
or timing assertions establish correctness.

The focused `moirai-executor` suite passes 144/144 tests and the
`moirai-parallel` worker quiescence test passes 1/1 on the pinned Windows
toolchain. Warning-denied Clippy passes for both affected packages. The
worker test waits on a condition variable with a five-second deadline; it does
not poll or sleep. Rustdoc, doctests and API classification remain release
gates. Existing wake/shutdown and Loom suites must remain green; no new
guarantee about worker replacement, hook-triggered shutdown or callback
execution on every park is claimed.
