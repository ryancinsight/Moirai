# ADR 0042: Worker idle hooks for owner-thread reclamation

- Status: Accepted
- Date: 2026-09-04
- Driver: `ATLAS-APOLLO-WORKER-RETENTION-2026-09-03`

## Context

Moirai workers outlive individual data-parallel operations. Thread-local
scratch and cache providers therefore retain their high-water allocations after
the operation completes. Reclamation must run on the owning worker thread and
must not add work to an active transform.

## Decision

Expose one bounded worker-idle maintenance seam. Providers register a plain
function pointer once during cold initialization. A worker invokes every
registered hook after exhausting its work-search budget and immediately before
parking. Registration uses a fixed table, returns a typed capacity error, and
the worker snapshots the table before invoking hooks so registration cannot
deadlock against a hook call. The hook contract is owner-thread-only,
non-blocking, and cheap when no capacity is reclaimable.

Apollo registers one consolidated hook for its FFT scratch banks. The hook
releases only the current worker's idle capacity; it is not called on each
scratch borrow and a coordinator call does not pretend to reach worker-local
state.

## Rejected alternatives

- Releasing at every scratch-borrow boundary reintroduces allocation churn and
  destroys warm reuse.
- Releasing from the coordinator cannot access worker thread-local storage.
- A global allocator purge has a broader scope than the provider's idle scratch
  and can evict unrelated allocations.

## Verification

The executor unit suite covers bounded registration, snapshot execution, empty
registries, and registration from a running hook. The parallel integration test
observes a worker hook through a condition variable after a completed fan-out.
Apollo supplies the retained-capacity and zero-warm-allocation measurement.
