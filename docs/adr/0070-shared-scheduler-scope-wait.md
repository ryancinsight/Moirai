# 0070 — Share the scheduler wait primitive across borrowing scopes

Status: Accepted

Date: 2026-10-12

## Contract and decision

The existing indexed-allocation test measures 16 allocations for eight warmed
fan-outs on macOS. Darwin's standard mutex and condition variable allocate
native storage on first use; constructing both inside every scope violates the
existing zero-allocation contract. Workload, repetitions and the zero threshold
remain unchanged.

Each scope borrows the scheduler's existing wait mutex and condition variable.
Its pending, failed and unrun counters remain independent stack values. The
scheduler outlives every scope by the borrowing API and every job by scope
drain. No new shared allocation or per-call reference count is introduced.

The last completion holds the shared mutex before changing its counter 1 to 0
and notifying all waiters. A waiter holds the same mutex when checking its own
counter and releases it atomically with parking. Therefore it cannot miss its
terminal notification. Wakes from another scope or scheduler join are spurious
and cannot satisfy the waiter's independent pending predicate. Every waiter
reacquires the mutex before destroying its stack state, so final completion
finishes its last state access before destruction.

Private state carries explicit borrowed lifetimes; every constructor and raw
completion pointer annotation migrates together. Public scheduler APIs remain
unchanged. Replacing the OS wait with spinning, allocating new per-call owners
or weakening allocation tests is rejected.

## Verification

Run the unchanged indexed allocation/value tests, real concurrent independent
scopes, terminal-completion lifetime tests and the complete bounded native suite.
Existing Miri and Loom gates retain their domains. The 0.8.0 changelog records
this internal correction alongside the browser PAL increment.
