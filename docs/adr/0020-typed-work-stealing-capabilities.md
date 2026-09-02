# ADR 0020: Typed work-stealing capabilities

Status: Accepted

- Date: 2026-07-13
- Change class: [major]
- Refs: ISSUE-211

**Context.** `ChaseLevDeque` and the unused alternative `BlockBasedDeque` exposed owner-only
`push`/`pop` and thief-only `steal` through one `Sync` type. Cloning an `Arc` of
that type lets safe callers run multiple bottom-side operations concurrently,
invalidating the `UnsafeCell<MaybeUninit<T>>` aliasing proof. Exclusive access
to an owner endpoint also does not prove reclamation quiescence while stealer
endpoints remain alive.

**Decision.** `ChaseLevDeque` constructs one non-`Clone`, `Send + !Sync` owner
and cloneable `Send + Sync` stealers over private `Arc` storage. Owner
operations require `&mut self`; steal operations exist only on stealers. The
default `DeferredReclaim` ZST retains resized arrays until the final endpoint
drops; shared live array reclamation remains opt-in through the Moirai-owned
access-counted policy. Batch
steal returns an allocation-free owning iterator whose destructor drops an
unconsumed tail, so panic cannot leak transferred values.

Delete `BlockBasedDeque`: no production path consumes it, it duplicates the
canonical deque role, and safe node reuse requires a reclamation subsystem
solely for that unused alternative. Introducing `crossbeam-epoch` or a new
hand-rolled EBR would violate first-party ownership or add unjustified unsafe
synchronization. `SplitDeque` remains a distinct, consumed composition over the
canonical typed Chase-Lev endpoints.

The executor stores only stealers and external injectors in shared worker
state. Each worker thread owns its bottom-side endpoints on its stack. Nested
scope helping and diagnostic paths use shared steal endpoints; they never
recover owner access through TLS, raw pointers, or runtime alias checks.

**Rejected alternatives.** A mutex around the public combined deque encodes
ownership at runtime and adds hot-path contention. A thread-local raw owner
pointer recreates an unsafe aliasing contract. Treating `&mut owner` as a
quiescent reclamation proof ignores concurrently active stealers. Preserving
the unconsumed block deque via Crossbeam EBR adds an external production
substrate; implementing EBR locally for one dead API adds unsafe debt.

**Verification.** Compile-time auto-trait assertions and compile-fail doctests
encode capability separation; Loom checks bounded owner/thief interleavings;
generic contention tests check exact-once delivery; Miri checks endpoint drop
order and retired-storage lifetime. Criterion compares the endpoint migration
against the stored scheduler baseline. Evidence claims remain bounded to the
checks actually executed.
