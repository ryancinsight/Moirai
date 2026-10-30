# ADR 0085: Result state access

Status: Accepted

## Context

`ResultCell` uses a public unsafe `StateWord` trait so packed and cache-aligned
atomic storage run the same completion protocol. The trait requires atomic load,
store, and compare-exchange operations. It also requires the safe method
`get_mut(&mut self) -> &mut u8`, although destruction only needs to classify the
final state and never mutates the word.

The existing protocol uses one state word. Result publication and waiter
ownership remain separate obligations within that protocol: release/acquire
transitions publish each payload before another thread reads it, registration
continues to wait while the producer holds `WRITING`, and the final owner alone
destroys any retained payload.

## Decision

Remove `StateWord::get_mut` without a replacement accessor. `ResultCell::drop`
classifies the final state through the existing `load(Ordering::Relaxed)` method.
This load is not result acquisition: `Drop` has exclusive ownership, and safe
ownership transfer establishes the happens-before edge for prior accesses. In
the shared `Arc` use, the final strong-count release/acquire edge precedes inner
value destruction.

The unsafe trait continues to require every load to observe the same coherent
atomic cell used by its stores and transitions. All publication and consumption
orderings, state values, waiter behavior, and payload-access rules remain
unchanged.

## Alternatives

An `exclusive_state(&mut self) -> u8` replacement would add a second way to read
the same cell solely for destruction. Keeping the borrowed getter preserves an
unneeded public implementation obligation. Strengthening the destructor load to
`Acquire` would imply that it acquires result publication, although ownership
handoff already provides that ordering.

## Consequences

External `StateWord` implementations remove their `get_mut` method. This is a
breaking public trait change and sets the workspace compatibility boundary to
0.12.0. The inherent `AtomicU8::get_mut` and `CacheAligned::get_mut` APIs remain
unchanged because they are independent storage APIs.

The completion protocol still waits during `WRITING`; this decision does not
claim broader task-lifecycle or waiter-progress changes.

## Verification

Unit tests cover untaken and consumed results after cross-thread publication,
including exact payload identity, exactly-once destruction, and rejection of a
second take. Existing pending, waiter, clone-panic, reentrant-destruction, and
model suites retain their contracts. SemVer comparison must report removal of
the required public trait method under patch compatibility.

## Overturning evidence

A safe ownership path that allows `Drop` to race a state transition, a backend
whose `load` is not coherent with its other state operations, or a regression in
payload destruction invalidates this access decision.
