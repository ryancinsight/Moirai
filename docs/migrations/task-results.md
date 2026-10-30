# Task result state migration

The workspace compatibility boundary is 0.12.0 under
[ADR 0087](../adr/0087-task-lifecycle-release.md).

`moirai_utils::result_cell::StateWord` no longer requires
`get_mut(&mut self) -> &mut u8`. Remove that method from external
implementations. The existing `load`, `store`, `compare_exchange`, and
`pending` requirements are unchanged; `load` must observe the same atomic cell
used by the other operations.

`ResultCell` now uses `load(Ordering::Relaxed)` during destruction to classify
the final state. Implementors do not add an exclusive snapshot method or cache
the state. The load occurs only after ownership becomes exclusive and does not
replace the release/acquire edges used for concurrent result publication and
consumption.

The inherent `AtomicU8::get_mut` and `CacheAligned::get_mut` APIs are unaffected.
[ADR 0085](../adr/0085-result-state-access.md) records the access decision.
