### Added
- `moirai-transport` `IpcTransport`: a **real** same-machine inter-process
  transport over shared memory (`moirai_core::ipc::SharedQueue`), replacing the
  former placeholder that returned constant errors. Carries length-prefixed
  fixed-size frames (≤ ~4 KiB/message), Unix/Windows only, used directly (not via
  `TransportManager`, to avoid the `Local`-address overlap with
  `InMemoryTransport`). Surfaced in the `moirai` umbrella.
- `moirai-transport`/`moirai` now re-export the rkyv-style archive channels
  (`ArchivedUniversalSender`/`ArchivedUniversalReceiver`/`ArchivedMessage` +
  `ArchiveSerialize`/`ArchiveView`) at the crate root — the canonical typed
  cross-boundary channel.
- `moirai-scheduler`: a bounded `loom` model of the Chase-Lev steal/pop ordering
  protocol (`tests/loom_chase_lev.rs`, gated behind `--cfg loom`), checking the
  exactly-once invariant across the modeled interleavings. `loom` is wired as a
  `cfg(loom)`-only dependency, so normal builds are unaffected.
- `moirai-transport`: `MessageRouter::{unsubscribe, subscriber_count}` and
  `ConnectionManager::{state, is_connected, connected_addresses}` query methods;
  `ConnectionState` is now a public, `Copy` enum.

