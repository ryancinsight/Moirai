### Fixed
- `moirai-core` `SecurityAuditor`: the audit-event buffer is now a `VecDeque`
  with a hard count cap (16,384) and amortized-O(1) front-pop eviction. The prior
  `Vec` ran an O(n) `retain` on *every* recorded event (O(n²) under a spawn
  storm) and was bounded only by a days-to-weeks retention window, so it grew to
  millions of entries under load. Also uses `checked_sub` for the retention
  cutoff (no panic on an absurd clock).
- `moirai-async` broadcast channel: `BroadcastRecv` now clears its registered
  waker on drop (mirroring `WatchChanged`), so a cancelled `recv` future no
  longer leaves a stale waker that the next `send` spuriously wakes/retains.
- `moirai-transport` network transport: bounded read/write timeouts
  (`NETWORK_IO_TIMEOUT`, 30s) on accepted and connected TCP streams in
  `NetworkTransport` and the remote-task server, so a peer that connects then
  stalls mid-frame can no longer pin a worker thread indefinitely.
- `moirai-core` rate limiter: saturating time arithmetic on the window-advance
  computation (`numerical_discipline`); documented the limiter's approximate
  (non-hard-quota) semantics.

[Version 0.3.0 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
