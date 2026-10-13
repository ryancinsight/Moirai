### Changed
- `moirai-core` `SharedQueue<T>` now bounds `T` by `bytemuck::Pod` (was `Copy`):
  shared-memory contents are written by one process and read as `T` by another,
  so the element type must be valid for every bit pattern. `bytemuck` is promoted
  to a workspace SSOT dependency (also adopted by `moirai-gpu`).
- `moirai-async` `ConnectionPool` is now keyed by a unique `ConnectionId` instead
  of peer `SocketAddr`; `ConnectionInfo` gains a `peer_addr` field and
  `add_connection`/`add_connection_reserved` return the id. Streams untrack by id
  at drop. (Pre-1.0 breaking change to the connection-pool surface.)

[Version 0.3.0 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
