## [0.6.0] - 2026-09-07

### Added

- Bounded RFC 6455 WebSocket service support for native Moirai async streams,
  including strict HTTP upgrade validation, bounded frames and messages,
  protocol deadlines, and terminal close/error handling.
- Pure-Rust SHA-1 and Base64 primitives for protocol handshakes, plus owned
  browser DOM, timer, and cancellable local-task handles.

### Changed

- The workspace release line now carries the Atlas provider graph at version
  0.6.0, with the scheduler, iterator, and GPU surfaces sharing the current
  first-party dependency contracts.

