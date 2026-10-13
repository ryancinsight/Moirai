### Added
- `Moirai::spawn_detached` (and `TaskSpawner::spawn_detached`): a fire-and-forget
  spawn that returns no handle and skips the per-task `Arc<TaskResultSlot>`
  allocation and atomic refcount that result-bearing spawns require — the
  cheapest dispatch path for background work whose output is not needed.
  Lifecycle/metrics tracking and shutdown drain are preserved, and panics are
  isolated. The trait method has a non-breaking default (routes through
  `spawn_blocking`); `HybridExecutor` overrides it for the no-allocation path.
- `moirai-pal`: a **real readiness reactor on Windows** (`WsaPollReactor`, backed
  by `WSAPoll`), replacing the non-functional IOCP backend. The IOCP completion
  model signals completions of *posted overlapped operations*, not socket
  *readiness*, so it could never drive the readiness-based `net.rs` futures —
  Windows async sockets therefore fell back to a 100%-CPU cooperative busy-poll.
  Windows now activates the process-global reactor like every other platform, so
  async TCP/UDP I/O is driven by real readiness instead of busy-polling. The
  reactor is `WSAPoll`-level-triggered (self-heals lost edges), self-cleans
  closed sockets via `POLLNVAL`, and is interrupted by a loopback wake socket.
  Verified by the full async net suite plus direct reactor tests (readiness
  delivery, wake interruption, stale-socket self-cleaning).

[Version 0.3.0 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
