### Fixed
- `moirai-sync`: make `ShardedResourcePool::clear` linearizable with
  reservation and publication by retaining per-bin guards through counter
  reset; add a deterministic cross-thread regression for the interleaving.
- `moirai-core`: re-read the SPSC publication index after acquiring sender
  closure so a value published immediately before sender drop is drained rather
  than misreported as a closed empty channel.
- `moirai-parallel`: honor the explicit `Parallel` policy for small domains by
  partitioning indexed work across worker-plus-caller lanes. Adaptive policy
  remains the operation-level threshold owner; the executor no longer silently
  serializes expensive low-cardinality work.
- `moirai-executor`: keep nested indexed fan-out and map/reduce work-conserving
  by using the scheduler's worker help path while waiting. This prevents a
  saturated outer parallel region from parking every worker while inner indexed
  chunks remain queued.
- `moirai-executor`: flatten worker-nested indexed regions onto the current lane,
  preserving outer parallelism without recursively stealing outer jobs onto the
  worker stack.
- `moirai-executor`: track indexed-region depth on the participating caller so
  its nested fan-out and reductions also flatten onto the outer caller lane.
- `moirai-executor`: distribute indexed remainders across every selected lane
  instead of leaving worker lanes unused when the item count is just above the
  worker-plus-caller cap.
- `moirai-executor`: align panicking indexed-reduction metric coverage with the
  worker-plus-caller lane contract: two worker chunks complete scheduler
  lifecycles while the caller owns the third lane.
- `moirai-core`: read the thread-local operating-system error through
  `std::io::Error` on Unix and Windows, removing the Linux-only errno symbol
  that prevented IPC consumers from compiling on macOS.
- `moirai-pal`: keep the pointer-bearing kqueue output buffer thread-local, so
  the reactor remains `Send + Sync` without an unsafe marker implementation or
  a per-poll allocation.

[Version 0.3.0 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
