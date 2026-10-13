### Changed

- **Per-plane local queue capacity.** Each worker built four priority deques at
  the configured initial capacity, but a submission carries one priority, so a
  consumer that never sets one paid for four planes and used one. Only the
  default-priority plane now carries the configured capacity; the others start
  at `DequeCapacity::minimum()` and grow on the owner's push. Warm-pool local
  storage on a chunked-transform workload falls from 1,572,864 to 540,672 bytes
  at 24 workers. A plane that is used pays one owner-side growth and then
  performs identically. `DequeCapacity::minimum()` and `ChaseLevDeque::capacity`
  are new. ADR 0035 carries the dated revision.

[Version 0.6.1 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
