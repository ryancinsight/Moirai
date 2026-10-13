### Performance

- Enter the Chase-Lev resize gate once per batch steal instead of once per
  element. `steal_batch` moves up to sixteen items and previously paid a
  `SeqCst` increment and decrement on the counter every thief shares for each
  one, up to 32 contended read-modify-writes on one line per batch; it now pays
  two. Under a criterion harness with two, four, and eight thieves, batch drain
  of a pre-filled deque improves 23.2%, 12.0%, and 12.0%. The gate is
  consequently held longer, so `resize` — which spins until it is empty — can
  wait behind a whole batch: the owner's growth path under the same thief
  counts improves 19.6% at two thieves and shows no change at four or eight,
  against a same-code repeat drift of at most 6.2% on that host. Exactly-once
  transfer, the steal/pop ordering protocol, and every memory ordering are
  unchanged.
- Retain the 14-word inline scheduled-job capacity while removing forced
  cache-line alignment from each queue payload. A 256-slot worker injector now
  requests 36,864 bytes instead of 65,536 bytes on 64-bit targets; oversized
  and over-aligned closures continue through the existing typed boxed fallback.
  Apollo's exact pool-warmup probe drops from 1,857,224 to 1,169,112 retained
  bytes (37.1%) with no statistically significant retained-worker throughput
  regression.
- Size worker injectors from the process-wide admission bound instead of
  allocating 1024 slots per worker independently. With the default 8192-task
  bound and 24 workers, injector storage falls from 24 × 1024 to 24 × 256
  slots. Apollo's unchanged retained-allocation probe confirms the queue blocks
  fall from 24 × 262,144 bytes (6 MiB) to 24 × 65,536 bytes (1.5 MiB), while
  its warm forward transform remains allocation-free.
- Weaken 5 of the 11 `SeqCst` accesses on the MPMC waiter counters to
  `Relaxed`: the four deregistering `fetch_sub`es (over-counting costs only a
  spurious `notify_one`) and the two notify-gating loads whose paired queue
  operation happens under the channel mutex. The remaining six are annotated
  with the Store→Load edge they carry.
- Weaken the async executor's `running` flag from `SeqCst` to `Relaxed`
  stores plus one `Release`/`Acquire` shutdown edge, removing a full barrier
  from every iteration of the executor poll loop.

- Make the interleaved priority, resource-contention, and memory-ordering
  regressions deterministic across host topologies and concurrent test
  execution by using explicit worker pools, event gates, and value-checked
  task joins instead of wall-clock polling.
- Make the `ChaseLevDeque` retired-array reclamation poison-tolerant: resize,
  drop, and test observation recover the guarded pointer list via `into_inner()`
  instead of panicking on a poisoned mutex after a panicking lock holder.
  A regression poisons the lock, forces a further resize, drains all items
  exactly once, and verifies final destruction (MOI-DEQUE-POISON-215).
- Bind the `themis` crate alias to the renamed `themis-topology` package so
  fresh Git dependency resolution follows the provider identity.
- Bind the `mnemosyne` and `mnemosyne_core` crate aliases to packages
  `mnemosyne-memory` and `mnemosyne-memory-core` while preserving Rust imports.

