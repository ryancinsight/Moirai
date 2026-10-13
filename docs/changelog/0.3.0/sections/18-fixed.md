### Fixed
- `moirai-async` TCP listener: `accept` no longer leaks a connection-pool
  reservation when the accept future is cancelled (dropped while pending). The
  reservation is held in an RAII guard released on every early exit, so a
  cancelled accept can no longer permanently exhaust `max_connections`.
- `moirai-async` `TcpStream::drop`: untrack the connection by the id captured at
  accept time instead of re-querying `peer_addr()`, which fails on an
  already-reset socket and leaked the pool slot plus the `active_connections`
  counter. Unique-id keying also removes the address-reuse collision that could
  undercount connections.
- `moirai-sync` `ShardedResourcePool::recycle`: reserve count/bytes before
  inserting so concurrent recyclers observe each other and evict on a fresh view.
  The prior load-decide-insert let N concurrent recyclers each skip eviction and
  overshoot the shard cap (and byte budget) by up to N-1.
- `moirai-core` `SharedQueue`: reject zero capacity (`% capacity` divide-by-zero),
  reject `capacity * size_of::<T>()` / total-size overflow (undersized mapping →
  out-of-bounds access) and over-aligned element types, and record capacity in
  the segment header so `open` validates it (a mismatched peer view is rejected
  rather than faulting on access).
- `moirai-async` `TimerWheel`: cancelling an already-fired or never-scheduled
  timer is now a no-op, fixing unbounded growth of the `cancelled` tombstone set
  (a membership index keeps the invariant `cancelled ⊆ active`).
- `moirai-core` `AdaptiveBatchSender::flush_batch`: bound consecutive no-progress
  retries and surface `WouldBlock` backpressure instead of spinning forever when a
  receiver stalls without closing.
- `moirai-core` hybrid SPSC channel: documented the `Send` soundness contract on
  the manual `unsafe impl`s and added a self-validating compile-time guard that
  turns any future `Clone` on `HybridSender`/`HybridReceiver` into a build error
  (a second producer/consumer would race the `!Sync` ring).
- `moirai-executor` thread scheduler: closed a producer/worker lost-wakeup. A
  parking worker performs `idle_workers` set (SeqCst) then `pending_tasks` load
  (SeqCst), but `schedule_job` incremented `pending_tasks` with `Release`, which
  is not in the SeqCst total order; the resulting store-buffer race allowed a
  worker to observe no work while the producer observed no idle worker, leaving
  a submitted task stranded until an unrelated submission. The increment is now
  `SeqCst` (free on x86, where `lock xadd` is already a full barrier) and stays
  ordered before the queue push so the `execute_job` decrement cannot underflow.
- `moirai-executor` thread scheduler: replaced the single-`AtomicU64` idle map
  with a multi-word `IdleBitset` so workers with id >= 64 are registered in and
  reachable by the wake lottery. On pools larger than 64 workers, high-index
  workers were previously invisible to the wake path and stayed parked under
  load (a throughput stall). Added `IdleBitset` unit tests plus a 100-worker
  multi-round end-to-end wake regression.
- `moirai-sync` `FutexMutex` (non-Linux/Windows fallback path): closed a lost
  wakeup in the condvar fallback. `unlock` released `locked` (Release) then read
  `waiters` (Relaxed); StoreLoad reordering let the `waiters` check float ahead
  of the release, so a concurrently-registering waiter could be skipped and left
  parked forever. Added paired `SeqCst` fences (waiter and unlocker) and a
  high-contention blocking-path regression test.
- `moirai-scheduler` NUMA `BackoffStrategy::record_failure`: use `saturating_mul`
  for the exponential delay so a pathologically large base delay cannot overflow
  (panic under `overflow-checks`).

