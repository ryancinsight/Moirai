# Concurrency safety audit — residual risk & decision log

Append-only log of adversarial concurrency/memory-safety audit rounds. Each
round records what was fixed, what was investigated and found sound (so it is
not re-chased), and real-but-deferred items.

## Round 11 (2026-06-23)

### Fixed (this round)
- **Executor producer/worker lost-wakeup** — `schedule_job` incremented
  `pending_tasks` with `Release`, breaking the SeqCst store-buffer handshake
  with a parking worker (`idle_workers` set + `pending_tasks` load, both SeqCst).
  Now `SeqCst`, kept before the queue push (so `execute_job`'s decrement cannot
  underflow). `core.rs`.
- **Executor id≥64 wake stall** — single `AtomicU64` idle map could not address
  workers beyond bit 63; on >64-worker pools they were unreachable by the wake
  lottery and stayed parked under load. Replaced with multi-word `IdleBitset`
  (`idle.rs`); every worker registers (unified `wait_for_work`).
- **`FutexMutex` non-Linux/Windows fallback lost-wakeup** — `unlock` stored
  `locked` (Release) then read `waiters` (Relaxed); StoreLoad reorder skipped a
  concurrently-registering waiter. Added paired `SeqCst` fences.
- **NUMA backoff overflow** — `record_failure` exponential delay now
  `saturating_mul`.

### Investigated and found SOUND (do not re-chase)
- **`moirai-scheduler/deque/chase_lev.rs` `steal_batch_with`** — claimed
  double-free on CAS failure. FALSE: on failure the speculative `MaybeUninit`
  copies are never `assume_init_read`, so no Drop runs (mirrors the single-item
  `mem::forget` discipline). Correct.
- **`chase_lev.rs` `QuiescentReclaim` retired-array UAF** — FALSE: `reclaim_memory`
  and `Drop` take `&mut self`, so Rust aliasing already enforces stealer
  quiescence; retired arrays are only freed when no `&self` stealer can run.
- **`moirai-core/scheduler/deque.rs` grow/Drop double-free** — claimed the
  retired buffer's moved-out slots get re-dropped. FALSE: `Buffer::Drop` only
  deallocates and never drops elements; the live `[top,bottom)` range is dropped
  exactly once from the current buffer. Retired buffers carry no element drops.

### Real but DEFERRED (next round candidates, ranked)
- `moirai-async/net/listener.rs` `accept`: connection reservation
  (`try_reserve`) leaks if the accept future is cancelled before commit — wrap in
  an RAII guard that releases on drop, disarmed only after `add_connection`.
- `moirai-async/net/stream.rs` `TcpStream::drop`: pool entry / `active_connections`
  leak when `peer_addr()` fails at drop (abrupt reset). Store the peer addr at
  construction; key the pool on a unique id, not `SocketAddr`.
- `moirai-sync/resource_pool.rs` `recycle`: shard-wide capacity invariant enforced
  with bin-local locks → concurrent recyclers overshoot the configured cap
  (retention beyond budget). Reserve count atomically before insert, or hold a
  shard-level critical section over decide+insert.
- `moirai-core/ipc/{queue,memory}.rs`: `capacity * size_of::<T>()` overflow,
  `meta_size` vs `align_of::<T>()` assumption, and `munmap(ptr, caller_size)`
  using a possibly-wrong length. Bound `T: bytemuck::Pod`, `checked_mul`, store
  and unmap the exact mapping length. (Cross-process/untrusted-input hardening.)
- `moirai-async/timer/wheel.rs`: `cancel` inserts a tombstone into `cancelled`
  for already-fired ids → unbounded growth. Latent (production timer path is
  `driver.rs`/`registration.rs`; `TimerWheel` is currently test-only).
- `moirai-core/communication/zero_copy/adaptive.rs` `flush_batch`: unbounded spin
  on persistent `Full`/`WouldBlock` ignoring a stalled-but-open receiver (liveness).
- `moirai-core/channel/hybrid.rs` + `communication/ring_buffer.rs`: manual
  `unsafe impl Send` on the hybrid halves suppresses the auto-trait check that a
  `!Sync` SPSC ring relies on; sound today only because neither half is `Clone`.
  Encode the SPSC invariant at the type level (one `#[derive(Clone)]` from UB).
