### Fixed
- `moirai-executor` scheduler: idle workers no longer redundantly drive the
  global `IoReactor` with a 1 ms `run_iteration` while waiting for work. That
  poll is not interruptible by `unpark` and rounds up to the OS timer
  granularity (~15 ms on Windows), so once a pool had parked, scheduling sync
  work to it stalled for ~15 ms — a latency the large `SPIN_LIMIT` (~6 ms of
  pre-park busy-spin) was masking. Idle workers now simply `park()`; async I/O
  readiness is driven by moirai-pal's dedicated global reactor thread, whose
  wakers reschedule their tasks through the same `schedule_job` path, so a
  parked worker is woken identically for async completions and fresh sync work.
  Measured submit→execute wake latency under intermittent load drops from
  ~15 ms to ~8 µs (8-worker pool, Windows). Verified by the new
  `spin_budget_bench` instrument.
- `moirai-pal` reactor: `IoReactor::get_active()` no longer panics if the
  process-global readiness reactor cannot be created or its driver thread cannot
  be spawned — it now caches and returns `None`, so socket operations degrade
  gracefully to the cooperative busy-poll self-wake fallback in `net.rs` instead
  of aborting. This makes that fallback (previously unreachable, since the old
  code always returned `Some` or panicked) a real, tested path; the async
  TCP/UDP self-wake-without-reactor round-trips are now value-verified.
- `moirai-sync::FutexMutex` (Linux): fixed a lost-wakeup **deadlock** in the
  three-state futex slow path. `lock_slow` acquired a woken-up contended lock via
  `CAS 0 -> 1`, erasing the "waiters present" marker (state 2) even when other
  waiters were still parked; the next `unlock` (which only `futex_wake`s when
  `swap(0)` observes 2) then skipped the wake, stranding those waiters forever.
  The slow path now acquires by `swap(2)` (Drepper / Rust-std algorithm), so the
  marker is conservatively preserved across hand-offs. Linux-only path;
  Windows/fallback path unchanged and value-verified.
- `moirai-async::Notify`: `notify_waiters()` no longer destroys a permit stored
  by a prior `notify_one()` issued with no waiters present. The two mechanisms
  are independent (matching `tokio::sync::Notify`); previously `notify_waiters`
  cleared the single-permit flag, so a subsequent `notified().await` could block
  forever. Regression test `notify_waiters_preserves_stored_notify_one_permit`.

