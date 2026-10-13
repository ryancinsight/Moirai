### Fixed
- `moirai-pal` I/O reactor lost-edge hang (Linux/BSD): the epoll/kqueue backends
  registered fds **edge-triggered** (`EPOLLET` / `EV_CLEAR`), but the reactor
  registers a waker only *after* a `WouldBlock`, so readiness arriving in the
  register window — or already present at registration — emitted no edge and the
  task hung forever. Both backends are now **level-triggered**, which re-reports
  readiness on every wait until the I/O consumes it, self-healing the race (and
  the interest-widening `unregister`+`register` window). On Windows `get_active()`
  returns `None`, so the live platform uses the cooperative self-wake fallback and
  was unaffected; this is documented at the `get_active` site (a readiness-capable
  Windows/IOCP reactor remains a separate, larger feature).
- `moirai-pal` epoll/kqueue timeout truncation: saturate the `as_millis()`→`c_int`
  and `as_secs()`→`time_t` casts so a multi-week/century timeout cannot wrap to a
  negative value (turning a finite wait into an infinite block).
- `moirai-pal` `IoReactor::with_active`: restore the previous thread-local reactor
  via RAII so a panic in the closure cannot leave a dangling reactor pointer in
  the thread-local that a later `get_active()` would dereference (use-after-free
  on unwind).
- `moirai-pal` `Timer::new`: clamp the duration and use `checked_add` so a
  near-`Duration::MAX` timeout no longer panics the deadline computation
  (`Instant::now() + duration` overflow).
- `moirai-iter` `NumaAllocator`: on Linux, `numa_alloc` returns null on `mmap`
  failure instead of falling back to the global allocator, so every non-null
  pointer is `mmap`-backed and `numa_free`'s `munmap` is never applied to a heap
  pointer (which was undefined behavior).
- `moirai-utils` `prefetch_range_read`: saturating/checked address arithmetic so
  a range ending near `usize::MAX` cannot overflow (panic under `overflow-checks`).
- `moirai-async` `io::compat`: feature-gate the `tokio-compat`-only imports so the
  crate is warning-clean without the feature.

