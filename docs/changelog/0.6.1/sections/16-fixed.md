### Fixed

- Remove obsolete nightly `thread_local` detection from `moirai-core`, which
  does not invoke Melinoe's TLS macro. The platform and executor crates retain
  their nightly `#[thread_local]` fast path.
- Preserve indexed fan-out and map-reduce work when bounded scheduler admission
  saturates by executing rejected chunks on the submitting caller under the
  same panic boundary as worker chunks. A monotonic diagnostic counter exposes
  each caller-run backpressure event.
- Reject zero-length and out-of-`off_t` Unix shared-memory mappings before
  acquiring an operating-system descriptor instead of truncating `usize`.
- Keep stack-owned scheduler scope state alive until the final completion
  releases its wait synchronization, preventing a multi-job scoped fan-out from
  hanging or dereferencing destroyed state.
- Reject a Unix `SharedMemory::open` whose requested size exceeds the existing
  segment. `mmap` accepts a length past the end of the object and leaves the
  surplus pages unbacked, so `as_slice` handed out bytes that raise `SIGBUS` on
  read; `open` now checks the segment with `fstat` first. Windows already
  refused the oversized view in `MapViewOfFile`.
- Detect a panicked pooled worker when joining `moirai-iter` fan-out. The join
  discarded the result of each `recv`, so a worker that unwound without sending
  left the channel disconnected and every later `recv` failed instantly — the
  join then returned as though all tasks had finished.
  `ZeroCopyParallelIter::map` consequently called `assume_init` over a slice no
  worker had written, and `for_each`, `reduce`, and the parallel sorts returned
  silently partial results. The join now counts completions and panics unless
  every task reported one.
- Stop the `moirai-iter` parallel sorts deadlocking on large inputs. The
  recursion forks one half onto the shared pool and blocks on it, so a forked
  half that forks again occupied a worker while depending on another; the pool
  does not steal work, so once every worker was blocked that way the queued
  halves had nobody left to run them and `par_sort` never returned. Reachable
  around `2048 * 2^(workers + 1)` elements — roughly a million on an eight-core
  machine. A fork budget now keeps at least one worker free to drain the queue,
  and the recursion sorts both halves in place once the budget is spent.
- Keep a `moirai-iter` pool worker alive when a job panics. The worker ran jobs
  without catching unwinds and workers are never replaced, so each panicking job
  removed one permanently; once all had gone, `execute` kept queueing onto a
  channel nobody received from and the next join blocked forever, turning a
  caller's panic into a later, unrelated hang. The panic still reaches the
  caller through the join, which sees the missing completion.
- Start at least one worker in `ThreadPool::new`. A zero-sized pool accepted
  jobs and ran none, so anything awaiting them waited indefinitely.
- Keep `CacheAlignedChunks` advancing for elements wider than a cache line. The
  chunk size was `(CACHE_LINE_SIZE / element_size) * (CACHE_CHUNK_SIZE /
  CACHE_LINE_SIZE)`, whose first term truncates to zero above 64 bytes, leaving
  the iterator yielding empty slices forever without advancing.
- Reserve the backing store of a Linux `SharedMemory::create` segment with
  `posix_fallocate`. `ftruncate` only sets the length of a tmpfs object and
  leaves its pages sparse, so a correctly sized segment still raised `SIGBUS`
  through `as_slice`/`as_mut_slice` when the store could not produce a page on
  first touch; the shortage now surfaces as an error at creation, and the failed
  segment is unlinked rather than left for a later `open` to map. Unix targets
  without `posix_fallocate`, macOS among them, are unchanged.
- Test the async task completion flag under the future lock rather than before
  acquiring it, so a second thread polling the same executor cannot pass the
  check, wait for the lock, and then poll a future that completed meanwhile —
  which panics with "resumed after completion".

