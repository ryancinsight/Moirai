### Fixed

- **The Win32 window state is shared without aliasing.** The window procedure
  wrote through a raw pointer derived from a temporary reborrow of the
  `Box<WindowState>` that `NativeWindow` also wrote through, held a
  `&mut WindowState` across `DefWindowProcW(WM_PRINT)` and `BeginPaint`, which
  deliver `WM_PRINTCLIENT` and `WM_ERASEBKGND` to the same procedure
  synchronously, and freed the state under a live window when `DestroyWindow`
  failed. The state is now one reference-counted cell lent mutably only to
  closures that make no native call, and the window owns its own count from
  `WM_NCCREATE` to `WM_NCDESTROY`. Painting moves the retained frame out for the
  GDI call and restores it afterwards.

- **A WebView2 host renders the page it hosts.** `WebViewHost::new` left the
  controller at empty bounds and, under a hidden window, not visible, so
  WebView2 held `capture_preview_png`'s completion until the finite wait
  expired unless a frame happened to be pending (4 of 55 runs alone, 30 of 60
  with three captures running at once), and a capture that did complete was
  1924x2100 rather than the window's client area. The controller now fills
  the client area and is visible, and a capture of a controller hidden
  through `set_visible(false)` fails at once with `InvalidInput` instead of
  waiting out the deadline.
- **A failed Windows frame allocation no longer leaves a frame that outruns its
  buffer.** `present_argb8888_region` stored the new frame's dimensions before
  reserving its pixels, so an allocation failure left a frame with dimensions
  and an empty buffer, and the next `WM_PAINT` passed it to `StretchDIBits`,
  which reads `width * height * 4` bytes. The old allocation is now taken out,
  resized, and reinstalled only once it holds the whole frame.

- **A thief no longer steals a slot the owner's fence-free pop already emptied.**
  A slot's state equals its index both while it holds an item and while it is
  free, and `pop` republishes the slot after moving its item out, so a thief
  that had read an older `bottom` could claim the slot and move the same item
  out a second time. x86-TSO leaves the delay before a `bottom` store reaches
  a thief unbounded, so the `bottom - top >= MAX_BATCH_STEAL` guard on the
  fence-free path made that unlikely but not impossible. The thief now re-reads
  `bottom` after its claim, which synchronizes with the owner's publish, and
  gives the slot back when `bottom` no longer covers its index. A Loom model
  of the slot-claim protocol reaches the double take with the fast path forced
  and passes with the re-read.

- **A scoped job dropped without running now fails its scope.** A scoped job's
  completion token released its registration on drop whether or not the job had
  run, so a job dropped after admission (never executed) let the scope return
  `Ok`, and `for_each_indexed` and `map_reduce_indexed` callers read result
  slots the job was to fill. The token now records that its job ran, and a scope
  that saw an unrun drop returns `SpawnFailed(Cancelled)` unless the drop was a
  refusal the submitter already answered with an inline run or its own error.
  No current path was found that drops an admitted job unrun; this closes the
  hole for the first one.

- **Async file operations no longer block the polling thread.** Every
  `moirai_async::fs` operation ran its file syscall inside `poll`, so a slow
  disk or network mount stalled the executor thread and a dropped future could
  not be cancelled. They now run on a bounded file-system pool: four workers
  behind a bounded queue and an async admission wait. It is the blocking
  facility the resolver already used, now one shared implementation. A handle
  keeps one stream operation in flight and settles it before the next, so
  stream order holds under cancellation. A dropped read's bytes go to the next
  read, and a write, seek, or sync submitted before its future dropped still
  takes effect. `AsyncWrite::poll_write` queues the write and reports its
  failure through the next operation. Before a handle's `metadata` or
  positioned read runs, every stream operation already queued on the handle
  finishes, so the observer sees earlier writes. `write_all` is one pool job,
  so it is written in full once submitted. Panic containment belongs to the
  pool and holds only in unwind builds.
- **`SlabAllocator` is `Sync` only for values that are `Send` and `Sync`.** It
  moves values in and out through `&self`, so sharing it across threads could
  drop a thread-affine value such as a `MutexGuard` on another thread.
  `CacheAlignedAllocator::allocate` multiplied element size by count without
  an overflow check, which in a release build wrapped into a short allocation
  reported as success; the size now comes from `Layout::array` and an
  overflowing count returns `None`.
- **`TcpStream::connect` no longer blocks the polling thread.**
  `moirai_pal::net::AsyncTcpStream::connect` ran std's blocking connect inside
  `poll`, so an unanswered SYN held the executor thread for the OS connect
  timeout (about 21 s on Windows, 127 s on Linux) and an outer `timeout` or a
  dropped future could not interrupt it. It now creates a non-blocking socket,
  starts the connect, and waits for writable readiness from the reactor,
  deciding the outcome from `SO_ERROR` and `getpeername`; dropping the future
  closes the half-open socket. `moirai_async`'s `TcpStream::connect`,
  `TcpListener::bind`, and `UdpSocket::bind` resolve hostnames on a fixed pool
  of four resolver threads behind a bounded queue and an async admission wait,
  instead of calling `getaddrinfo` in `poll` (literal addresses still parse in
  place). `connect` now tries every resolved address in order rather than
  only the first. On Windows, connect completion is decided with `select`,
  which reports a failed connect on every Winsock version, and a pending
  connect re-polls every 100 ms, because `WSAPoll` before Windows 10 version
  2004 never signals the failure. The re-poll registration ends when the
  connect settles or is dropped. A resolver or re-probe thread that fails to
  start surfaces the spawn error as `moirai_pal::thread::ThreadStartError` and
  is retried on the next call. The re-probe drops released wakers outside its
  registry lock. In unwind builds, a panic inside a resolver job (in
  `getaddrinfo` or in a caller's waker) fails only that lookup and leaves its
  worker serving. A re-probe waker that panics in its clone or wake is
  likewise contained. The containment does not apply under `panic = "abort"`,
  which the workspace release profile sets: there any such panic ends the
  process.
- **`ResultCell` hand-off is `unsafe` at its single-consumer edges.** `register`
  wrote the waiter cell with no exclusion, so two safe callers could race on it,
  and `StateWord` was a safe trait whose implementor could report `READY` before
  the result existed. `register` is now an `unsafe fn` whose caller vouches for
  the one consumer, `StateWord` is an `unsafe trait`, and the blocking and async
  handles carry the `SAFETY` argument at their call sites.

- Shut down and join compute workers already started when a later worker thread
  fails to spawn. Failed `ThreadScheduler` construction no longer leaves a
  partial worker set parked with retained scheduler state.
- Restore the Atlas source-hygiene ratchet without changing runtime behavior:
  channel StoreLoad handshakes now share one documented
  sequentially-consistent ordering policy, exact scheduler-route values replace
  presence-only assertions, and oversized SIMD benchmark, MPMC role, transport
  process-client, and SIMD scalar concerns live in dedicated modules.
- Publish scoped scheduler completion only after the borrowing task's call
  frame and captures are destroyed. The previous task-owned completion token
  could release the caller's stack state while the worker still held a shared
  borrow; Miri reproduced the invalid deallocation between task return and
  worker-frame teardown. Inline and typed-boxed jobs now share the ordered
  completion path, and dropped jobs release their token without running work.
  Directly scheduled and indexed panics reach failed-task metrics; batched
  scoped failures remain scope-local while the enclosing physical batch
  completes. The indexed allocation test keeps exact
  native allocation counts while using Miri's allocator for provenance-safe
  lifetime verification.
- **`ChaseLevStealer::capacity` no longer races reclamation.** Under
  `SharedEpochReclaim` it loaded the published array without the gate or reclaim
  guard a steal takes, so a grow followed by `try_reclaim_shared` could free the
  array between the load and the read. It now enters both, like a steal.

- Consume delivered socket readiness as a one-shot interest before waking its
  task. Independent read/write waiters remain armed, while completed writable
  registrations no longer spin the reactor or remain stale across raw socket
  reuse until a request deadline forces another poll. Native epoll, kqueue, and
  `WSAPoll` dispatch retain the polled registration generation through central
  consumption. A backend transition failure wakes delivered and independent
  waiters after unlocking, while central state mirrors the interest the backend
  reports as still armed instead of discarding a live registration. Kqueue
  collapses every sibling filter when an expected filter is already absent,
  and installs a replacement generation only after the prior lifecycle is
  wholly absent. Receipt changes are not replayed after `EINTR`, matching the
  native changelist-before-interruption contract.
- Prevent stale epoll, kqueue, and Windows `WSAPoll` results from deleting or
  waking a newer registration that reused the same raw descriptor value. Poll
  results carry a private registration generation while the public `Event`
  contract remains unchanged. Re-registering an existing descriptor installs a
  fresh generation, including when platform cleanup removed only its prior poll
  entry. Reused polling buffers still avoid per-iteration snapshot allocation.
- Keep task registry and metrics storage alive once through scheduler worker
  teardown, including re-entrant executor destruction, while standalone
  lifecycle tokens retain their dense block. Slot reclamation and task-ID reuse
  wait for token retirement without adding production per-task ownership traffic.
- **Indexed fan-out no longer reports `ShuttingDown` after chunks have run.** A
  shutdown landing between two chunk admissions returned `ShuttingDown` while
  earlier chunks were already executing, and `moirai-iter` answered by re-running
  every chunk on the caller. A mapper panic during that second pass dropped the
  outputs of an already-published chunk twice. The refused chunk now runs on the
  submitting lane, as a full queue already did, so `ShuttingDown` means no item
  was visited.

- Complete saturated async wakes inline after the first rejected scheduler
  admission instead of issuing 64 spin/yield retries. Repeated self-wakes use a
  bounded non-recursive requeue and surface persistent saturation as
  `TaskError::ResourceExhausted`. Lifecycle completion offsets also preserve
  `Instant`'s documented saturating behavior.
- Recover poisoned GPU buffer-pool mutexes instead of propagating a prior
  worker panic into every later pool operation.

- **Dropping a Unix socket or its future retires the reactor registration.**
  On epoll and kqueue a dropped read, write, accept, connect or datagram
  future, or a dropped `AsyncTcpStream`, left its descriptor registered with
  the task waker, so a later socket reusing the descriptor number inherited
  the stale interest and a readiness event dispatched after `close` failed
  the whole reactor with `EBADF`. Each pending operation now holds the
  per-interest cancellation Windows already used, and a removal that finds
  the descriptor closed retires the registration instead of failing.

- **`IoReactor::get_active` no longer hands out a `'static` reference to a
  scoped reactor.** It returned `&'static IoReactor` built from the pointer
  `with_active` installs, so safe code could return the reference out of the
  closure, drop the reactor, and call through the dangling reference. It is
  replaced by `IoReactor::with_current`, which passes the reactor to a
  higher-ranked closure so the reference cannot escape; the process-global
  reactor path is unchanged.

- Validate GPU buffer write and mapping ranges with checked arithmetic, so
  invalid offsets, bounds, and overflowing spans return typed validation
  errors before reaching wgpu.

- Close a lost wakeup in the bounded MPMC channel's receiver park path.
  `recv_bounded` re-checked the ring *before* registering in
  `receiver_waiter_count`, while `send_bounded` registered first; a producer
  that pushed in that window read zero waiters, skipped the notify, and left
  the receiver parked on a `Condvar` with no timeout, holding an item it could
  not see. Registration now precedes the re-check on both sides.
- **Shared-memory IPC enforces the single-writer and single-endpoint contracts
  it assumed.** `SharedMemory::create` now uses an exclusive create
  (`O_EXCL` / `ERROR_ALREADY_EXISTS`) and returns the new
  `IpcError::AlreadyExists`, where a second `create` on a live POSIX name used
  to `ftruncate` it and shrink every open mapping (`SIGBUS` from safe code) and
  reinitialise a `SharedQueue` header under live handles. `SharedQueue` claims
  its sender and receiver endpoints through header flags on first use, so a second
  sending or receiving handle is refused instead of racing on a slot; `send`
  returns the new `SendError` and `recv` returns `Result<Option<T>, IpcError>`
  (`IpcError::EndpointInUse`). The header is initialised and its capacity read
  with atomics. **Breaking:** `SharedMemory::as_slice` and `as_mut_slice` are
  `unsafe fn`, since a second handle to one name aliases the same pages through
  an address the borrow checker cannot relate. `IpcTransport` attaches through
  the exclusive create and gives the second handle on a name `Closed`. The
  `ipc_header` fuzz target is now `ipc_layout`; the header-byte parser it
  exercised no longer exists.

- Add the missing Store→Load barrier to the four lock-free notify paths
  (`send_bounded`, `recv_bounded`, `try_send`, `try_recv`). The queue write and
  the waiter-count read were separated only by a `SeqCst` load, which is an
  ordinary `mov` on x86-64 and orders nothing against a preceding store, so
  half of the Dekker pair the `SeqCst` counters exist for was absent.
  `loom_mpmc_waiter::notifier_without_the_store_load_barrier_loses_the_wakeup`
  enumerates the interleaving.
- **A scope body can no longer spawn a job that borrows a value local to it.**
  `SchedulerScope` and `moirai_parallel::Scope` were covariant in their
  lifetime, so `spawn` accepted a borrow of a body-local that the buffered job
  read after the body returned and dropped it. `SchedulerScope` is now
  invariant in `'scope`, and `moirai_parallel::Scope` takes a second lifetime
  (`Scope<'scope, 'env>`, as `std::thread::Scope` does) so tasks borrow the
  environment and nothing shorter. A worker of one scheduler opening a scope
  on another no longer indexes the second scheduler's worker table with its own
  worker id, and a panic that escapes the scope drain aborts instead of
  unwinding while scoped jobs still use the caller's frame. A panic payload
  whose `Drop` panics aborts instead of unwinding out of job execution.


