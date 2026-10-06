# ADR 0006: Async I/O Compatibility and Tokio Trait Integration

Status: Accepted

**Date**: 2026-05-25

**Revision 2026-09-30**: Rewritten to the as-built state audited under
MOI-NATIVE-REACTOR-001 (PR 570). The earlier text named conversion methods,
a file-readiness strategy, and a write-queue water-mark scheme that the source
does not contain; file and readiness decisions now live in ADR 0014 only.
The Tokio wrapper decision text follows MOI-TOKIO-IO-COMPAT-001.

## Context

Moirai provides its own asynchronous I/O traits and a facade for files and
sockets. Code written against Tokio's `AsyncRead`/`AsyncWrite` must be able to
call Moirai types and the reverse, without Tokio entering the default build.
Cancellation and backpressure must be defined for the facade because a dropped
future must not leave kernel-visible references to user memory.

## Decision

1. **Trait Equivalence and Interoperability**:
   - Moirai defines `moirai_async::io::{AsyncRead, AsyncWrite, AsyncBufRead}` traits.
   - Interoperation with `tokio::io` is two transparent wrappers under the `tokio-compat` feature. `TokioCompat<T>` exposes a Moirai type through Tokio's traits and `MoiraiCompat<T>` exposes a Tokio type through Moirai's. Each wrapper is `#[repr(transparent)]` over `T`, with a compile-time size and alignment assertion, is built with `new` or `From<T>`, allocates nothing, and forwards every poll with the caller's `Context`, so the waker a caller registers is the waker the wrapped type stores and a re-poll under a new waker replaces it.
   - `tokio` is an optional dependency with the `io-util` feature only; the default build has no Tokio dependency.
   - Mapped in both directions: read, write, vectored write with `is_write_vectored`, flush, shutdown, and buffered reads (`poll_fill_buf`, `consume`). `TokioCompat` hands Tokio's `ReadBuf` to the Moirai reader as its initialized unfilled slice and advances by the returned count; `MoiraiCompat` wraps the caller's slice in a `ReadBuf` and reports the filled length.
   - Moirai's `AsyncWrite` carries defaulted `poll_write_vectored` (the first non-empty slice through `poll_write`) and `is_write_vectored` (`false`), so a writer without a scatter-gather primitive needs no code and a vectored one overrides both. Not mapped: `AsyncSeek`, because Moirai defines no async seek trait (`File::seek` is an inherent method); MOI-ASYNC-SEEK-001 tracks it.
2. **Files and readiness.** ADR 0014 owns them: file syscalls run on a bounded
   blocking pool, sockets register wakers with the reactor after `WouldBlock`.
   Completion-based file I/O (Windows overlapped, io_uring) is not built;
   ADR 0067 states the Windows plan.
3. **Cancellation.** Every I/O future is safe to drop:
   - file jobs own their buffers, so a dropped future leaves the pool job
     running to completion with nothing borrowed from the caller;
   - socket syscalls execute inside `poll` and never outlive it;
   - a dropped readiness waiter retires its reactor registration (ADR 0014,
     decision 6);
   - a future completion backend frees a buffer or operation record only after
     the kernel completion is dequeued, cancelling with `CancelIoEx` or io_uring
     cancellation first (ADR 0067).
4. **Backpressure.** A write returns `Pending` when the socket reports
   `WouldBlock` and registers for writability; the bound is the kernel socket
   buffer. No reactor-side write queue or water mark exists. File operations
   are admitted through the blocking pool's bounded queue with an async wait.

## Rationale

- Transparent wrappers give both directions of interoperability at no runtime
  or scheduling cost and keep Tokio out of the default dependency graph.
- A single owner per concern (ADR 0014 for readiness and files) prevents this
  record and the reactor record from drifting.

## Verification

- `moirai-async/src/io/compat/tests.rs` drives one 64 KiB payload through a 7-byte in-memory pipe natively, through `TokioCompat`, through a Tokio `duplex` behind `MoiraiCompat`, and through both wrappers stacked, and asserts identical bytes and that the pipe exerted backpressure.
- Counting wakers assert which task context a `Pending` read or write registers, that a re-poll replaces it, and that the peer's progress wakes only the latest one.
- EOF after shutdown, `BrokenPipe` after close, and zero-length reads and writes are asserted in both directions.
- `compat/tests/buffered.rs` compares the windows `poll_fill_buf` exposes, line reads through `read_until`, the registering context, and `consume` progress across the native and wrapped paths.
- `compat/tests/vectored.rs` compares vectored writes against a native gather and the `Vec<u8>` writer, asserts the default's first-non-empty-slice behavior, and asserts that `is_write_vectored` follows the wrapped writer.
- `io::tests` keeps the native-reader and Tokio-duplex value tests; `async_io_compat_comparison` measures the wrappers against native extension futures.
- File offload and cancellation: `moirai-async/src/fs/tests/`.
- Socket readiness, drop, and loopback behavior: `moirai-async/src/net/tests/`
  and `moirai-pal/src/net/tests.rs`.

## Residual Risk

- Disk-cache and non-blocking behavior differ by platform, so file latency is
  platform-dependent; no cross-platform latency baseline is recorded.
- The Tokio comparison benchmarks cover the implemented facade only (ADR 0013).
