# ADR 0006: Async I/O Compatibility and Tokio Trait Integration

Status: Accepted

**Date**: 2026-05-25
**Context**: To provide a complete, low-overhead alternative to Tokio, Moirai needs a unified compatibility strategy for asynchronous I/O operations. This involves supporting or matching `AsyncRead`, `AsyncWrite`, and `AsyncBufRead` semantics, implementing a robust file readiness strategy, and ensuring strict cancellation safety and backpressure guarantees.

### Decision

1. **Trait Equivalence and Interoperability**:
   - Moirai defines `moirai_async::io::{AsyncRead, AsyncWrite, AsyncBufRead}` traits.
   - Interoperation with `tokio::io` is two transparent wrappers under the `tokio-compat` feature. `TokioCompat<T>` exposes a Moirai type through Tokio's traits and `MoiraiCompat<T>` exposes a Tokio type through Moirai's. Each wrapper is `#[repr(transparent)]` over `T`, with a compile-time size and alignment assertion, is built with `new` or `From<T>`, allocates nothing, and forwards every poll with the caller's `Context`, so the waker a caller registers is the waker the wrapped type stores and a re-poll under a new waker replaces it.
   - `tokio` is an optional dependency with the `io-util` feature only; the default build has no Tokio dependency.
   - Mapped in both directions: read, write, vectored write with `is_write_vectored`, flush, shutdown, and buffered reads (`poll_fill_buf`, `consume`). `TokioCompat` hands Tokio's `ReadBuf` to the Moirai reader as its initialized unfilled slice and advances by the returned count; `MoiraiCompat` wraps the caller's slice in a `ReadBuf` and reports the filled length.
   - Moirai's `AsyncWrite` carries defaulted `poll_write_vectored` (the first non-empty slice through `poll_write`) and `is_write_vectored` (`false`), so a writer without a scatter-gather primitive needs no code and a vectored one overrides both. Not mapped: `AsyncSeek`, because Moirai defines no async seek trait (`File::seek` is an inherent method); MOI-ASYNC-SEEK-001 tracks it.
2. **File Readiness and Blocking I/O Strategy**:
   - Since standard disk files do not support traditional poll-based readiness (e.g., via epoll/kqueue) on typical Unix platforms, Moirai implements a dual-path file readiness strategy:
     - **Cooperative Worker Offloading**: Standard disk file operations that would otherwise block are dispatched to the `BlockingTask` scheduler pool using `spawn_blocking` wrappers, ensuring that asynchronous worker threads remain free.
     - **Platform Native AIO/IOCP** (deferred, ADR 0014): completion-based file I/O through Windows IOCP or Linux io_uring is not built; every file operation runs on the blocking pool.
3. **Cancellation Safety Contracts**:
   - All async I/O futures (e.g., `Read`, `Write`, `Flush`) must be fully cancellation-safe. If an I/O future is dropped before completion:
     - The pending operation must not leave kernel-visible references to user buffers: pool jobs own their buffers, socket syscalls run inside `poll`, and a dropped readiness waiter retires its reactor registration (ADR 0014). A future completion backend cancels with `CancelIoEx` or io_uring cancellation and frees a buffer only after the original completion is reaped.
     - Shared buffer ownership is structured using zero-copy primitives or Rust's ownership model so that no buffer is leaked or left in an undefined state upon early drop.
4. **Backpressure and Resource Limits**:
   - Write streams must enforce backpressure by returning `Poll::Pending` when reactor write queues are saturated.
   - Flow control is mediated by a cooperative waker-registration scheme where writers are notified to wake only when the underlying socket or descriptor buffer has drained below a configured water-mark threshold.

### Rationale

- **Ecosystem Coexistence**: Transparent wrappers over the Tokio traits let Moirai coexist with Tokio-based libraries in mixed environments without polluting the core dependency tree.
- **Worker Isolation**: Keeping blocking file I/O separate from async task scheduling prevents CPU-bound tasks and async event loops from starving, matching Moirai's hybrid execution model goals.
- **Safety and Correctness**: Explicit cancellation semantics and buffer lifetime guarantees prevent memory corruption and resource leaks during future cancellation (e.g., under timeouts).

### Verification

- `moirai-async/src/io/compat/tests.rs` drives one 64 KiB payload through a 7-byte in-memory pipe natively, through `TokioCompat`, through a Tokio `duplex` behind `MoiraiCompat`, and through both wrappers stacked, and asserts identical bytes and that the pipe exerted backpressure.
- Counting wakers assert which task context a `Pending` read or write registers, that a re-poll replaces it, and that the peer's progress wakes only the latest one.
- EOF after shutdown, `BrokenPipe` after close, and zero-length reads and writes are asserted in both directions.
- `compat/tests/buffered.rs` compares the windows `poll_fill_buf` exposes, line reads through `read_until`, the registering context, and `consume` progress across the native and wrapped paths.
- `compat/tests/vectored.rs` compares vectored writes against a native gather and the `Vec<u8>` writer, asserts the default's first-non-empty-slice behavior, and asserts that `is_write_vectored` follows the wrapped writer.
- `io::tests` keeps the native-reader and Tokio-duplex value tests; `async_io_compat_comparison` measures the wrappers against native extension futures.

### Revision

2026-09-30: decision 1 replaces the `into_tokio()`/`from_tokio()` conversion sketch with the wrappers as built, and verification lists the tests that exist (MOI-TOKIO-IO-COMPAT-001).

### Residual Risk

- OS-specific differences in disk caching and non-blocking I/O support may result in varying file I/O latency profiles between platforms. Continuous empirical validation is required.
