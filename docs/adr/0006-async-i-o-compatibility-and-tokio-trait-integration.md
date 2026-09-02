# ADR 0006: Async I/O Compatibility and Tokio Trait Integration

Status: Accepted

**Date**: 2026-05-25
**Context**: To provide a complete, low-overhead alternative to Tokio, Moirai needs a unified compatibility strategy for asynchronous I/O operations. This involves supporting or matching `AsyncRead`, `AsyncWrite`, and `AsyncBufRead` semantics, implementing a robust file readiness strategy, and ensuring strict cancellation safety and backpressure guarantees.

### Decision

1. **Trait Equivalence and Interoperability**:
   - Moirai defines `moirai_async::io::{AsyncRead, AsyncWrite, AsyncBufRead}` traits.
   - For ecosystem integration, Moirai provides feature-gated conversion shim layers (e.g., `into_tokio()` / `from_tokio()`) mapping Moirai's native I/O structures to `tokio::io` traits and vice-versa, avoiding any compile-time or runtime dependencies in the default build configuration.
2. **File Readiness and Blocking I/O Strategy**:
   - Since standard disk files do not support traditional poll-based readiness (e.g., via epoll/kqueue) on typical Unix platforms, Moirai implements a dual-path file readiness strategy:
     - **Cooperative Worker Offloading**: Standard disk file operations that would otherwise block are dispatched to the `BlockingTask` scheduler pool using `spawn_blocking` wrappers, ensuring that asynchronous worker threads remain free.
     - **Platform Native AIO/IOCP**: On platforms supporting true non-blocking file systems (such as Windows IOCP or Linux io_uring when enabled), Moirai registers the file handle directly with the `IoReactor` to receive completion notifications.
3. **Cancellation Safety Contracts**:
   - All async I/O futures (e.g., `Read`, `Write`, `Flush`) must be fully cancellation-safe. If an I/O future is dropped before completion:
     - The internal handle state must cleanly cancel the pending I/O operation (e.g., via `CancelIoEx` on Windows or cancellation queues in io_uring) to prevent dangling references to stack-allocated or heap-allocated user buffers.
     - Shared buffer ownership is structured using zero-copy primitives or Rust's ownership model so that no buffer is leaked or left in an undefined state upon early drop.
4. **Backpressure and Resource Limits**:
   - Write streams must enforce backpressure by returning `Poll::Pending` when reactor write queues are saturated.
   - Flow control is mediated by a cooperative waker-registration scheme where writers are notified to wake only when the underlying socket or descriptor buffer has drained below a configured water-mark threshold.

### Rationale

- **Ecosystem Coexistence**: Allowing clean shims for Tokio traits allows Moirai to serve as a drop-in replacement or coexist in mixed-library environments without polluting the core dependency tree.
- **Worker Isolation**: Keeping blocking file I/O separate from async task scheduling prevents CPU-bound tasks and async event loops from starving, matching Moirai's hybrid execution model goals.
- **Safety and Correctness**: Explicit cancellation semantics and buffer lifetime guarantees prevent memory corruption and resource leaks during future cancellation (e.g., under timeouts).

### Verification

- Comprehensive unit testing of read, write, seek, and cancel operations under simulated slow connections.
- Benchmark validation mapping throughput and latency against equivalent Tokio streams.
- Clippy and cargo checks verified on target files.

### Residual Risk

- OS-specific differences in disk caching and non-blocking I/O support may result in varying file I/O latency profiles between platforms. Continuous empirical validation is required.
