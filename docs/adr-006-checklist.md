# ADR-006 Implementation Checklist: Tokio Reactor-Native I/O Compatibility

This document defines the concrete contracts, specifications, and checklist items required to achieve reactor-native I/O compatibility matching the performance and safety semantics of Tokio.

## 1. OS-Specific Reactor-Native File Readiness and Completion

To provide low-overhead non-blocking I/O across platforms, the Moirai platform abstraction layer (PAL) must map OS-level asynchronous completion and readiness primitives directly to task wakers.

- [ ] **Windows IOCP Event Loop**:
  - [ ] Support completion-based asynchronous file and socket operations using Windows overlapped I/O.
  - [ ] Ensure `OVERLAPPED` structs are pinned in memory for the entire duration of the kernel operation.
  - [ ] Bind file and socket handles to the active IOCP completion port (`CreateIoCompletionPort`) exactly once.
  - [ ] Retrieve completed events in `PlatformReactor::poll_events` via `GetQueuedCompletionStatus`.
  - [ ] Map completion keys and overlapped pointers back to the waiting task's waker without thread contention or heap allocation in the poll loop.
- [ ] **Unix epoll (Linux) & kqueue (macOS/BSD) Event Loop**:
  - [ ] Map descriptors (`RawFd`) to platform interest flags (epoll `EPOLLIN`/`EPOLLOUT` with `EPOLLET` edge-triggered mode, kqueue `EVFILT_READ`/`EVFILT_WRITE`).
  - [ ] Register interest on read/write futures when socket operations return `WouldBlock`.
  - [ ] Wake the associated tasks when platform events are dispatched in `poll_events`.
  - [ ] Handle platform-specific events such as peer hangup (`EPOLLRDHUP` / `EV_EOF`) and error flags cleanly, translating them into standard Rust `io::ErrorKind` values.

## 2. Asynchronous I/O Cancellation and Memory Safety

Cancellation safety is a critical memory safety boundary when using asynchronous OS kernels. If a future is dropped before completing, the kernel must not write to or read from buffers that have been reclaimed or repurposed on the stack or heap.

- [ ] **IOCP / Overlapped I/O Cancellation (Windows)**:
  - [ ] Invoke `CancelIoEx` on drop for any pending overlapped operation associated with the handle.
  - [ ] Block the drop handler or defer buffer reclamation until the OS signals cancellation completion via the completion port (receiving `ERROR_OPERATION_ABORTED` or `STATUS_CANCELLED` from `GetQueuedCompletionStatus`).
  - [ ] Maintain an atomic tracking state indicating if a buffer is currently owned by the OS kernel to prevent double-frees or use-after-free conditions.
- [ ] **io_uring Cancellation (Linux)**:
  - [ ] Submit an asynchronous cancellation request (`IORING_OP_ASYNC_CANCEL`) matching the user data pointer of the target operation.
  - [ ] Defer buffer reclamation or reuse until the cancellation completion queue entry (CQE) is processed.
- [ ] **Reactor Unregistration**:
  - [ ] Ensure that closing or dropping an I/O type automatically unregisters the underlying descriptor from the active reactor.

## 3. Tokio Compatibility Wrappers and Trait Mapping

To allow Moirai types to integrate seamlessly with the broader Rust async ecosystem, shims must translate between Moirai and Tokio traits.

- [ ] **Tokio Trait Mapping**:
  - [ ] Implement `tokio::io::AsyncRead` and `tokio::io::AsyncWrite` for `TokioCompat<T>` wrappers under the `tokio-compat` feature.
  - [ ] Implement Moirai's native `AsyncRead` and `AsyncWrite` for `MoiraiCompat<T>` wrappers.
  - [ ] Ensure both compatibility wrappers are zero-overhead transparent newtypes (`#[repr(transparent)]`) with no runtime scheduling or allocation costs.
- [ ] **Readiness Polling Semantics**:
  - [ ] Map Tokio's poll-based read/write readiness checks (`poll_read_ready` / `poll_write_ready`) to Moirai's internal readiness state.
  - [ ] Wake Tokio-registered tasks via their associated `Context` waker when socket readiness transitions.

## 4. Verification and Benchmark Contracts

- [ ] **Correctness Tests**:
  - [ ] Write integration tests simulating network and file I/O cancellation under heavy loads (e.g. dropping 100,000 pending socket read futures).
  - [ ] Verify that no memory leaks, double-frees, or corrupted buffer reads occur after cancellation.
- [ ] **Benchmark Contracts**:
  - [ ] Quantify the overhead of reactor registration and unregistration.
  - [ ] Measure Moirai's native TCP loopback read/write throughput and latency directly against Tokio equivalents.
  - [ ] Add the following benchmark rows under `async_tcp_comparison`:
    - `async_tcp_loopback_echo_native`
    - `async_tcp_loopback_echo_tokio_compat`
