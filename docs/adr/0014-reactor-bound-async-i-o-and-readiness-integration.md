# ADR-014: Reactor-Bound Async I/O and Readiness Integration

Status: Accepted

**Date**: 2026-05-25
**Context**: We needed to complete the transition from a cooperative/blocking async I/O simulation to a true event-driven, reactor-backed asynchronous I/O and execution architecture. The busy-polling loop in the async executor consumed excessive CPU, and file/socket operations lacked real readiness integration.

### Decision

1. **Reactor-Bound Event Loop**: Integrate a thread-safe `IoReactor` that manages OS-level handles (using `epoll` on Linux, `kqueue` on macOS, and readiness structures on Windows). Establish thread-local `ACTIVE_REACTOR` bindings.
2. **Readiness-Driven Sockets**: Implement non-blocking `AsyncTcpStream` and `AsyncTcpListener` in `moirai-pal::net` that register wakers with the `IoReactor` on `WouldBlock` errors and self-wake when no active reactor is present.
3. **Cooperative File Operations**: Build a clean `AsyncFile` abstraction in `moirai-pal::fs` that executes non-blocking read, write, seek, and flush operations, relying on a cooperative waker-yielding mechanism for safety.
4. **Executor Run-Queue Scheduling**: Replace the task-queue busy-polling loop in `moirai-async::executor::AsyncExecutor` with a thread-safe run-queue and block-on notification powered by a platform-specific `ExecutorWaker`.
5. **Clean Modular Delegation**: Decouple `moirai-async::net` and `moirai-async::fs` facades by delegating entirely to their `moirai-pal` counterparts, adhering to the 500-line structural limit.

### Rationale

- **High-Performance Event Dispatch**: Eliminates unnecessary polling loops, reducing CPU utilization of idle executors to zero.
- **Zero-Copy Readiness Integration**: Avoids buffer allocations and copies by delegating handle registration and waker updates directly to the platform reactor.
- **Progress Guarantee**: The fallback waker yield ensures that execution progresses even when an I/O reactor is absent or when operations are synchronous.
- **Strict Domain Boundaries**: Keeps platform-specific socket/file descriptors confined to `moirai-pal`, exposing clean traits and facades to `moirai-async`.

### Verification

- `cargo test -p moirai-pal --all-targets`
- `cargo test -p moirai-async --all-targets`
- `cargo test --workspace`
- `cargo bench -p moirai-benchmarks --test benchmark_contracts`

### Residual Risk

Platform-specific async file I/O (e.g., via io_uring or Windows IOCP) remains deferred in favor of cooperative standard-file abstractions. Future work must define thread-pool scheduling for file blocking operations if true non-blocking disk access is required under high load.
