# ADR 0014: Reactor-Bound Async I/O and Readiness Integration

Status: Accepted

**Date**: 2026-05-25

**Revision 2026-09-16**: Windows `POLLNVAL` cleanup now crosses both ownership
stores. `WSAPoll` reports the removed registration generation to `IoReactor`,
which wakes and removes its central waiters. Re-registration after platform
removal starts a fresh interest set and wakes the retired waiters; central
generation identity prevents an older delayed invalidation from consuming a
newer invalidated generation for the same reused socket value.

**Revision 2026-09-16**: A driven event loop now treats a platform iteration
error as terminal. It retains the first `io::Error`, removes every central
waiter and Windows generation while registration is serialized, then wakes the
removed waiters after releasing every lock. The cached process-global reactor
remains discoverable so later socket polls receive an error whose source is the
retained failure. The driver does not retry, restart, or switch to cooperative
polling after terminal failure. This failure contract does not establish the
cause of a downstream HTTP timeout.

**Context**: We needed to complete the transition from a cooperative/blocking async I/O simulation to a true event-driven, reactor-backed asynchronous I/O and execution architecture. The busy-polling loop in the async executor consumed excessive CPU, and file/socket operations lacked real readiness integration.

### Decision

1. **Reactor-Bound Event Loop**: Integrate a thread-safe `IoReactor` that manages OS-level handles (using `epoll` on Linux, `kqueue` on macOS, and readiness structures on Windows). Establish thread-local `ACTIVE_REACTOR` bindings.
2. **Readiness-Driven Sockets**: Implement non-blocking `AsyncTcpStream` and `AsyncTcpListener` in `moirai-pal::net` that register wakers with the `IoReactor` on `WouldBlock` errors and self-wake when no active reactor is present.
3. **Cooperative File Operations**: Build a clean `AsyncFile` abstraction in `moirai-pal::fs` that executes non-blocking read, write, seek, and flush operations, relying on a cooperative waker-yielding mechanism for safety.
4. **Executor Run-Queue Scheduling**: Replace the task-queue busy-polling loop in `moirai-async::executor::AsyncExecutor` with a thread-safe run-queue and block-on notification powered by a platform-specific `ExecutorWaker`.
5. **Clean Modular Delegation**: Decouple `moirai-async::net` and `moirai-async::fs` facades by delegating entirely to their `moirai-pal` counterparts, adhering to the 500-line structural limit.
6. **Generation-Bound Windows Cleanup**: Treat `POLLNVAL` as a generation-tagged invalidation. Remove its platform registration, then wake and remove the corresponding central waiters only while no replacement generation exists.
7. **Terminal Driver Failure**: Retain the first error returned by a driven platform iteration. Serialize failure publication with descriptor and waiter registration, remove all central waiters and Windows generations, wake those waiters outside locks, and reject later registrations with the retained error as their source. A direct `run_iteration` call remains caller-owned; normal `stop` and an attempted second `run` do not publish terminal platform failure.

### Rationale

- **High-Performance Event Dispatch**: Eliminates unnecessary polling loops, reducing CPU utilization of idle executors to zero.
- **Zero-Copy Readiness Integration**: Avoids buffer allocations and copies by delegating handle registration and waker updates directly to the platform reactor.
- **Progress Guarantee**: The fallback waker yield ensures that execution progresses even when an I/O reactor is absent or when operations are synchronous.
- **Strict Domain Boundaries**: Keeps platform-specific socket/file descriptors confined to `moirai-pal`, exposing clean traits and facades to `moirai-async`.

### Verification

- `cargo nextest run --locked -p moirai-pal`
- `cargo nextest run --locked -p moirai-async`
- `cargo nextest run --locked --workspace`
- `cargo bench -p moirai-benchmarks --test benchmark_contracts`

### Residual Risk

Platform-specific async file I/O (e.g., via io_uring or Windows IOCP) remains deferred in favor of cooperative standard-file abstractions. Future work must define thread-pool scheduling for file blocking operations if true non-blocking disk access is required under high load.
