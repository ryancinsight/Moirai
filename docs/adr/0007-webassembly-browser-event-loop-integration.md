# ADR-007: WebAssembly Browser Event-Loop Integration

Status: Accepted

**Date**: 2026-05-25
**Context**: Moirai's WASM target architecture must run reliably in standard web browsers where OS threads are unavailable, requiring integration with the browser's JavaScript event loop and cooperative task scheduling.

### Decision

1. **Cooperative Web Worker Event Loop**:
   - In WASM environments lacking native threading, Moirai's `WasmExecutor` cooperative mode runs directly inside the JavaScript thread or schedules work across Web Workers.
   - Tasks queue directly in Moirai's light internal queue. Tick dispatching cooperates with JS via `requestAnimationFrame`, `setTimeout`, or JS Microtasks (e.g., `Promise.resolve().then(...)`) to prevent blocking the browser's rendering thread.
2. **Browser Callback Ownership and Lifetime Management**:
   - WASM-JS boundaries use clear ownership patterns for closures and event listeners:
     - Event listener callbacks (e.g., for fetch, websockets, or timer events) are wrapped in Rust-managed types that automatically deallocate and unregister listeners when dropped.
     - Rust futures wait on JS Promises via thread-safe channels or local polling loops mapped to JS events.
3. **Event Queue Mutation**:
   - The event loop in WASM uses a lock-free or single-threaded cooperative queue. Interrupts and events from JS (such as I/O readiness, timers, or worker messages) write directly to the event queue and wake the Moirai reactor.
4. **Static and Dynamic WASM Verification**:
   - The workspace enforces WASM target compilation check via:
     `cargo check --target wasm32-unknown-unknown --all-features`
   - CI runs headless WASM tests using `wasm-pack test` or `cargo-diners` equivalents to guarantee event-loop correctness, callback deallocation, and task execution progress.

### Rationale

- **Browser Responsive Execution**: Cooperating with JavaScript's execution cycles ensures that Moirai applications do not cause page freezes or browser warnings.
- **Leak Prevention**: Explicit callback lifecycle tracking prevents memory accumulation at the boundary of WASM and JS runtimes.
- **Universal Portability**: Enables Moirai to compile cleanly for browsers, Node.js, and WASI hosts.

### Verification

- `cargo check --target wasm32-unknown-unknown --workspace`
- Automated test runs in simulated browser environments.

### Residual Risk

- Dynamic browser memory management and JS engine garbage collection cycles can introduce non-deterministic latency. Performance testing under browser load is necessary to isolate engine-specific variances.
