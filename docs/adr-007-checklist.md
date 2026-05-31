# ADR-007 Implementation Checklist: WebAssembly Browser Event-Loop Integration

This document defines the concrete contracts, specifications, and checklist items required to achieve WebAssembly browser event-loop integration.

## 1. Cooperative Browser Event-Loop Integration

In single-threaded WebAssembly environments (without native threading or shared memory), the Moirai reactor must cooperate with the browser's execution thread to execute async tasks without freezing the user interface.

- [ ] **Cooperative Microtask Execution**:
  - [ ] Implement `WasmExecutor` to dispatch tasks using microtasks (`Promise.resolve().then`) via `wasm-bindgen-futures::spawn_local` for immediate scheduling.
  - [ ] Avoid spinning or executing long loops that block the browser's main thread for more than 16 milliseconds per frame.
- [ ] **Macro-task Yielding**:
  - [ ] Yield execution back to the browser's event loop when the task budget is exceeded.
  - [ ] Use `setTimeout(callback, 0)` or `requestIdleCallback` to schedule the next batch of background tasks, allowing the browser to render frames and handle user inputs.
- [ ] **UI-Bound Frame Scheduling**:
  - [ ] Integrate with `requestAnimationFrame` for rendering-dependent tasks, synchronizing execution with the browser's refresh rate.

## 2. Multi-threaded Web Worker Scheduling

On browser platforms supporting WebAssembly threads (with `SharedArrayBuffer` and atomics), Moirai can schedule tasks across multiple Web Workers.

- [ ] **Web Worker Spawning**:
  - [ ] Initialize and spawn Web Workers dynamically from the main WASM module.
  - [ ] Distribute a shared WebAssembly memory instance (`SharedArrayBuffer`) across all workers.
- [ ] **Message Routing and Stealing**:
  - [ ] Map Moirai's work-stealing scheduler to Web Workers using atomic operations for queue sync.
  - [ ] Route tasks via `postMessage` or synchronized ring buffers in shared memory.
  - [ ] Ensure that worker worker pools block on atomics using browser-native futex-equivalent operations (`Atomics.wait` / `Atomics.notify`) without consuming 100% CPU on the main thread.

## 3. JS Callback Ownership and Lifetime Management

Interoperating with JS APIs (timers, event listeners, fetch) requires strict callback lifecycle management to prevent memory leaks in WASM.

- [ ] **Callback Lifetime Tracking**:
  - [ ] Wrap `wasm_bindgen::prelude::Closure` callbacks in Rust wrapper structs that implement `Drop`.
  - [ ] Automatically unregister JS event listeners and drop closures when the corresponding Rust handle or future is dropped.
- [ ] **Promise-to-Future Conversion**:
  - [ ] Map JS Promises to Rust Futures cleanly using thread-safe event queues.
  - [ ] Wake the waiting Rust task when the JS promise resolves or rejects.

## 4. Headless Browser Verification

- [ ] **Headless Test Suite**:
  - [ ] Run the WASM verification tests in headless browser environments:
    ```powershell
    wasm-pack test --headless --chrome --firefox
    ```
  - [ ] Verify that tasks progress, event wakers trigger, and callback cleanups occur without leaking JS objects.
