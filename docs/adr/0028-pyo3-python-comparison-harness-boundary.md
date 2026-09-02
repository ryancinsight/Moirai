# ADR 0028: PyO3 Python Comparison Harness Boundary

Status: Accepted

**Date**: 2026-05-23
**Context**: Python runtime bindings for native Moirai execution

### Decision

`moirai-python` is a PyO3 binding crate over `moirai::Moirai`. It is a workspace crate and Python package, but it is not a dependency of the Rust runtime crates. The binding crate does not own scheduler, planner, backend logic, workload kernels, or comparison harnesses; it forwards to the Rust `moirai` crate. Python code provides the runtime facade and lifecycle tests.

### Rationale

- Preserves Rust runtime dependency boundaries while exposing `moirai::Moirai` to Python.
- Keeps one authoritative execution path through the Rust `moirai` crate instead of a Python or binding-crate stand-in.
- Separates FFI registration, facade code, and lifecycle tests.
- Prevents benchmark-specific Python functions from becoming public wrappers unless they correspond to comparable joblib or Tokio runtime primitives.

### Verification

- `cargo test -p moirai-python`
- `cargo clippy -p moirai-python -- -D warnings`
- `py -3.13 -m unittest discover moirai-python\tests`

### Residual Risk

Scoped completion-only ready work now exceeds the Tokio/Rayon scope and spawn baselines in scheduler-focused, industry-style, and single scoped-job targets. Indexed map/reduce exceeds Rayon indexed at 64, 256, and 1024 ready items. `Moirai::join` drains all work visible before quiescence without shutting down worker threads; work submitted after quiescence is a later batch. Public result-bearing `spawn_fn` and `spawn_async` no longer use mutexed result storage, condvar work notifications, condvar completion wakes, waiter-mutex registration, READY/park-racy waiter registration, per-task lifecycle `Arc` allocation, duplicate task-local timing for metrics, dynamic future dispatch in the async result path, boxed future pinning in the async result path, lifecycle mutexes in async polling, wrapper waker allocation, heap allocation for common small scheduled jobs, `Box<dyn FnOnce>` dispatch for oversized scheduled jobs, or a separate raw-pointer heap job variant for oversized scheduled jobs. The standalone async executor queue uses monomorphized poll/drop function pointers instead of `dyn Future` queue dispatch, and its handles use inline atomic result/waker slots instead of mutexed result storage and global waker hash maps. They remain a separate diagnostic category because each logical task still owns a result slot and Rayon `scope` is not result-handle equivalent. The public result-handle diagnostic includes real Tokio `JoinHandle` rows and measures Moirai ahead on the equivalent ready, captured-ready, oversized-captured, async-ready, and wake-once result-handle paths; it also includes a direct Moirai `scope` row ahead of Rayon's scoped completion row. The public-handle Criterion timeout was isolated to plot/report generation and closed by disabling plots in that target. A raw-pointer two-endpoint result slot was rejected after earlier stress variants reproduced a join hang and the latest targeted variant regressed `task_scheduling_overhead` to 633.01-640.02 ns; relaxed lifecycle metadata atomics were rejected after `task_scheduling_overhead` regressed to 608.31-641.98 ns; duplicate worker identity removal was rejected after the scheduling gate failed to retain an improvement; production QPC lifecycle timing was rejected after the public oversized-capture path and an earlier scheduling gate regressed; a larger spin threshold was rejected after no statistically significant improvement; an unconditional load-before-CAS result take path, per-task metrics timestamp removal, public `spawn_fn` routing through `SyncTask`, and per-worker running-bit wake suppression were rejected after benchmark regressions or no improvement. The retained result wait path keeps the already-ready claim as one direct CAS and uses a monomorphized zero-sized policy for load-gated pending spins. Direct result-slot diagnostics now show same-thread slot completion below 50 ns, so the remaining public result-handle work moves to scheduler wake/result-handoff variance, async wake/requeue locality, and registry lifecycle bookkeeping rather than result-slot pooling. Transport safe-channel receives now avoid owned deserialization for archived `String` payloads while preserving malformed-input rejection. Active competitive batch targets keep public-handle rows separate from scoped and indexed batch rows so value semantics remain explicit.

This ADR establishes the foundational architectural principles that guide all implementation decisions in the Moirai concurrency library, ensuring consistency with the project's vision of being a complete alternative to existing fragmented concurrency solutions.
