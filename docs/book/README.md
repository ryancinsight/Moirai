# moirai — Hybrid Concurrency for Atlas

`moirai` is the scheduling and parallelism layer of the Atlas stack.  It
provides a work-stealing hybrid async+parallel runtime that replaces `rayon`
and `tokio` for Atlas consumers.

## Design goals

- **Hybrid execution** — async futures and CPU-bound closures share one
  scheduler.  No explicit thread-pool boundaries.
- **Work-stealing** — idle workers steal tasks from busy workers' local queues,
  keeping all cores busy without manual partitioning.
- **Priority hints** — `Priority::High` / `Low` / `Normal` let physics kernels
  and I/O tasks share one pool without a separate executor per concern.
- **NUMA awareness** — the `numa` feature routes tasks to workers on the same
  NUMA node as their input data.
- **Atlas integration** — `mnemosyne` provides the allocator; `themis` provides
  the topology; `melinoe` provides branded thread-local state.

## What this book covers

1. Creating a `Moirai` runtime and spawning tasks.
2. The work-stealing scheduler and task lifecycle.
3. `TaskHandle<T>` and result collection.
4. Priority hints.
5. Parallel closures and data-parallel reduction patterns.
6. Async tasks and `block_on`.
7. Blocking tasks for I/O and FFI.
8. Channels, barriers, and mutexes.
9. Where moirai fits in the Atlas stack.


## Part IV — Routing and transport

Part IV extends this book from one process to many. Its chapters:

| Chapter | Source of truth | Status |
| --- | --- | --- |
| [Transports and their capability contract](transports.md) | `transport.rs`, `network.rs`, `process.rs` | written |
| [Payload framing and ownership regions](payloads.md) | `payload.rs` | written |
| [Safe channels: typed endpoints over raw links](safe-channels.md) | `safe_channel.rs` | written |
| Routes: from scheduler decision to wire address | `route.rs` | planned |
| The router: dispatch, retries, backpressure | `router.rs` | planned |
| Remote tasks: capabilities and server lifecycle | `remote_task/` | planned |

Planned entries gain links as their teaching content lands under
`MOI-AUDIT-DOC-009`; no placeholder pages.
