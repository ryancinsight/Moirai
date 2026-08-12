# Position in the Stack

## What Moirai Owns

Moirai is the Atlas concurrency and runtime layer. It owns:

- **The worker thread pool** — work-stealing scheduler for CPU-bound tasks
- **The async executor** — polling infrastructure for `Future`s and async tasks
- **The blocking pool** — isolated threads for synchronous I/O and FFI
- **Channel implementations** — SPSC, MPMC, Unified, and Select primitives
- **Parallel iterators** — `ParallelIteratorExt` with SIMD fast paths

Moirai does **not** own memory allocation policy (Mnemosyne), placement
vocabulary (Themis), tensor operations (Coeus), or domain physics.

## Where Moirai Sits

`	ext
themis (placement vocabulary)
  |
  v
moirai (runtime: scheduling, parallelism, channels)
  |               |
  v               v
helios         kwavers        CFDrs        ritk
`

## Themis Integration

- `WorkerId` identifies the worker responsible for a task
- `NumaNodeId` routes tasks to NUMA-local workers
- `CpuTopology` determines work-stealing domain boundaries

## Mnemosyne Integration

Mnemosyne's first-touch allocation policy requires memory to be touched on the
worker thread that will use it. Moirai satisfies this by pinning initialization
tasks to the target worker via `WorkerId`-based routing.

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Full runtime, channels, IPC |
| `ipc` | no | Shared-memory cross-process queues |
| `metrics` | no | Per-task performance counters |
