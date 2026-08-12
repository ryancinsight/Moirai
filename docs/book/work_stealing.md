# Work Stealing

Work stealing is Moirai's primary load-balancing mechanism for CPU-bound tasks.
Each worker thread owns a local deque; idle workers steal tasks from the tail
of a busy peer's deque.

## How It Works

1. A spawned task is pushed onto the spawning worker's local deque.
2. The worker pops from the head of its own deque (LIFO for cache locality).
3. An idle worker scans peer deques and steals from a victim's tail (FIFO).

This combination gives good cache locality for self-contained computation
chains while keeping all cores busy during uneven workloads.

## NUMA-Aware Stealing

Themis `NumaNodeId` and `WorkerId` identify each worker's NUMA home. Moirai
prefers to steal from workers on the same NUMA node, escalating to cross-node
stealing only when the local domain is empty. This reduces remote-memory
latency on multi-socket systems.

The topology is snapshotted at runtime start from `CpuTopology::detect()` and
invalidated when `TopologyEpoch` advances (hot-plug events).

## Benchmark Contracts

Moirai's benchmark suite includes a `thread_schedule_comparison` bench that
validates the work-stealing contract against equivalent Rayon workloads.
The benchmark passes only when per-task scheduling overhead stays below 1 µs
and linear core-count scaling holds up to the available logical core count.
