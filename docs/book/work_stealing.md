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

Themis topology identifies each logical processor's NUMA node. When the
construction-time worker assignment represents at least two distinct nodes,
Moirai first scans workers assigned to the same node and then falls back to the
complete worker ring. Absent, partial-one-node, and single-node assignments use
only the complete ring, avoiding a duplicate scan with no locality benefit.

The topology is snapshotted at runtime construction. A running scheduler does
not refresh that snapshot after processor hot-plug; constructing a new runtime
obtains a new snapshot. Worker assignments are advisory until the tracked
Themis binding seam enforces processor placement at worker startup.

## Benchmark Contracts

Moirai's benchmark suite includes a `thread_schedule_comparison` bench that
validates the work-stealing contract against equivalent Rayon workloads.
The benchmark passes only when per-task scheduling overhead stays below 1 µs
and linear core-count scaling holds up to the available logical core count.
