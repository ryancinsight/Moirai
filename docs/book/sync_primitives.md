# Sync Primitives

## `CacheAligned<T>`

`CacheAligned<T>` pads and aligns `T` to a hardware cache line (64 bytes on
x86-64). All channel counters and task queue heads/tails use `CacheAligned`
to prevent false sharing.

## Communication Patterns

`moirai-core::communication` provides higher-level patterns:

| Pattern | Type | Description |
|---------|------|-------------|
| Broadcast | `Broadcast<T>` | One sender, many receivers |
| Publish-subscribe | `PubSub<T>` | Topic-routed message fan-out |
| Collective ops | `CollectiveOps` | Barrier, reduce, all-reduce |
| Ring buffer | `RingBuffer<T>` | Fixed-capacity circular queue |
| Router | `Router<T>` | Content-based message dispatch |

## Event-Gated Scheduling

A task waiting for an event parks without spinning; the scheduler wakes it
directly when the event fires. Value-checked joins verify woken tasks receive
the expected result.

## Memory Pools

| Pool | Description |
|------|-------------|
| `SlabPool<T>` | Fixed-size slab with free-list |
| `StackPool<T>` | LIFO stack of recycled objects |
| `ThreadLocalPool<T>` | Per-thread wrapper, no lock overhead |
