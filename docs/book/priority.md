# Priority Scheduling

Moirai's scheduler uses four-level priority queuing to order task dispatch.

## Priority Levels

```rust,ignore
pub enum Priority {
    Low      = 0,  // background work
    Normal   = 1,  // default
    High     = 2,  // interactive/latency-sensitive
    Critical = 3,  // system-level
}
```

`Priority::index()` returns the numeric level used by the priority-partitioned
run queue.

## Setting Task Priority

```rust,ignore
use moirai_core::{TaskBuilder, Priority};

let handle = runtime.spawn_fn_with(
    TaskBuilder::new().priority(Priority::High),
    || expensive_computation(),
);
```

## Scheduling Semantics

Workers pop from the highest non-empty priority bucket first. Within a bucket,
FIFO order is maintained. Themis `WorkerId` and `NumaNodeId` are factored into
routing: a `Critical` task goes to the worker with the shortest queue on the
caller's NUMA node.
