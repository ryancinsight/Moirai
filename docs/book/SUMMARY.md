# Summary

[Introduction](README.md)

# Part I — Task Model

- [1. Tasks and the Runtime](runtime.md)
- [2. Work-Stealing Scheduler](work_stealing.md)
  - [Example: Task Spawning](examples/task_spawning.md)
- [3. Task Handles and Results](task_handles.md)
- [4. Priority and Scheduling Hints](priority.md)

# Part II — Execution Modes

- [5. Parallel Tasks](parallel.md)
  - [Example: Parallel Reduce](examples/parallel_reduce.md)
- [6. Async Tasks](async_tasks.md)
- [7. Blocking Tasks](blocking.md)

# Part III — Synchronization

- [8. Channels](channels.md)
- [9. Barriers and Mutexes](sync_primitives.md)

# Part IV — The Atlas Stack

- [10. Position in the Stack](stack_position.md)


# Part IV — Routing and Transport

- [Transports and their capability contract](transports.md)
- [Payload framing and ownership regions](payloads.md)
- [Safe channels: typed endpoints over raw links](safe-channels.md)
- [Routes: from scheduler decision to wire address](routes.md)
- [The message router: topic fan-out over transports](router.md)
