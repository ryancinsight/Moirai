### Changed

- **The executor releases finished-task state.** `HybridExecutor` now derives a
  `RetentionPolicy` from `CleanupConfig` and releases the task registry in whole
  blocks of 1,024 tasks as it registers new ones, once every task in a block has
  finished and either its newest completion is older than
  `task_retention_duration` or more than `max_retained_tasks` finished tasks are
  retained. Before this, completed state stayed resident for the executor's
  lifetime (about 73 bytes per task spawned) because nothing called the cleanup.
  `wait_for_task` on a released task still resolves and `cancel_task` is still a
  no-op; `task_status` and `task_stats` now return `None` for it. Set
  `enable_automatic_cleanup` to `false` to keep every task's metadata. The
  registry's `cleanup_completed` returns the number of blocks it released and no
  longer holds the directory write lock across its scan, and registration into
  an already-registered id now always panics.

- **The unified channel reports its configured capacity.** `UnifiedChannel` now
  runs on the workspace's one bounded MPMC queue core (ADR-0016) instead of a
  per-channel copy of the same ring that serialized each side behind its own
  mutex. The copy it replaced rounded the requested capacity up to a power of two
  and then wasted a slot, so a configured capacity of 2 held a single item and
  `capacity()` reported the padded value; the canonical core holds exactly the
  request and reports it back, so a channel configured for `n >= 2` now holds one
  more item than before. The send and receive fast paths are lock-free rather
  than two-mutex, and the channel's public API, FIFO order, and full-at-both-full
  contract are unchanged.

- **Worker idle-hook admission is bounded.** `register_idle_hook` now returns
  `Result<(), IdleHookRegistrationError>` and rejects registrations beyond
  `MAX_IDLE_HOOKS` without mutation. Snapshots copy a fixed array before
  callbacks run, preserving registration order and the owner-thread
  reclamation point without heap allocation.

- **Bounded Windows IME composition events.** The thread-owned native window
  now reports IME start, preedit, commit and cancellation phases with validated
  UTF-8 snapshots bounded by `MAX_COMPOSITION_UNITS`.

- **The GPU planner budget is constructed through `moirai-gpu`.**
  `KernelResourceBudget` is re-exported by the GPU facade and the occupancy
  planner consumes that same public type, preventing downstream providers
  from splitting the planner contract across Mnemosyne source revisions.

[Version 0.6.1 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
