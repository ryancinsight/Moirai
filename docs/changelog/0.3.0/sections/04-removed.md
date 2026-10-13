### Removed
- **Breaking:** removed the unconsumed `moirai_iter::numa` helper and its
  Rayon comparison benchmark. Placement belongs to Themis, allocation belongs
  to Mnemosyne, and data-parallel execution belongs to the scheduler-backed
  `moirai-parallel` surface; the deleted iterator helper applied none of those
  provider contracts.
- **Breaking:** removed combined owner/thief deque capabilities,
  `QuiescentReclaim`, explicit default-policy `reclaim_memory`, callback
  `steal_batch_with`, and the unconsumed `BlockBasedDeque`. `ChaseLevDeque` is
  the canonical unique bottom-side owner; callers clone `ChaseLevStealer` for
  top-side access. Batch steals return an owning `StolenBatch` iterator.
- **Breaking:** removed the obsolete `moirai/no-global-alloc` no-op feature.
  The library no longer registers a global allocator; final binaries own that
  process-wide choice, while `mnemosyne` continues to forward provider
  integration into core and executor crates.
- **Breaking:** removed the dead, mis-shaped passive scheduler abstraction from
  `moirai-core`: the `Scheduler` trait, the `ScheduledTask` erased-task type
  (+ `INLINE_SCHEDULED_TASK_WORDS`), and the `SchedulerConfig`/
  `WorkStealingStrategy`/`QueueType`/`StealContext`/`Stats` config vocabulary.
  None had a production consumer — `ThreadScheduler` is active (owns workers and
  executes internally), so the passive `next_task`/`steal_task` contract could
  only ever be a mock. The live erased-task type is the executor's `ScheduledJob`
  and the canonical abstraction is the new `WorkScheduler` seam. `SchedulerId`
  (the one live export, used by metrics aggregation) is retained.
- **Breaking:** redundant work-stealing scheduler implementations consolidated
  onto the canonical runtime scheduler (`moirai_executor::ThreadScheduler`,
  which already executes every `spawn*`/`block_on`/`scope`). Removed
  `moirai_scheduler::{WorkStealingScheduler, WorkStealingCoordinator,
  SchedulerStats, SchedulerStatsSnapshot}`, `moirai_scheduler::numa_scheduler::
  {NumaAwareScheduler, NumaSchedulerStats, NumaSchedulerError, StealStatistics}`,
  the generic `moirai_core::scheduler::WorkStealingCoordinator`, and the
  `moirai::WorkStealingScheduler` re-export. None were on any runtime code path.
  `moirai-scheduler` is now a primitives crate: the lock-free deques
  (`ChaseLevDeque`/`SplitDeque`) and NUMA primitives
  (`CpuTopology`, `AdaptiveBackoff`) remain its canonical surface. Migration:
  construct the runtime via `moirai::Moirai`/the global runtime rather than the
  removed scheduler types directly.

[Version 0.3.0 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
