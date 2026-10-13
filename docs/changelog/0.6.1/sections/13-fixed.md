### Fixed

- Preserve Miri-valid provenance in `SplitDeque`'s panic-repair memmove by
  deriving one mutable base pointer before the overlapping copy. The deque
  unit suite now passes the focused Miri run without raw-copy violations.

- Return every chunk from `ParallelContext::execute_iter`. It collected results
  from a channel until the senders dropped, so a panicking chunk ended the
  collect early and the call returned a short `Vec` with `Ok` — 32 of 40 items
  in the regression case. Results now land in per-index slots and a missing
  chunk surfaces as a panic.

- Run a scoped job on the calling lane when the scheduler's admission queue
  rejects it. `SchedulerScope::flush` previously dropped such a job and
  returned the error, so a scope broke its "every spawned job runs before the
  scope returns" promise silently — the dropped job's completion token
  decrements the scope counter exactly as a finished one does, so the caller
  resumed as though borrowed work had happened, and `moirai_parallel::scope`
  turned the error into a panic. Admission now leaves a refused job in the
  caller's slot, and the caller-run event is counted through the existing
  `admission_caller_runs` surface. Shutdown is not backpressure and still
  propagates.
- Run a `moirai_parallel::join_with` branch on the caller when the scheduler
  refuses it. The scheduled branch was moved into the job, so a job refused
  while shutting down — or rejected by a full per-worker admission queue — was
  dropped with the branch inside it, and the join turned the resulting error
  into a panic. Both lanes now claim the branch from a shared slot, so it runs
  exactly once on whichever lane reaches it, and backpressure makes a join
  sequential rather than fatal.

[Version 0.6.1 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
