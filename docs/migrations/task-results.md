# Task result storage migration

These changes ship in 0.12.0 under [ADR 0087](../adr/0087-task-lifecycle-release.md).

`moirai_utils::result_cell::StateWord` replaces its required
`get_mut(&mut self) -> &mut u8` method with
`exclusive_state(&mut self) -> u8`. Implementors return the current state from
the same atomic storage while holding exclusive access to the storage object.
The cell uses this value only to decide whether an untaken result needs dropping.
The load, store, compare-exchange, and initial-state contracts are unchanged.

For an ordinary atomic field, return `*self.atomic.get_mut()`. For a backend that
grants closure-scoped exclusive access, return the value from that closure.
Do not return a cached state or manufacture a reference to a snapshot.

The accompanying [handle change](https://github.com/ryancinsight/Moirai/pull/646)
adds awaiting to existing task handles; their output remains
`Option<Result<T, TaskError>>`, matching blocking `join`. `None` means the handle
is detached or its result was already consumed by polling. Dropping a result
handle alone does not cancel the task. Async code must await dependent work
instead of blocking its worker with `join`.

The `Moirai` facade now implements `TaskManager`, imported from
`moirai::prelude`. `cancel_task(handle.id())` requests cancellation explicitly:
queued work skips its body, async work stops at a poll boundary, and running
synchronous or blocking closures continue until they return. A task already
returning its result may win the cancellation race.

`wait_for_task(id, timeout)` observes task settlement without consuming the
result handle. A positive timeout owns a PAL timer; expiry returns
`ExecutorError::WaitTimedOut { id }` and leaves the underlying task running.
Unknown identifiers return `ExecutorError::UnknownTask { id }`. PAL timer
construction and worker failures return `ExecutorError::TimerFailed { id,
source }`, preserving the original `std::io::Error` as the error source. `None`
creates no timer. Completion already visible when polled wins a simultaneous
successful or failed timer observation.

PAL `Timer::new(duration)` and `timer::sleep(duration)` now return
`std::io::Result<Timer>`. Durations outside the platform monotonic-clock range
return `ErrorKind::InvalidInput` instead of being clamped. `ExecutorError` no
longer implements `Clone`, `PartialEq`, or `Eq` because a preserved
`std::io::Error` implements none of those contracts. Match its variants and
fields directly; shared owners may clone their existing `Arc<ExecutorError>`.

`TaskRegistry::register_waker(id, &waker)` now returns `Option<TaskWaiter>`.
Retain the token while waiting and call `token.update(&waker)` on subsequent
polls. `None` means the task is unknown or completion already closed its waiter
set; recheck the task status. `update` returns false after the set closes.
Dropping the non-cloneable token unregisters that wait, including after timeout.
Independent tokens remain independent even when they use the same waker.
`wait_for_task` owns this token internally and releases it and its timer before
any terminal result. See [ADR 0089](../adr/0089-owned-completion-waiters.md).

`shutdown()` cancels pending async work and joins the worker drain.
`shutdown_timeout(duration)` bounds only the caller's wait; the drain continues
and runtime destruction joins it. Neither call preempts a running synchronous
or blocking closure.

[ADR 0085](../adr/0085-result-state-storage.md) records the storage decision.
