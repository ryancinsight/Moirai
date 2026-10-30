# ADR 0090: Shared shutdown ownership

Status: Proposed

## Context

The lifecycle candidate preserved by [PR 650](https://github.com/ryancinsight/Moirai/pull/650)
does not yet satisfy the shutdown boundary. Its timed control method creates a
helper per call, while executor and native scheduler destructors join workers.
The executor admission flag does not guard every submission path. Async work
can also complete on an external waking thread, outside scheduler joining.

An empty task table is insufficient evidence of drainage when registration
retirement precedes result publication or callbacks. A callback may request
shutdown while it still owns the settlement obligation being awaited.

## Recommended decision

Create one shutdown worker inside fallible executor construction. It retains
the scheduler, registry, task ownership, and metrics until settlement finishes.
Keep one request predicate, a worker wake handle, and shared completion state;
use no request queue or detached helper for individual shutdown calls.

A request closes executor admission before delivering cancellation callbacks.
Task registration, scoped execution, and indexed entry observe the same gate.
Operations admitted before closure either join the tracked drain or receive
the scheduler's typed rejection. Closing registration and taking its owner
snapshot share their synchronization boundary.

First request cancellation for every captured task, then deliver their wakes
through the existing panic-containment boundary. Retain each settlement owner
through future destruction, result publication, waiter callbacks, and contained
panic-payload cleanup. Retirement occurs once after that settlement, including
handled failure. Shared completion requires both scheduler drainage and release
of outstanding settlement owners.

External explicit waits observe that one completion. Each timed caller has
its own entry-relative deadline and cannot replace or cancel the operation.
Executor destruction only publishes a request and unparks the worker; retained
owners prevent transitive scheduler joining or user callback delivery there.

Split native scheduler request publication from joining. Final scheduler and
blocking-lane destruction request closure without joining or acquiring a queue
mutex. Explicit drainage retains the existing join election. Accepted jobs
retain their queues until drainage; partial construction explicitly joins every
started worker before returning its error. Lazy blocking-lane initialization
must observe a concurrent closure request.

Avoid self-waiting. Scheduler workers, the shutdown worker, and this executor's
inline execution or settlement context request closure and return. Other
callers wait normally. Expose accurate worker membership through the existing
`SchedulerControl` role, and restore executor-specific thread context on unwind.
An unrelated executor's worker does not acquire this exemption.

Preserve the `ExecutorControl` return signatures. Report shutdown-worker
construction failure through a typed executor error retaining its original
`std::io::Error` source. The new exhaustive-error variant and required scheduler
membership query join the 0.12 public migration boundary in ADR 0087.

## Alternatives

Only splitting native request and join leaves cancellation callbacks on the
requesting thread. Only adding a helper retains blocking final-handle
destruction and overlooks external inline settlement. A helper per call permits
unbounded overlapping operations. Per-callback threads obscure resource bounds
and cannot establish bounded completion of arbitrary user code.

## Consequences and limits

There is one additional worker per successfully constructed executor. Failure
occurs at construction, before exposing the executor. Its ownership must not
form a cycle that prevents a shutdown request from reaching the worker.

Non-yielding polls, closures, destructors, and callbacks remain nonpreemptible.
They may retain the shutdown owner until they return. Caller deadlines and
library-owned destruction remain independent of that work; this is not a
hard-real-time scheduling guarantee or bounded completion of arbitrary code.

## Verification

Gate an external inline completion callback: its published result is readable
while its settlement owner remains retained, and retirement occurs once after
release. Exercise callback and destructor panic, first-admission rejection,
saved stale wakers, and overlapping cancellation/shutdown requests.

Gate non-yielding compute and blocking work while checking independent caller
timeouts and request-only destruction through event synchronization. Cover
shutdown from native workers, external inline settlement, callbacks, and an
unrelated executor. Reuse construction rollback tests and production-cell Loom
models; applicable Miri cases must select the new test paths explicitly.

These are required executable oracles, not claims that they currently pass.

## Overturning evidence

A retained-owner cycle, self-wait, admission after closure, premature shared
completion, lost wake, missing error source, or destructor join rejects this
design or its implementation. An implementation that satisfies the same
ownership and reentrancy laws without the additional worker can replace it.
