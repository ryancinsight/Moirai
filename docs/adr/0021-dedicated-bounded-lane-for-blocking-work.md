# ADR 0021: Dedicated bounded lane for blocking work

Status: Accepted

- Date: 2026-07-15
- Change class: [arch]
- Refs: ISSUE-213

**Context.** `BlockingTask` currently supplies only a zero-sized routing marker.
Its jobs enter the same per-worker queues and worker set as synchronous and
async-ready jobs. A starvation construction with one blocking job per compute
worker prevents every compute worker from reaching a queued synchronous job.
Affinity offsets change placement but do not provide admission isolation,
backpressure, cancellation ownership, or a shutdown boundary.

**Decision.** `ThreadScheduler` owns one lazily initialized, Moirai-native
blocking lane with a bounded, per-blocking-worker synchronous queue. Ordinary
compute-only executors therefore allocate no blocking worker stacks or queue
storage. `BlockingTask` dispatch is
selected through the sealed `WorkClass` associated capability, so the public
work-class API remains zero-sized and statically routed. Blocking workers
execute the same `ScheduledJob` lifecycle boundary as compute workers, but
maintain separate pending and active counters. The scheduler's quiescence and
metrics surfaces aggregate both lanes, while compute-worker parking and
shutdown observe only compute-lane pending work. This prevents idle compute
workers from spinning behind blocking backlog.

Lane admission is non-blocking and returns typed resource exhaustion when the
selected bounded queue is full. A locality hint selects a blocking lane;
otherwise a thread-local ticket distributes submissions without a shared
round-robin atomic. Shutdown closes all lane queues, drains admitted jobs,
and joins the blocking workers. Queued task cancellation remains owned by the
existing lifecycle token: the lane drops a cancelled job only after its
worker dequeues it, so result publication and cancellation metrics retain one
canonical path.

**Rejected alternatives.** Keeping one worker set cannot prove starvation
freedom. An unbounded blocking queue violates the memory bound. A second
executor implementation would duplicate scheduling, lifecycle, and panic
containment logic. A global mutex around one blocking queue would serialize
all producers and add avoidable contention; per-worker bounded channels keep
the synchronization boundary local to admission and one receiver.

**Verification.** The implementation must provide value-semantic tests for
compute progress while every blocking worker is occupied, queue-full
backpressure, queued cancellation, graceful drain, and shutdown rejection.
Focused nextest and warning-denied Clippy are the primary evidence tier;
Criterion compares blocking admission and execution handoff after correctness
passes. No Tokio or Smol production dependency is introduced.
