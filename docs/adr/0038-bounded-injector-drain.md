# ADR 0038: Bounded injector drain

Status: Rejected

- Date: 2026-09-02
- Change class: [patch]
- Refs: ADR-034, ADR-035, ADR-036, `MOI-QUEUE-BOUNDED-DRAIN-2026-09-02`

## Context

`WorkerQueueOwner::next_job` reaches the injector only when every local plane is
empty, and it then drains the injector with an unbounded loop, pushing each
entry onto a local plane until the injector is empty.

The Chase-Lev planes only ever grow. ADR-035 sizes their initial storage and
ADR-036 sizes each slot, but neither bounds what a single drain pass moves, so
the largest burst a worker ever drains — not the configured capacity — sets the
retained slot count for the life of the process. Measured on a one-worker
scheduler starting at the 16-slot minimum, with the worker held inside a
blocking job while the burst accumulated:

| Burst jobs | Retained local slots after the drain |
| --- | --- |
| 200 | 256 |
| 2,000 | 2,048 |

Growth also retires each superseded buffer into `ChaseLevInner::retired_arrays`,
and the executor's planes take the default `DeferredReclaim`, which retains them
until the final owner or stealer endpoint drops. Every intermediate buffer of a
doubling run is therefore held alongside the live one, so the 16-to-2,048 growth
above retains 2,032 dead slots on top of 2,048 live.

Retained storage tracks burst size linearly. At Apollo's 24 workers and the
128-byte `ScheduledJob` of ADR-036, a 256-slot plane retains 32,768 bytes per
worker where the 128-slot default provisions 16,384.

## Decision

Rejected. The proposal was to bound one drain pass to the batch the steal paths
already move (`MAX_BATCH_STEAL`, 16). Because a pass runs only when every plane
is empty, that bound is also an occupancy bound, and it held: the same bursts
settled at 32 slots for both 200 and 2,000 jobs, one growth step above the
start.

It is rejected because draining to exhaustion is load-bearing for priority
ordering, which this ADR's own failure modes required to be preserved. The
injector is a single cross-priority FIFO of `(Priority, ScheduledJob)`; only the
local planes are priority-ordered. Emptying the injector into the planes is
therefore what lets a high-priority job preempt work already queued ahead of it.
Under a bounded pass a `Priority::High` job enqueued behind 257 `Normal` jobs is
reached only after those jobs run — priority inversion proportional to the
queued burst. `local_queue_growth_and_cross_worker_steal_execute_each_job_once`
fails against the bounded drain for exactly this reason: the owner consumes the
first batch itself instead of preempting into the marker and yielding the burst
to a thief.

The retention is real and stays open. It is addressed by releasing an oversized
plane once it drains — the storage is dead at that moment — not by bounding
what a pass may move. That work is `MOI-QUEUE-PLANE-SHRINK`.

## Alternatives rejected

1. Lower `DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY`. Rejected: with an unbounded
   drain the retained size is set by the burst, so a lower default is erased by
   the first burst. It becomes a question only once growth is bounded.
2. Make the injector priority-ordered so a bounded pass could still preempt.
   Not evaluated here; it changes the admission structure of ADR-034 and is a
   larger decision than the retention it would serve.
3. Drain to exhaustion but cap plane growth. Rejected because the cap would have
   to reject or reroute jobs already dequeued from the injector, changing the
   admission contract of ADR-034.

## Evidence

`retained_local_plane_storage_tracks_burst_size` measures 256 slots for 200 jobs
and 2,048 for 2,000 against the shipped drain, and ships as a characterization
of that cost. Against the bounded drain both settled at 32, so the saving was
established before the approach was rejected on priority grounds; warning-denied
workspace Clippy and formatting passed, and workspace Nextest surfaced the
priority regression as a single failing test.
