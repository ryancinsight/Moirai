# ADR 0035: Resizable local-queue initial capacity

Status: Accepted

- Date: 2026-08-28
- Change class: [major] [arch]
- Refs: `MOI-LOCAL-QUEUE-CAPACITY-036`,
  `MOI-LOCAL-QUEUE-FOOTPRINT-2026-08-28`, ADR-034
- Revision: 2026-08-28 — reduce the default initial capacity from 256 to 128
  after exact retained-footprint attribution and controlled queue-kernel
  measurements.
- Revision: 2026-09-02 — apply the configured capacity to the default-priority
  plane only; the other three planes start at the minimum and grow. The
  priority factor in this decision's own retention formula was never paid for:
  a submission carries one priority, so a consumer that never sets one uses one
  plane. Measured on Apollo's workload shape, warm-pool local storage falls
  from 1,572,864 to 540,672 bytes (65.6%), and the cost to a plane that *is*
  used is one owner-side growth, after which it performs identically.

## Context

`ExecutorConfig::max_local_queue_size` and both public builders expose a hard
maximum, but executor construction discards the value. Each worker instead
constructs four priority-local Chase-Lev deques from `ThreadScheduler`'s first
const parameter. Those deques grow when full, so the constructor value is an
initial allocation policy rather than an admission bound.

The same const parameter also bounds every queue in the dedicated blocking
lane. One value therefore controls two unrelated policies: resizable compute
storage and bounded blocking admission. Reinterpreting the const as the public
local-queue setting would keep that coupling and make runtime configuration
impossible. Treating the resizable deque as bounded would misstate the
algorithm and could silently lose work.

The Chase-Lev constructor rounds to the next power of two. Its unchecked
normalization overflows for sufficiently large inputs, which either panics or
produces a capacity smaller than the request depending on build semantics. A
public configuration path must reject that value before worker threads start.

## Decision

Replace the false maximum contract directly:

- `DEFAULT_LOCAL_QUEUE_CAPACITY` becomes
  `DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY`.
- `ExecutorConfig::max_local_queue_size` becomes
  `local_queue_initial_capacity`.
- Both builder methods become `local_queue_initial_capacity`.
- The former symbols are deleted; no alias, deprecated forwarder, or parallel
  configuration field remains.

Introduce `DequeCapacity<T>`, a validating newtype in `moirai-scheduler`.
`TryFrom<usize>` rounds `0..=16` to the minimum 16 slots and otherwise rounds
up to the next power of two. An input whose next power of two is not
representable, or whose concrete element-slot or generation-state layout is
invalid, returns `DequeCapacityError`. `ChaseLevDeque::new` accepts only this
type-bound proof. Every initial deque allocation therefore receives a valid
power of two and both valid layouts without repeated checks or input-dependent
panics.

`ThreadScheduler`'s first const parameter is renamed internally to
`BLOCKING_QUEUE_CAPACITY` and continues to bound the independent blocking
lane. Local worker queue types lose that const parameter. Scheduler
construction validates the runtime local initial capacity once and passes the
result to the default-priority deque of every worker; the remaining planes take
`DequeCapacity::minimum()` and grow on the owner's push (2026-09-02 revision). The public
`new_with_config` constructor is replaced by
`new_with_local_queue_initial_capacity`; `new` retains the default policy.

The external-admission policy from ADR-034 is unchanged. Its aggregate bound
continues to partition into fixed per-worker injectors. The local initial
capacity is not backpressure: retained initial slots scale as

`workers * priority levels * normalized initial capacity`.

Growth remains the Chase-Lev algorithm's existing owner-only resize. Work is
never rejected because a local deque reaches its initial capacity.

The default initial capacity is 128 slots. At 24 workers, four 128-byte
`ScheduledJob` planes per worker retained 1,572,864 direct bytes, half the
3,145,728 bytes retained by the former 256-slot policy. Since the 2026-09-02
revision only the default plane carries that capacity, so the same 24 workers
retain 540,672 bytes: `24 x 16,384` for the default planes and `72 x 2,048` for
the rest. In the final
20-sample same-binary run, the warmed 15-item production-deque interval at
128 slots was 277.34–280.56 ns and overlapped the 256-slot interval of
280.42–282.49 ns. A cold 257-item burst was 5.412–5.550 us at 128 slots
versus 4.955–5.033 us at 256 slots: the selected policy accepts one
additional owner-only resize on that cold burst in exchange for the bounded
50% steady-state storage reduction. Capacities 16, 32, and 64 were rejected
because at least one controlled warm run produced a non-overlapping regression
against 256; 128 was the smallest candidate without that result.

Apollo's exact release retained-footprint probe, built against this local
Moirai source, observes the same 1,572,864 direct bytes during pool warmup.
Its warm in-place FFT remains at zero global and direct allocations for
1,024 through 262,144 elements. This confirms that the provider reduction
reaches the downstream consumer without moving allocation into its warm
transform path.

## Per-plane capacity (2026-09-02 revision)

Retention here is `workers x priority levels x normalized initial capacity`,
and the middle factor was never examined. A submission carries exactly one
priority, so the planes are not a partition of one workload's pushes: a
consumer that never sets a priority uses the default plane and pays for four.

Counting first pushes per plane, Apollo's chunked transforms touch only the
default plane, and across this repository's own suite the default plane is
touched by 85 test processes against 4, 4 and 7 for the other three. The
payload is eager and exactly `capacity x size_of::<ScheduledJob>()`, verified
by holding planes alive while varying the capacity: 16 slots allocate 2,048
bytes, 64 allocate 8,192, 128 allocate 16,384. Three unused planes therefore
retain 49,152 bytes per worker.

The other three planes now start at `DequeCapacity::minimum()`. Measured on
Apollo's workload shape through Mnemosyne's own accounting, with the uniform
policy as the paired baseline in the same probe:

| policy | live plane allocations | bytes |
|---|---|---|
| uniform 128 | 96 x 16,384 | 1,572,864 |
| default plane only | 24 x 16,384 + 72 x 2,048 | 540,672 |

The cost falls only on a plane a workload actually uses, and it is one-time.
Fresh-deque bursts starting at 16 slots run 1.03x-1.36x the 128-slot start for
16 to 1,024 pushes; repeating the same burst after the plane has grown gives
ratios of 1.00, 1.00, 1.00, 0.90 and 1.01 — the grown plane is the 128-slot
plane. This is the trade this decision already accepted going from 256 to 128
("one additional owner-only resize on that cold burst"), now confined to planes
a workload does not use.

Capacities 16, 32 and 64 were rejected above as the *global* capacity on a warm
regression measured on the busy plane. That rejection is unchanged: the busy
plane keeps 128.

## Failure modes

- An unrepresentable normalization or concrete element layout returns
  `ExecutorError::InvalidLocalQueueInitialCapacity` before any worker starts.
- Allocation failure remains the process allocator's failure policy; this
  decision does not claim recoverable allocation.
- A deque that later grows beyond a representable allocation layout follows
  the same unrecoverable process policy because owner `push` has no rejection
  channel after the deque has accepted work.
- Local growth preserves the deque's generation and reclamation protocol.
  Exactly-once behavior is verified with real owner/thief execution because
  the fixed-capacity Loom model does not model resize.
- A non-default plane must still accept work past its minimum capacity. The
  queue algorithm, its stealers and their publication are unchanged by the
  2026-09-02 revision — only the initial slot count differs per plane — so that
  revision requires no new concurrency model.

## Alternatives rejected

1. Keep the public maximum name and impose a hard bound. Rejected because the
   local deque has no rejection channel and bounded local admission would
   require a different scheduling algorithm.
2. Continue deriving local capacity from the blocking-lane const. Rejected
   because it preserves unrelated policy coupling and leaves `ExecutorConfig`
   ineffective.
3. Validate independently in each worker. Rejected because repeated checks can
   drift and cannot make invalid deque construction unrepresentable.
4. Clamp overflowing requests. Rejected because the resulting allocation would
   violate the documented at-least-requested contract.
5. Select the 16-slot minimum. Rejected because its five-step cold growth and
   non-overlapping warm regression in one controlled run do not justify the
   additional storage reduction over the measured 128-slot policy. Superseded
   in part by the 2026-09-02 revision: the rejection stands for the busy plane
   and is exactly why the default plane keeps 128, while a plane a workload
   never pushes to has no warm path to regress.
6. Allocate each plane lazily on its first push (2026-09-02). Deferred rather
   than rejected: stealers are published eagerly from each deque, so late
   creation needs a synchronization argument and the Loom model this decision
   requires for queue algorithm changes. Per-plane initial capacity reaches 87%
   of the same saving with no algorithm change, so it is the increment taken
   first. Tracked as ISSUE-226.

## Migration

Replace `max_local_queue_size(value)` with
`local_queue_initial_capacity(value)`, and replace direct
`ExecutorConfig::max_local_queue_size` field access with
`local_queue_initial_capacity`. The value controls initial retained storage,
not a maximum. Direct `ThreadScheduler` users replace `new_with_config` with
`new_with_local_queue_initial_capacity`, passing the desired runtime initial
capacity explicitly. Direct Chase-Lev users construct a
`DequeCapacity::<T>` with `TryFrom<usize>` and pass it to
`ChaseLevDeque::new`.

## Verification plan

- Test normalization at zero, below/at/above the minimum, a non-power-of-two,
  the first overflowing value, and a normalized capacity whose concrete slot
  allocation layout is invalid.
- Assert every worker receives the normalized configured capacity.
- Force one owner to grow past its initial capacity, then allow another worker
  to steal; assert every indexed job executes exactly once.
- Preserve global-admission, blocking-lane, scheduling, shutdown, and existing
  low-level Chase-Lev property/stress suites.
- Run configured Nextest, warning-denied Clippy/rustdoc, doctests, cross-target
  checks, SemVer classification, and an independent architecture review.
