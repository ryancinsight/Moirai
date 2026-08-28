# ADR-035: Resizable local-queue initial capacity

Status: Accepted

- Date: 2026-08-28
- Change class: [major] [arch]
- Refs: `MOI-LOCAL-QUEUE-CAPACITY-036`,
  `MOI-LOCAL-QUEUE-FOOTPRINT-2026-08-28`, ADR-034
- Revision: 2026-08-28 — reduce the default initial capacity from 256 to 128
  after exact retained-footprint attribution and controlled queue-kernel
  measurements.

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
result to all four priority deques of every worker. The public
`new_with_config` constructor is replaced by
`new_with_local_queue_initial_capacity`; `new` retains the default policy.

The external-admission policy from ADR-034 is unchanged. Its aggregate bound
continues to partition into fixed per-worker injectors. The local initial
capacity is not backpressure: retained initial slots scale as

`workers * priority levels * normalized initial capacity`.

Growth remains the Chase-Lev algorithm's existing owner-only resize. Work is
never rejected because a local deque reaches its initial capacity.

The default initial capacity is 128 slots. At 24 workers, four 128-byte
`ScheduledJob` planes per worker retain 1,572,864 direct bytes, half the
3,145,728 bytes retained by the former 256-slot policy. In the final
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
   additional storage reduction over the measured 128-slot policy.

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
