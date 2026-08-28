# ADR-034: Executor queue capacity reaches scheduler construction

Status: Accepted

- Date: 2026-08-27
- Change class: [arch] [patch]
- Refs: `MOI-QUEUE-CAPACITY-034`, Apollo retained-footprint investigation
- Revision: 2026-08-27 — independent review established that the queue's
  sequence generations alias at capacity one; require two slots per worker.

### Context

`ExecutorConfig::max_global_queue_size` and both public builders expose a
maximum task count for the global admission queue. `HybridExecutor::new`
discarded the value, while each worker unconditionally allocated a 1024-slot
injector. The configured maximum therefore did not control behavior or memory.
On the 24-worker machine used for Apollo FFT profiling, first parallel use
retained 24 allocations of 262,144 bytes: 6 MiB before any workload-dependent
scratch storage.

The scheduler uses one priority-carrying injector per worker rather than one
shared queue. `LockFreeQueue` requires a power-of-two capacity of at least two:
its sequence protocol needs distinct empty and full generations, which alias
in a one-slot ring. An arbitrary executor-wide maximum therefore cannot always
be divided exactly.

`ExecutorConfig::max_local_queue_size` is a separate contract discrepancy.
The Chase-Lev queues are resizable and interpret their constructor argument as
an initial capacity, not a maximum. This decision does not relabel that policy
inside the global-capacity correction.

### Decision

Keep the per-worker injectors because selected-worker placement preserves the
scheduler's priority, locality, and NUMA routing. At executor construction,
normalize the worker count to at least one and derive one injector capacity:

`C = 2^floor(log2(floor(M / W)))`

where `M` is `max_global_queue_size` and `W` is the normalized worker count.
Construction rejects `M < 2W`, because every worker requires at least two slots.
The resulting aggregate capacity satisfies `W * C <= M`; the unused remainder
is the cost of the queue's power-of-two indexing invariant.

Pass `C` once through `HybridExecutor` and `ThreadScheduler` into every
`WorkerQueues` allocation. Direct `ThreadScheduler` construction derives from
`DEFAULT_GLOBAL_QUEUE_CAPACITY`, preserving one construction implementation.

### Alternatives rejected

1. Retain a fixed per-worker constant. Rejected because worker count then
   multiplies memory independently of the public maximum.
2. Round each partition upward. Rejected because aggregate capacity can exceed
   the documented hard maximum.
3. Replace the worker injectors with one executor-wide queue. Rejected because
   it removes selected-worker placement and introduces a shared contention
   point to solve a construction-policy defect.
4. Change the default constant only. Rejected because another unexplained
   constant leaves custom configuration ineffective.

### Verification plan

- Unit-test exact and non-power-of-two partitions, minimum valid capacity, and
  rejection when the executor-wide maximum cannot supply two slots per worker.
- Assert scheduler construction uses the derived capacity on every worker.
- Preserve saturation, caller-runs, wake-retry, shutdown, and priority tests.
- Run configured Nextest, warning-denied Clippy, rustdoc, doctests, and the
  standalone locked workspace gate.
- Re-run Apollo's retained-allocation instrument against the provider commit;
  allocation bytes, not elapsed time, are the acceptance oracle.

### Evidence

Apollo's unchanged release-mode retained-footprint probe, built against this
provider tree, reports 24 surviving 65,536-byte queue blocks at N = 65,536.
The entry measurement was 24 blocks of 262,144 bytes. Injector retention is
therefore 1.5 MiB versus 6 MiB, exactly matching the configured 8192-task bound
partition (`8192 / 24` rounded down to 256 slots per worker). The complete
first-forward window retains 4,061,316 bytes, and the warm-forward window
remains allocation-free. The test body completed in 0.02 seconds; its release
build and link took 2 minutes 21 seconds and is compile-structure evidence, not
runtime-kernel evidence.
