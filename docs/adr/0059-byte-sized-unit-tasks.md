# ADR 0059: Byte-sized unit tasks

- Status: Accepted
- Date: 2026-09-15
- Item: [MOI-BYTE-SIZED-TASKS-2026-09-15](../backlog.md#MOI-BYTE-SIZED-TASKS-2026-09-15)

## Context

A data-parallel pass over lanes, rows or matrices has two scheduling
decisions: whether to run in parallel at all, and how much work one task
carries. Both depend on bytes moved per task, not on element counts.
Moirai's chunk operators take a fixed `chunk_size`, and `ExecutionPolicy`
sees an element count and a chunk count, never element width or bytes.
Each consumer therefore derives both decisions itself:

- apollo `plan/fft/lanes.rs` counts complex elements against
  `PARALLEL_THRESHOLD = 32_768`, converts it to `PARALLEL_BYTES` for a pass
  whose input and output lanes differ in length and type, sizes tasks to
  `TASK_BYTES = 64 KiB`, and branches between `Parallel` and `Sequential` by
  hand in `paired`. One lane per task made a 64³ pass 3.3x slower than serial;
  64 KiB tasks made it faster than serial (`dimension_3d::pass_attribution`,
  2026-09-09). Counting only the output side left the real 32³ half-spectrum
  pair serial and 28% slower than with both sides counted (apollo #442).
- leto-ops `layout/complex/batch.rs` keeps a private 1 MiB parallel floor and
  64 KiB destination-row tasks, measured on the same 32³ and 64³ volumes.
- kwavers' PSTD kernels call `enumerate_mut_with::<Adaptive>` per element and
  recover `(i, j, k)` by division to read a factor constant along a lane,
  because no operator hands a task whole lanes with their index.

The chunk-geometry change (MOI-CHUNK-POLICY-GEOMETRY) gave policies the chunk
count; it did not give them bytes, and it did not size tasks.

## Decision

Recommended option, adopted:

1. `ExecutionPolicy` gains a defaulted method that sees bytes:

   ```rust
   fn parallelize_work(len: usize, chunks: usize, bytes: usize) -> bool {
       let _ = bytes;
       Self::parallelize_chunks(len, chunks)
   }
   ```

   Every existing policy keeps its behaviour through the default.

2. A policy `WorkBytes<const N: usize>` returns `bytes >= N` from
   `parallelize_work` and keeps element-count behaviour elsewhere.

3. One operator, `for_each_unit_task_mut_with::<P, T, S, Init, F>(data,
   unit_len, unit_bytes, init, f)`:
   - `data.len()` is a whole number of units of `unit_len` elements;
   - `unit_bytes` is what one unit moves in total, including any input the
     closure reads beside it, so a pass over a wide input counts that input;
   - a task carries `max(1, UNIT_TASK_BYTES / unit_bytes)` whole units, with
     `UNIT_TASK_BYTES = 64 KiB` from the measurements above;
   - the decision is `P::parallelize_work(data.len(), tasks, units *
     unit_bytes)`;
   - `f(&mut state, first_unit, units)` receives the index of its first unit
     and a whole-unit slice; `init` runs once per scheduled task.

   Shared inputs stay outside the operator: the closure indexes them by unit,
   which covers apollo's output-beside-input pass and kwavers' one output
   beside several inputs without a family of buffer arities.

## Alternatives

- **A byte-aware policy with no new operator.** Rejected: task width would
  still be derived per consumer, which is the copy this removes.
- **Pair, triple and quad unit operators.** Rejected while every consumer
  wrote one buffer: shared inputs are indexed by unit, so one operator served
  them all. A pair form was admitted once a consumer wrote two fields per
  element in one pass (revision 2026-09-15, below), and a triple form once a
  consumer wrote three (revision 2026-09-15, triple, below). A quad form
  remains unadopted until a consumer writes four.
- **Leto-ops as the home.** Rejected: the decision needs no layout knowledge,
  and apollo would still reach it through leto only for scheduling.

## Consequences

- apollo `lanes::paired` and `lanes::each` and the kwavers PSTD kernels, whose
  units are whole lanes, migrate to the operator, each deleting its constants
  and hand branch in the same change.
- leto-ops' batched transpose does not fit as it stands: its tasks keep at
  least one cache line of source columns together (four rows per task
  measured 31–33 µs against 42–44 µs for one at 64³), and a batch's
  destination rows need not divide into whole groups of those columns. The
  operator has no per-task unit floor and no ragged final unit; whether to add
  a floor is decided, and measured, when leto-ops migrates.
- `UNIT_TASK_BYTES` becomes one measured constant owned by moirai, replacing
  apollo's `TASK_BYTES` and leto-ops' `PARALLEL_TRANSPOSE_TASK_BYTES`; a consumer whose
  unit exceeds it runs one unit per task, as today.

## Verification

- Native tests: whole-unit task boundaries, a ragged final task, the index of
  each task's first unit, both decisions under `WorkBytes`, and the default
  method's parity with `parallelize_chunks` for existing policies.
- apollo's `lane_threshold_crossover` probe reproduces its 32³ and 64³ pair
  timings with `paired` routed through the operator.

## Revision 2026-09-15 — Accepted

Implemented in moirai #346 (52 native tests, 4 doctests). kwavers #774 runs
the PSTD split-field kernels on the operator with bitwise differentials
against the per-element formulas. apollo #460 routes `lanes::each` and
`lanes::paired` through it: the policy decision and task width are the
ones they replace by construction, and `lane_threshold`'s per-run minima
agree within 1% (medians were invalid under host load).

## Revision 2026-09-15 — Paired unit tasks

kwavers' linear equation of state writes `div_u` and `p` per element in one
fused traversal, which the single-buffer operator cannot hand out, and it was
the last PSTD step kernel on a hand-sized chunk. moirai #348 adds
`for_each_unit_task_pair_mut_with(a, b, unit_len, unit_bytes, init, f)`: two
buffers of one length, each task receiving the same run of whole units from
both, with the task width and `parallelize_work` decision of the single-buffer
operator and `unit_bytes` counting one unit of each buffer plus the inputs read
beside them. Native tests cover aligned runs with a ragged tail under both
policies, one state per task, and mismatched lengths rejected.

## Revision 2026-09-15 — Triple unit tasks

kwavers adds its PSTD density source to the three split densities `rhox`,
`rhoy` and `rhoz` per element in one fused traversal, and that kernel was the
last PSTD step kernel on a hand-sized chunk (`DENSE_SOURCE_CHUNK`, scheduled
through `for_each_chunk_triple_mut_enumerated_with`).
`for_each_unit_task_triple_mut_with(a, b, c, unit_len, unit_bytes, init, f)`
extends the pair form to three buffers of one length. With three operators,
the whole-unit check, task width, task count and `parallelize_work` decision
move into one private planner that all three call. Every operator thereby
makes the same decision for the same `(len, unit_len, unit_bytes)`, and the
existing single and pair tests pin that the planner changed neither. Native
tests cover aligned triple runs with a ragged tail under both policies, one
state per task, and a mismatched second or third buffer rejected.
