# ADR 0090: Ordered map output ownership

Status: Accepted

Owner extraction [PR 710](https://github.com/ryancinsight/Moirai/pull/710).

## Context

The parallel collectors allocate their final result directly. A worker that
panics leaves published and in-flight values in uninitialized storage, whose
destructor cannot identify the initialized values. The iterator collectors
already track completion per chunk and guard each in-progress prefix.

## Decision

Move the initialized-prefix owner to `moirai-core::memory` for both consumers.
Each collection allocates final value storage and one completion endpoint per
chunk. A borrowed view gives each indexed job the range of output
slots for that index. The scheduler joins all jobs before the parent reads
completion. The view's unsafe write requires exclusive use of each index.

While a chunk is being filled, its guard drops the initialized prefix if
value production unwinds. A successful write transfers drop ownership to the
parent by publishing its ending index. The parent verifies complete
coverage before converting the same allocation into a vector.

## Alternatives

Per-chunk vectors allocate on each job and move results into a second buffer.
An option per output makes initialization safe but may add a discriminant and
change the layout of every result. Per-chunk endpoints preserve one final value
allocation and the existing result layout.

## Verification

The parallel collector acceptance tests require all four entry points to
drop an owned result produced before a mapper panic. Core tests cover
complete and incomplete chunks, unwinding, range arithmetic and allocation
reuse. The iterator allocation tests continue to check final storage.
