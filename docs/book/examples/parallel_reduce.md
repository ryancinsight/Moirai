# Example: Parallel Reduce

**Crate**: `moirai`
**Source**: `moirai/examples/book_parallel_reduce.rs`

Divide a 1 M-element array into 8 chunks, sum each chunk in a separate
work-stealing task, and reduce the partial sums.  The scheduler balances the
load across all available CPU cores automatically.

## Source

```rust
{{#include ../../../moirai/examples/book_parallel_reduce.rs}}
```

## Output

```text
parallel sum of 0..1000000: 499999500000
expected              : 499999500000
processed 1000000 elements across 8 tasks (125000 per task)
parallel-reduce assertion passed
```

## What to notice

- Each `spawn_fn` call hands the scheduler an independent unit of work.  The
  work-stealing algorithm assigns tasks to idle workers; on a 4-core machine
  with 8 tasks, the first 4 tasks start immediately and the remaining 4 are
  stolen as workers become free.

- The `chunk.to_vec()` copy moves ownership into the closure so it is
  `'static + Send`.  In a real physics kernel the data would be partitioned by
  index range rather than copied, keeping the example simple.

- The result is reduced with a sequential `.sum()` on the handles iterator.
  For larger fan-outs, the partial sums could themselves be reduced in a second
  wave of tasks.

- The Gauss formula `N×(N-1)/2` gives the expected value: 1_000_000 × 999_999 / 2 = 499_999_500_000.
