### Performance

- **Worker count is derived once, not per call.**
  `for_each_chunk_mut_with_state` and `fold_reduce_with` called
  `themis::CpuTopology::detect()` on every invocation to read one number,
  materializing the whole NUMA and cache-level description: 9,935 ns, 77
  allocations and 16,480 bytes per call on a 24-processor host. The count is a
  process constant, so it now comes from a cached
  `moirai_core::executor::logical_parallelism`, which replaces four copies of
  the same expression (including `moirai-core`'s `num_cpus`, called twice by
  `ExecutorConfig::default`). Semantics are unchanged. Measured at
  `for_each_chunk_mut_with_state`: 23.84 us/call with 77 allocations before,
  10.50 us/call with none after.

