### Added

- `ParallelIterator::seq_iter` lets compatible owned/borrowed vector sources and
  `map`, `filter`, `copied`, and `cloned` adapters expose their logical stream
  directly. Standard `sum` and `product` retain one whole-stream trait
  invocation while avoiding an intermediate vector on those pipelines. A
  warmed borrowed copied/map/filter sum drops from one allocation to zero; the
  retained 1,024/32,768/131,072-item Criterion medians improve by
  51.9%/67.5%/91.9% on the measured x86-64 Windows host.
- `ParallelIterator::{sum_reassociated, product_reassociated}` provide explicit
  deterministic shard-folding terminals for associative output types. Standard
  `sum` and `product` again invoke `Sum<Item>` and `Product<Item>` once over the
  complete logical stream, preserving lawful custom accumulator semantics and
  avoiding output-self trait bounds.
- `moirai_parallel::for_each_chunk_buffers_mut_enumerated_with` applies one
  indexed operation to matching chunks from any const-generic number of
  homogeneous mutable buffers. Equal lengths validate before mutation, ragged
  tails remain visible, and fixed arrays add no provider allocation after the
  process-wide executor is initialized.
- `moirai-http` now follows at most ten redirects by default with RFC 9110
  method/body behavior and RFC 3986 relative-reference resolution. One timeout
  bounds the complete logical request, including redirects and a stale
  idempotent-connection retry. Cross-origin redirects remove credentials,
  destination framing is regenerated, and pooled connections expire lazily
  after five idle minutes without a background task. `set_max_redirects` and
  `set_idle_timeout` configure those bounds.
- Native positioned reads for `moirai_async::fs::File` through the existing
  `AsyncReadAt` and `AsyncLength` contracts. Unix uses cursor-independent
  `read_at`; Windows serializes only the cursor save/read/restore sequence, so
  ordinary file operations retain their existing lock-free path.
- Deterministic TLS integration fixtures for trusted, wrong-hostname,
  untrusted, and expired certificate cases. The handshake tests now match
  Rustls certificate-error variants and require rejected server handshakes to
  terminate within their existing deadline. Replacing runtime `rcgen`
  certificate generation removes its native test-only dependency graph.
- `moirai_utils::DESTRUCTIVE_INTERFERENCE_SIZE` and `moirai_utils::CachePad`.
  `moirai-utils` now owns the per-target cache tables that six crates
  previously each hardcoded at 64, and distinguishes the transfer granularity
  (`CACHE_LINE_SIZE`, unchanged at 64 on x86-64/aarch64 — used for prefetch
  strides and chunk widths) from the false-sharing separation
  (`DESTRUCTIVE_INTERFERENCE_SIZE`, 128 on those targets, following
  `crossbeam-utils`' `CachePadded` table). `const _` assertions keep the
  constants and `CacheAligned`'s `repr(align)` from drifting apart.
- `moirai-core/tests/loom_mpmc_waiter.rs`: a loom model of the bounded MPMC
  channel's waiter-count protocol, covering the shipped shape plus three
  counter-examples (inverted registration, unfenced notifier, unfenced waiter).
- `moirai-scheduler/tests/loom_chase_lev_resize_gate.rs`: a bounded Loom model
  of the encoded resize-owner/active-stealer gate, covering entry retry and
  exclusion while single and batched steal access regions remain active.
- `moirai-scheduler` now encodes resize ownership and active thief access in one
  atomic gate. This closes the separate flag/counter admission window and also
  excludes new thief access while shared retired storage is reclaimed.

- `moirai-benchmarks/thread_schedule_comparison` now includes a bounded
  saturated-admission comparison: one-worker Moirai capacity-plus-one rejection
  is measured beside an equal-capacity Crossbeam `try_send` rejection. Both
  paths assert their full-queue state before timing and use bounded Criterion
  windows; the rows compare rejection latency only, not scheduler semantics.

[Version 0.6.1 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
