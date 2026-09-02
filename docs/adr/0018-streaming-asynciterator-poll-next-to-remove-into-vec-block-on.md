# ADR 0018: Streaming AsyncIterator (poll_next) to remove into_vec block_on

Status: Proposed

- Change class: [arch]

- `AsyncIterator`'s `into_vec()` materialize-then-process shape forces
  `AsyncMap`/`AsyncFilter` (and their parallel variants) to `block_on` the
  per-item async closure inside the synchronous `into_vec()`. The fix is a
  streaming trait (`fn poll_next(self: Pin<&mut Self>, cx) -> Poll<Option<Item>>`
  or `async fn next(&mut self)`), so adapters await natively and the terminals
  (already cooperative) drive a real stream. Breaking change: every
  `AsyncIterator` impl and caller updates in the same coordinated unit;
  the already-landed cooperative terminals are forward-compatible with it.
