### Changed

- The SPSC halves cache the opposite counter, so a send reads the consumer's
  cache line only when the queue looks full and a receive reads the producer's
  only when it looks empty, instead of on every operation. A stale cache always
  errs toward "full"/"empty" and is refreshed before reporting either, so the
  exact capacity check is unchanged. The `Cell` also supplies the `!Sync`
  property that a `PhantomData` marker used to carry, making it load-bearing in
  code rather than only in a comment. See ADR-025.

- **[breaking]** `SpscChannel` is no longer exported from `moirai-core`. The
  single-producer/single-consumer discipline is enforced by `SpscSender` and
  `SpscReceiver`, which are neither `Clone` nor `Sync`, but the bare channel
  escaped alongside them: it implements `Channel<T>`, whose `send`/`recv` take
  `&self`, and carried an `unsafe impl Sync`, so safe code could share one
  channel and drive two producers into the same buffer slot — a data race, with
  one value overwritten undropped, and a double free on the `recv` side.
  Construct channels with `moirai_core::channel::spsc(capacity)`, which is
  unchanged and returns the same pair; `SpscChannel::channel(n)` becomes
  `channel::spsc(n)`. See ADR-024.

- Run the `moirai-iter` parallel sorts' recursive fork-join on the unified
  scheduler's scope instead of the crate's own thread pool. That pool is a FIFO
  queue with no work stealing, so a worker blocked on another of its jobs could
  not run it — the starvation that the fork budget guarded against, at the cost
  of capping the work tree at the pool's width. A scheduler worker waiting
  inside a scope runs queued work instead of parking, so the budget is gone and
  the recursion is deadlock-free by construction. A fork the scheduler refuses,
  while shutting down or under admission backpressure, now runs on the calling
  lane rather than being dropped. Splitting stops once a sub-slice is smaller
  than `len / (workers * 8)` so fork count follows machine width rather than
  input size. `ThreadPool` remains for flat fan-out only.

