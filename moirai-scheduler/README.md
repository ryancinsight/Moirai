# moirai-scheduler

[![crates.io](https://img.shields.io/crates/v/moirai-scheduler.svg)](https://crates.io/crates/moirai-scheduler)
[![docs.rs](https://docs.rs/moirai-scheduler/badge.svg)](https://docs.rs/moirai-scheduler)

Work-stealing primitives for the [Moirai](https://github.com/ryancinsight/Moirai)
concurrency library. This crate provides the reusable building blocks consumed
by the canonical runtime scheduler in `moirai-executor`; it intentionally
provides no scheduler of its own.

- `ChaseLevDeque` — the canonical Chase-Lev deque: O(1) wait-free local
  push/pop for the owner, lock-free steal for thieves, dynamic resizing, and
  `bottom`/`top` isolated to separate cache lines. Storage-generation and
  resize-owner contention uses bounded processor hints followed by cooperative
  thread yields; it neither allocates nor sleeps.
- `SplitDeque` — a private owner stack backed by a shared deque, reducing steal
  contention when the spawn rate greatly exceeds the steal rate.
- `numa::CpuTopology` — hardware NUMA/cache topology discovery, and
  `numa::AdaptiveBackoff` for spin/yield/sleep backoff.

Correctness is covered by exactly-once concurrency stress tests and `loom`
models of the Chase-Lev steal/pop ordering and its combined resize-owner/
active-stealer admission protocol (`tests/loom_chase_lev*.rs`, compiled only
under `--cfg loom`).

```toml
[dependencies]
moirai-scheduler = "0.5"
```

```rust
use moirai_scheduler::{ChaseLevDeque, DequeCapacity};

let capacity = DequeCapacity::<u32>::try_from(64).expect("64 is representable");
let mut deque: ChaseLevDeque<u32> = ChaseLevDeque::new(capacity);
let stealer = deque.stealer();

deque.push(1);
deque.push(2);

assert_eq!(deque.pop(), Some(2)); // owner pops LIFO
let _ = stealer.steal();          // thieves steal FIFO
```

Full documentation: <https://docs.rs/moirai-scheduler>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
