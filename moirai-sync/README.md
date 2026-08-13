# moirai-sync

[![crates.io](https://img.shields.io/crates/v/moirai-sync.svg)](https://crates.io/crates/moirai-sync)
[![docs.rs](https://docs.rs/moirai-sync/badge.svg)](https://docs.rs/moirai-sync)

Synchronization primitives for the [Moirai](https://github.com/ryancinsight/Moirai)
concurrency library. The crate re-exports `std::sync`'s `Mutex`, `RwLock`,
`Condvar`, and `Barrier` unchanged and adds only primitives whose behavior std
does not provide:

- `FutexMutex` — adaptive spinning with futex-backed waiting on Linux.
- `WaitGroup` — Go-style completion counter.
- `ConcurrentHashMap` — segment-based locking, with poisoning surfaced as a
  typed `SegmentPoisoned` error rather than a panic.
- `ShardedResourcePool` — sharded pooling of reusable resources.
- `LockFreeStack` (re-exported from `moirai-core`) and `AtomicCounter`
  (re-exported from `moirai-utils`).

```toml
[dependencies]
moirai-sync = "0.5"
```

```rust
use moirai_sync::{FutexMutex, WaitGroup};

let mutex = FutexMutex::new(0u32);
{
    let mut guard = mutex.lock(); // returns the guard directly, not a Result
    *guard += 1;
}

let wg = WaitGroup::new();
wg.add(1);
wg.done();
wg.wait();
```

Full documentation: <https://docs.rs/moirai-sync>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
