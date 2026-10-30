# moirai-utils

[![crates.io](https://img.shields.io/crates/v/moirai-utils.svg)](https://crates.io/crates/moirai-utils)
[![docs.rs](https://docs.rs/moirai-utils/badge.svg)](https://docs.rs/moirai-utils)

Low-level utilities shared across the [Moirai](https://github.com/ryancinsight/Moirai)
concurrency library, organized by domain:

- `cache` — `CacheAligned<T>`, `align_to_cache_line`, `CACHE_LINE_SIZE`, and
  `DESTRUCTIVE_INTERFERENCE_SIZE`.
- `atomic` — `AtomicCounter` for lock-free counting.
- `queue` — `LockFreeQueue`.
- `memory` — `prefetch_read` / `prefetch_write`.
- `simd` — `SimdScalar` / `SimdReal` and `has_native_vector_path`.

```toml
[dependencies]
moirai-utils = "0.12"
```

```rust
use moirai_utils::{AtomicCounter, CacheAligned, DESTRUCTIVE_INTERFERENCE_SIZE};
use std::mem::align_of;
use std::sync::atomic::AtomicUsize;

// Keep two hot indices off one cache line to avoid false sharing.
struct Ring {
    head: CacheAligned<AtomicUsize>,
    tail: CacheAligned<AtomicUsize>,
}
assert_eq!(
    align_of::<CacheAligned<AtomicUsize>>(),
    DESTRUCTIVE_INTERFERENCE_SIZE,
);
let _ring = Ring {
    head: CacheAligned::new(AtomicUsize::new(0)),
    tail: CacheAligned::new(AtomicUsize::new(0)),
};

let counter = AtomicCounter::new();
assert_eq!(counter.increment(), 0); // returns the previous value
assert_eq!(counter.get(), 1);
```

## `no_std`

Disable default features to use the `atomic`, `cache`, `memory`, `queue`, and
`result_cell` modules with `core` and `alloc`:

```toml
[dependencies]
moirai-utils = { version = "0.12", default-features = false }
```

The final binary supplies an allocator for allocation-backed queues. The SIMD
module and process-global SIMD counter require `std`; queue contention uses
processor spin hints instead of an operating-system thread yield when `std` is
absent. No thread or timing capability is synthesized in this configuration.

Full documentation: <https://docs.rs/moirai-utils>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
