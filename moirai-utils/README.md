# moirai-utils

[![crates.io](https://img.shields.io/crates/v/moirai-utils.svg)](https://crates.io/crates/moirai-utils)
[![docs.rs](https://docs.rs/moirai-utils/badge.svg)](https://docs.rs/moirai-utils)

Low-level utilities shared across the [Moirai](https://github.com/ryancinsight/Moirai)
concurrency library, organized by domain:

- `cache` — `CacheAligned<T>` (`#[repr(align(64))]`), `align_to_cache_line`,
  `CACHE_LINE_SIZE`.
- `atomic` — `AtomicCounter` for lock-free counting.
- `queue` — `LockFreeQueue`.
- `memory` — `prefetch_read` / `prefetch_write`.
- `simd` — `SimdScalar` / `SimdReal` and `has_native_vector_path`.

```toml
[dependencies]
moirai-utils = "0.5"
```

```rust
use moirai_utils::{AtomicCounter, CacheAligned};
use std::mem::align_of;
use std::sync::atomic::AtomicUsize;

// Keep two hot indices off one cache line to avoid false sharing.
struct Ring {
    head: CacheAligned<AtomicUsize>,
    tail: CacheAligned<AtomicUsize>,
}
assert_eq!(align_of::<CacheAligned<AtomicUsize>>(), 64);
let _ring = Ring {
    head: CacheAligned::new(AtomicUsize::new(0)),
    tail: CacheAligned::new(AtomicUsize::new(0)),
};

let counter = AtomicCounter::new();
assert_eq!(counter.increment(), 0); // returns the previous value
assert_eq!(counter.get(), 1);
```

Full documentation: <https://docs.rs/moirai-utils>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
