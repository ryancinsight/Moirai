# moirai-async-macros

[![crates.io](https://img.shields.io/crates/v/moirai-async-macros.svg)](https://crates.io/crates/moirai-async-macros)
[![docs.rs](https://docs.rs/moirai-async-macros/badge.svg)](https://docs.rs/moirai-async-macros)

The `#[main]` attribute macro for the
[Moirai](https://github.com/ryancinsight/Moirai) async runtime.

Rust's entry point cannot itself be `async`, so something has to build a runtime
and block on the future. This crate is that shim, keeping executor construction
out of user code: the annotated function is rewritten into a synchronous `main`
that builds an `AsyncExecutor` with its defaults and blocks on the original
body, preserving the function's attributes and visibility.

The macro is re-exported by
[`moirai-async`](https://crates.io/crates/moirai-async); depend on that crate
rather than on this one directly.

```toml
[dependencies]
moirai-async = "0.5"
```

```rust
#[moirai_async::main]
async fn main() {
    println!("running on the Moirai async executor");
}
```

The attribute takes no arguments. A caller needing a configured executor should
build one directly and call `block_on` instead. The generated `main` panics if
the executor cannot be created — it runs before any user code, so there is no
state to unwind.

Full documentation: <https://docs.rs/moirai-async-macros>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
