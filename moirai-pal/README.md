# moirai-pal

[![crates.io](https://img.shields.io/crates/v/moirai-pal.svg)](https://crates.io/crates/moirai-pal)
[![docs.rs](https://docs.rs/moirai-pal/badge.svg)](https://docs.rs/moirai-pal)

Platform Abstraction Layer for the [Moirai](https://github.com/ryancinsight/Moirai)
runtime's async I/O. One `Reactor` trait over the platform's native readiness
mechanism, so the async stack needs no external runtime:

| Target | Mechanism |
|--------|-----------|
| Linux | `epoll` |
| macOS / BSD | `kqueue` |
| Windows | `WSAPoll` socket readiness polling |
| WebAssembly | Web APIs via JavaScript interop |

Modules: `reactor`, `net`, `fs`, `timer`, plus the per-platform `unix`,
`windows`, and `wasm` implementations.

```toml
[dependencies]
moirai-pal = "0.5"
```

```rust
use moirai_pal::{create_reactor, Interest};

fn setup() -> std::io::Result<()> {
    let reactor = create_reactor()?;         // PlatformReactor for this target
    let interest = Interest::READABLE;       // read readiness (plus errors)
    let _ = (reactor, interest);
    Ok(())
}
```

This is a runtime-internal layer; most users reach it through
[`moirai-async`](https://crates.io/crates/moirai-async) rather than directly.

Full documentation: <https://docs.rs/moirai-pal>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
