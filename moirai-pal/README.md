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

The WASM module owns the browser boundary used by Atlas applications. `WebDocument`
and `WebElement` provide bounded DOM updates, input/select values, checked
checkbox/radio state, disabled button/input/select state, modal dialog lifecycle
and focus control. Pointer events expose their browser identifier, and elements
can capture, query, and release that identifier through the same owned seam.
Pointer events also expose a value snapshot with normalized device type,
viewport coordinates, button state, modifier keys and primary-pointer state.
`WebEventListener` removes its callback registration when dropped. `spawn_local` uses the browser event loop for futures; applications
do not create a second executor or retain detached JavaScript closures.
`spawn_local_with_handle` adds
a single-owner `LocalTaskHandle`; cancelling or dropping it wakes the task and
drops its child future, releasing a pending PAL receive or timer.

```toml
[dependencies]
moirai-pal = "0.6"
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
