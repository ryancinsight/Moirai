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
| Windows | `WSAPoll` socket readiness polling and a thread-owned Win32 window |
| WebAssembly | Web APIs via JavaScript interop |

Modules: `reactor`, `net`, `fs`, `timer`, plus the per-platform `unix`,
`windows`, and `wasm` implementations. On Windows, `windows::window::NativeWindow`
provides a bounded message queue and ARGB software presentation surface for a
consumer-owned event loop. `NativeWindow::wait_events` adds a finite message
queue wait for event-driven hosts; waits beyond its 30-second bound are
rejected. Native IME start, preedit, commit and cancellation phases are
returned as bounded UTF-8 snapshots.

The WASM module owns the browser boundary used by Atlas applications. `WebDocument`
and `WebElement` provide bounded DOM updates, input/select values, checked
checkbox/radio state, disabled button/input/select state, modal dialog lifecycle
and focus control. Pointer events expose their browser identifier, and elements
can capture, query, and release that identifier through the same owned seam.
Pointer events also expose a value snapshot with normalized device type,
viewport coordinates, button state, modifier keys and primary-pointer state.
Wheel events expose bounded deltas with their browser unit, viewport
coordinates and the same modifier-key snapshot.
Drag events expose CSS-pixel coordinates and an owned `DropMetadata` snapshot
with at most 64 validated files; names, media types and byte sizes are bounded
before they reach an application. The seam does not read file bytes or treat a
name as a filesystem path.
Text controls expose bounded values, UTF-16 selection ranges with direction,
and bounded `InputEvent`/`CompositionEvent` metadata through owned snapshots.
Unsupported targets and browser metadata failures return explicit errors or
`None`; grapheme segmentation and editing policy stay with the application or
host layer. Native IME event production stays in the Windows provider.
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
