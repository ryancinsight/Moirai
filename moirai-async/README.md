# moirai-async

[![crates.io](https://img.shields.io/crates/v/moirai-async.svg)](https://crates.io/crates/moirai-async)
[![docs.rs](https://docs.rs/moirai-async/badge.svg)](https://docs.rs/moirai-async)

Async runtime integration for the [Moirai](https://github.com/ryancinsight/Moirai)
concurrency library — the **concurrent** domain, distinct from the synchronous
data-parallel domain in `moirai-parallel`.

- `io` — `AsyncRead` / `AsyncWrite` (signature-identical to `futures-io`) with
  `AsyncReadExt` / `AsyncWriteExt`, plus `TokioCompat` / `MoiraiCompat` bridges.
- `net` — native `TcpStream`, `TcpListener`, `UdpSocket`, and connection
  pooling on Unix and Windows. Browser networking uses the WebSocket handles
  in `moirai-pal`.
- `fs` — async file operations (`read`, `write`, `File`, `FileOpenOptions`, …),
  including exact cursor-preserving `AsyncReadAt` and `AsyncLength` access.
- `timer` — `sleep`, `timeout`, `interval`, `RateLimiter`, `TimerWheel`.
- `sync` — `Mutex`, `RwLock`, `Semaphore`, `Notify`, `Broadcast`, `Watch`.
  `sync::mpsc` has bounded `channel` and synchronous `unbounded_channel` over
  one FIFO ownership and cancellation state machine. `Receiver::poll_recv`
  retains the current caller's waker until progress, replacement or cancellation.
- `executor` — native `AsyncExecutor` and `AsyncHandle`.

```toml
[dependencies]
moirai-async = "0.5"
```

```rust
use std::time::Duration;

#[moirai_async::main]
async fn main() {
    moirai_async::sleep(Duration::from_millis(10)).await;
    println!("done");
}
```

The `#[moirai_async::main]` attribute (from
[`moirai-async-macros`](https://crates.io/crates/moirai-async-macros)) rewrites
the function into a synchronous native `main` that builds a default
`AsyncExecutor` and blocks on the body.

Full documentation: <https://docs.rs/moirai-async>

The proposed 0.12.0 native transport prerequisite adds
`io::ReadBuffer` for initialized or uninitialized read destinations. It borrows
storage without allocation and maintains `filled <= initialized <= capacity`.
Appending initializes exactly the bytes supplied; advancing cannot expose
uninitialized bytes. This is a buffer API, independent of stream scheduling.
Package/release and native Quinn integration qualification remain pending.

`UdpSocket::bind_with_config` applies PAL-owned native options before returning
the socket. `UdpConfig.buffers` has independent optional `NonZeroUsize` receive
and send requests; absent requests retain kernel defaults. `buffer_sizes()`
reads effective native sizes, including caps, rounding and metadata accounting.
`UdpConfig.multicast` takes an explicit `MulticastConfig` group/interface variant
and joins that group while selecting the same outgoing interface. Invalid
native integer ranges, groups, families and interfaces return errors. See
[ADR 0081](../docs/adr/0081-applied-udp-socket-configuration.md) for the migration
from the former ignored `buffer_size` and multicast boolean.

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
