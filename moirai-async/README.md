# moirai-async

[![crates.io](https://img.shields.io/crates/v/moirai-async.svg)](https://crates.io/crates/moirai-async)
[![docs.rs](https://docs.rs/moirai-async/badge.svg)](https://docs.rs/moirai-async)

Async runtime integration for the [Moirai](https://github.com/ryancinsight/Moirai)
concurrency library — the **concurrent** domain, distinct from the synchronous
data-parallel domain in `moirai-parallel`.

- `io` — `AsyncRead` / `AsyncWrite` (signature-identical to `futures-io`) with
  `AsyncReadExt` / `AsyncWriteExt`, plus `TokioCompat` / `MoiraiCompat` bridges.
- `net` — `TcpStream`, `TcpListener`, `UdpSocket`, and connection pooling.
- `fs` — async file operations (`read`, `write`, `File`, `FileOpenOptions`, …).
- `timer` — `sleep`, `timeout`, `interval`, `RateLimiter`, `TimerWheel`.
- `sync` — `Mutex`, `RwLock`, `Semaphore`, `Notify`, `Broadcast`, `Watch`.
- `executor` — `AsyncExecutor` and `AsyncHandle`.

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
the function into a synchronous `main` that builds a default `AsyncExecutor` and
blocks on the body.

Full documentation: <https://docs.rs/moirai-async>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
