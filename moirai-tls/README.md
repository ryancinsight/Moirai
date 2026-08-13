# moirai-tls

[![crates.io](https://img.shields.io/crates/v/moirai-tls.svg)](https://crates.io/crates/moirai-tls)
[![docs.rs](https://docs.rs/moirai-tls/badge.svg)](https://docs.rs/moirai-tls)

Async TLS client for the [Moirai](https://github.com/ryancinsight/Moirai)
runtime. Drives the sans-I/O [`rustls`](https://docs.rs/rustls) state machine
(via the runtime-agnostic `futures-rustls` adapter) over a Moirai async socket —
**no Tokio**.

Moirai's `AsyncRead` / `AsyncWrite` traits are signature-identical to
`futures-io`, so the bridge is two zero-cost newtypes; all TLS cryptography and
record framing are delegated to the audited `rustls` stack. Cryptography comes
from [`moirai-crypto`](https://crates.io/crates/moirai-crypto), a pure-Rust
RustCrypto provider, so no C toolchain is required.

```toml
[dependencies]
moirai-tls = "0.5"
```

```rust
use moirai_tls::{ServerName, TlsConnector};

async fn connect<S>(sock: S)
where
    S: moirai_async::io::AsyncRead + moirai_async::io::AsyncWrite + Unpin,
{
    let connector = TlsConnector::with_webpki_roots();
    let domain = ServerName::try_from("example.com").unwrap();
    let _tls = connector.connect(domain, sock).await.unwrap();
}
```

Full documentation: <https://docs.rs/moirai-tls>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
