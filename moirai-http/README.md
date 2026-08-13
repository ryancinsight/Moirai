# moirai-http

[![crates.io](https://img.shields.io/crates/v/moirai-http.svg)](https://crates.io/crates/moirai-http)
[![docs.rs](https://docs.rs/moirai-http/badge.svg)](https://docs.rs/moirai-http)

Minimal async HTTP/1.1 client for the [Moirai](https://github.com/ryancinsight/Moirai)
runtime. Runs over Moirai async sockets and
[`moirai-tls`](https://crates.io/crates/moirai-tls) — **no Tokio**.

Scope is the request shapes object-storage clients need: `GET` with `Range`,
`HEAD`, and small `PUT`/`POST` bodies, with Content-Length and chunked response
framing, a bounded keep-alive connection pool, and per-request timeouts. HTTP/2
is out of scope. Vendor protocols (for example S3 SigV4) are built by callers on
top of this — the crate knows HTTP, not S3.

```toml
[dependencies]
moirai-http = "0.5"
```

```rust
use moirai_http::HttpClient;

async fn fetch(url: &str) -> std::io::Result<()> {
    let client = HttpClient::new();
    let response = client.get(url, &[("accept", "application/json")]).await?;

    println!("status {}", response.status);
    println!("length {:?}", response.header("content-length"));
    println!("{} body bytes", response.body.len());
    Ok(())
}
```

`Response` exposes `status`, `headers` (lowercased, in receive order), `body`,
and `keep_alive`, plus `header(name)` for a case-insensitive lookup. Limits are
configured on the client: `set_timeout`, `set_max_response_bytes`, and
`set_max_idle_per_host`.

Full documentation: <https://docs.rs/moirai-http>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
