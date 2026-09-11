# moirai-http

[![crates.io](https://img.shields.io/crates/v/moirai-http.svg)](https://crates.io/crates/moirai-http)
[![docs.rs](https://docs.rs/moirai-http/badge.svg)](https://docs.rs/moirai-http)

Minimal async HTTP/1.1 client, bounded one-shot server transport, and bounded WebSocket service for the
[Moirai](https://github.com/ryancinsight/Moirai) runtime. Runs over Moirai async sockets and
[`moirai-tls`](https://crates.io/crates/moirai-tls) — **no Tokio**.

Scope is the request shapes object-storage clients need: `GET` with `Range`,
`HEAD`, and small `PUT`/`POST` bodies, with Content-Length and chunked response
framing, a bounded keep-alive connection pool with access-triggered idle
eviction, capped RFC 9110 redirects, and one deadline across each logical
request. HTTP/2 is out of scope. Vendor protocols (for example S3 SigV4) are
built by callers on top of this — the crate knows HTTP, not S3.

```toml
[dependencies]
moirai-http = "0.6"
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
`set_max_idle_per_host`, plus `set_idle_timeout` and `set_max_redirects` for
connection reuse and redirect chains. Redirects resolve relative references per
RFC 3986, never forward credentials across origins, and preserve methods and
bodies for 307/308 responses.

Full documentation: <https://docs.rs/moirai-http>

## HTTP server transport

`HttpServer` binds a Moirai TCP listener and accepts one request per
connection. `ServerConfig` bounds headers, request bodies, responses,
connections, and each read/write deadline. The typestate flow makes the
request-then-response lifecycle explicit:

```rust,no_run
use moirai_http::{HttpResponse, HttpServer, ServerConfig};

async fn example() -> std::io::Result<()> {
    let server = HttpServer::bind("127.0.0.1:8080", ServerConfig::default()).await?;
    serve_once(&server).await
}

async fn serve_once(server: &HttpServer) -> std::io::Result<()> {
    let connection = server.accept().await?;
    let (request, connection) = connection.read_request().await?;
    let response = HttpResponse::new(200, request.body().to_vec())?;
    connection.write_response(response).await
}
```

The transport owns framing and closes after the response. It does not select
routes, authorize origins or sessions, generate markup, or parse DICOM; those
policies remain in Metis and RITK respectively.

## WebSocket service

The service side accepts one bounded HTTP/1.1 upgrade (including the required
`Host` header) and exposes complete binary messages over the existing Moirai
async stream. It rejects unmasked, fragmented, reserved, text and
non-minimally encoded frames, handles ping/pong and close, and wraps header,
message and frame operations in finite deadlines. A frame timeout or partial
I/O error terminalizes the stream so callers cannot retry from an ambiguous
wire position. The optional browser `Origin` header is returned to the
consumer; `accept_websocket_with_validator` additionally lets the consumer
reject it before the `101` response, while authorization remains the
consumer's responsibility.

```rust,no_run
use moirai_http::{accept_websocket, WebSocketConfig};

async fn serve<S>(stream: S) -> std::io::Result<()>
where
    S: moirai_async::io::AsyncRead + moirai_async::io::AsyncWrite + Unpin,
{
    let (mut socket, upgrade) = accept_websocket(stream, WebSocketConfig::default()).await?;
    let _origin = upgrade.origin();
    let message = socket.recv_message().await?;
    socket.send_binary(&message).await?;
    Ok(())
}
```

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
