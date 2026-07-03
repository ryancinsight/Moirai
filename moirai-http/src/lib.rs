//! Minimal async HTTP/1.1 client for the Moirai runtime.
//!
//! Runs over Moirai async sockets and [`moirai_tls`] — **no Tokio**. Scope is the
//! request shapes object-storage clients need (`GET` with `Range`, `HEAD`, small
//! `PUT`/`POST` bodies): Content-Length and chunked response framing, a bounded
//! keep-alive connection pool, and per-request timeouts. HTTP/2 is out of scope
//! (ADR-015). Vendor protocols (e.g. S3 SigV4) are built by callers on top of this
//! — this crate knows HTTP, not S3.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod codec;
pub mod conn;

pub use codec::Response;
pub use conn::Origin;

use std::collections::HashMap;
use std::io;
use std::str::FromStr;
use std::sync::Mutex;
use std::time::Duration;

use codec::{read_response, write_request, DEFAULT_MAX_RESPONSE_BYTES};
use conn::Conn;
use moirai_async::timer::timeout;
use moirai_tls::TlsConnector;

/// Async HTTP/1.1 client with a bounded keep-alive connection pool.
pub struct HttpClient {
    tls: TlsConnector,
    pool: Mutex<HashMap<Origin, Vec<Conn>>>,
    max_idle_per_host: usize,
    request_timeout: Duration,
    max_response_bytes: usize,
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpClient {
    /// New client trusting the Mozilla root CA set, 8 idle conns/host, 30s timeout.
    #[must_use]
    pub fn new() -> Self {
        Self::with_tls(TlsConnector::with_webpki_roots())
    }

    /// New client with a caller-supplied TLS connector (e.g. custom roots).
    #[must_use]
    pub fn with_tls(tls: TlsConnector) -> Self {
        Self {
            tls,
            pool: Mutex::new(HashMap::new()),
            max_idle_per_host: 8,
            request_timeout: Duration::from_secs(30),
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        }
    }

    /// Set the per-request timeout (applied to the full write+read cycle).
    pub fn set_timeout(&mut self, dur: Duration) {
        self.request_timeout = dur;
    }

    /// Set the maximum bytes buffered while parsing one response (headers +
    /// body). Responses exceeding this — including a peer trickling bytes or
    /// advertising an oversized Content-Length — are rejected with `InvalidData`.
    pub fn set_max_response_bytes(&mut self, n: usize) {
        self.max_response_bytes = n;
    }

    /// Set the maximum idle connections retained per origin.
    pub fn set_max_idle_per_host(&mut self, n: usize) {
        self.max_idle_per_host = n;
    }

    /// `GET url`.
    ///
    /// # Errors
    /// Propagates connection, timeout, and protocol errors.
    pub async fn get(&self, url: &str, headers: &[(&str, &str)]) -> io::Result<Response> {
        self.request("GET", url, headers, None).await
    }

    /// `HEAD url`.
    ///
    /// # Errors
    /// Propagates connection, timeout, and protocol errors.
    pub async fn head(&self, url: &str, headers: &[(&str, &str)]) -> io::Result<Response> {
        self.request("HEAD", url, headers, None).await
    }

    /// Perform `method url` with optional `body`.
    ///
    /// Reuses a pooled connection when available. If the pooled attempt fails
    /// (typically a server-closed idle socket), the request is retried once on
    /// a fresh connection **only for idempotent methods** (RFC 9110 §9.2.2:
    /// GET, HEAD, OPTIONS, TRACE, PUT, DELETE). A non-idempotent method (POST,
    /// PATCH) is never re-sent: the failed pooled attempt may already have
    /// executed server-side, so its error is surfaced to the caller instead.
    /// When the idempotent retry also fails, the returned error carries the
    /// discarded pooled-attempt error as its message context.
    ///
    /// # Errors
    /// Propagates URL-parse, connection, timeout, and protocol errors.
    pub async fn request(
        &self,
        method: &str,
        url: &str,
        headers: &[(&str, &str)],
        body: Option<&[u8]>,
    ) -> io::Result<Response> {
        let (origin, path) = parse_url(url)?;

        // Fast path: reuse a pooled connection.
        if let Some(conn) = self.take_pooled(&origin) {
            match self
                .try_once(conn, method, &origin, &path, headers, body)
                .await
            {
                Ok(resp) => return Ok(resp),
                // Retrying is sound only when a duplicate execution is
                // harmless: the pooled socket may have carried the request
                // bytes before failing.
                Err(pooled_err) if !is_idempotent(method) => return Err(pooled_err),
                Err(pooled_err) => {
                    let conn = Conn::connect(&origin, &self.tls).await?;
                    return self
                        .try_once(conn, method, &origin, &path, headers, body)
                        .await
                        .map_err(|retry_err| {
                            io::Error::new(
                                retry_err.kind(),
                                format!(
                                    "{retry_err} (retry after stale pooled connection failed \
                                     with: {pooled_err})"
                                ),
                            )
                        });
                }
            }
        }

        let conn = Conn::connect(&origin, &self.tls).await?;
        self.try_once(conn, method, &origin, &path, headers, body)
            .await
    }

    async fn try_once(
        &self,
        mut conn: Conn,
        method: &str,
        origin: &Origin,
        path: &str,
        headers: &[(&str, &str)],
        body: Option<&[u8]>,
    ) -> io::Result<Response> {
        let is_head = method.eq_ignore_ascii_case("HEAD");
        let host_header = origin.host_header();
        let exchange = async {
            write_request(&mut conn, method, &host_header, path, headers, body).await?;
            read_response(&mut conn, is_head, self.max_response_bytes).await
        };
        let resp = match timeout(self.request_timeout, exchange).await {
            Ok(result) => result?,
            Err(_) => return Err(io::Error::new(io::ErrorKind::TimedOut, "request timed out")),
        };
        if resp.keep_alive {
            self.put_pooled(origin, conn);
        }
        Ok(resp)
    }

    fn take_pooled(&self, origin: &Origin) -> Option<Conn> {
        let mut pool = self.pool.lock().expect("http pool mutex poisoned");
        pool.get_mut(origin).and_then(Vec::pop)
    }

    fn put_pooled(&self, origin: &Origin, conn: Conn) {
        let mut pool = self.pool.lock().expect("http pool mutex poisoned");
        let bucket = pool.entry(origin.clone()).or_default();
        if bucket.len() < self.max_idle_per_host {
            bucket.push(conn);
        }
    }
}

/// Idempotent request methods per RFC 9110 §9.2.2; only these may be safely
/// re-sent after a failed pooled attempt whose bytes may have reached the peer.
fn is_idempotent(method: &str) -> bool {
    ["GET", "HEAD", "OPTIONS", "TRACE", "PUT", "DELETE"]
        .iter()
        .any(|m| method.eq_ignore_ascii_case(m))
}

/// Parse an absolute `http`/`https` URL into an [`Origin`] and request target
/// (path + query, defaulting to `/`).
fn parse_url(url: &str) -> io::Result<(Origin, String)> {
    let uri = http::Uri::from_str(url)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("bad URL: {e}")))?;
    let secure = match uri.scheme_str() {
        Some("https") => true,
        Some("http") => false,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported scheme: {other:?}"),
            ))
        }
    };
    let host = uri
        .host()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "URL missing host"))?
        .to_string();
    let port = uri.port_u16().unwrap_or(if secure { 443 } else { 80 });
    let path = uri
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| "/".to_string());
    Ok((Origin { secure, host, port }, path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read as _, Write as _};
    use std::net::TcpListener;
    use std::time::Instant;

    /// Read one HTTP request head (through the blank line) from `stream`.
    fn read_request_head(stream: &mut std::net::TcpStream) {
        let mut buf = Vec::new();
        let mut byte = [0u8; 1];
        while !buf.ends_with(b"\r\n\r\n") {
            match stream.read(&mut byte) {
                Ok(0) => break,
                Ok(_) => buf.push(byte[0]),
                Err(_) => break,
            }
        }
    }

    /// Seed the client's pool with a connection the server has already closed.
    fn seed_stale_pooled_conn(client: &HttpClient, origin: &Origin, listener: &TcpListener) {
        let conn = moirai::block_on(Conn::connect(origin, &client.tls))
            .expect("pooled connection must establish");
        let (stale, _) = listener.accept().expect("server must accept pooled conn");
        drop(stale); // server closes the idle socket -> pooled conn is stale
        client.put_pooled(origin, conn);
    }

    #[test]
    fn idempotent_get_retries_stale_pooled_connection_and_succeeds() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let url = format!("http://127.0.0.1:{port}/x");

        let client = HttpClient::new();
        let origin = Origin {
            secure: false,
            host: "127.0.0.1".to_string(),
            port,
        };
        seed_stale_pooled_conn(&client, &origin, &listener);

        // Second accept serves the retried request.
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("retry connection must arrive");
            read_request_head(&mut stream);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                .expect("response write");
        });

        let resp = moirai::block_on(client.get(&url, &[]))
            .expect("GET must transparently retry the stale pooled connection");
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, b"ok");
        server.join().expect("server thread");
    }

    #[test]
    fn non_idempotent_post_is_not_retried_after_stale_pooled_failure() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let url = format!("http://127.0.0.1:{port}/submit");

        let client = HttpClient::new();
        let origin = Origin {
            secure: false,
            host: "127.0.0.1".to_string(),
            port,
        };
        seed_stale_pooled_conn(&client, &origin, &listener);

        // Watch for any further connection: a retry would open one.
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");

        let err = moirai::block_on(client.request("POST", &url, &[], Some(b"payload")))
            .expect_err("POST over a stale pooled connection must surface the error, not retry");
        // The pooled socket was closed by the server before any response; the
        // exact kind depends on whether the peer's close surfaced as EOF or
        // RST at the moment of the exchange, but it must be a connection-level
        // failure, not a parsed response.
        assert!(
            matches!(
                err.kind(),
                io::ErrorKind::UnexpectedEof
                    | io::ErrorKind::ConnectionReset
                    | io::ErrorKind::ConnectionAborted
                    | io::ErrorKind::BrokenPipe
            ),
            "unexpected error kind {:?}: {err}",
            err.kind()
        );

        // No second connection may be opened for a non-idempotent method.
        let deadline = Instant::now() + Duration::from_millis(300);
        while Instant::now() < deadline {
            match listener.accept() {
                Ok(_) => panic!("non-idempotent request must not be retried on a new connection"),
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => panic!("listener poll failed: {e}"),
            }
        }
    }
}
