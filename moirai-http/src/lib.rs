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
    /// Reuses a pooled connection when available, transparently retrying once on a
    /// fresh connection if the pooled one was stale (server-closed idle socket).
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

        // Fast path: reuse a pooled connection; a failure here is treated as a
        // stale socket and retried once on a fresh connection.
        if let Some(conn) = self.take_pooled(&origin) {
            if let Ok(resp) = self
                .try_once(conn, method, &origin, &path, headers, body)
                .await
            {
                return Ok(resp);
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
