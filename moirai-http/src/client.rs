//! HTTP client orchestration across redirects, retries, framing, and pooling.

use std::borrow::Cow;
use std::io;
use std::time::Duration;

use moirai_async::timer::timeout;
use moirai_tls::TlsConnector;

use crate::codec::{read_response, write_request, DEFAULT_MAX_RESPONSE_BYTES};
use crate::conn::Conn;
use crate::pool::IdlePool;
use crate::redirect::{
    forwarded_headers, is_redirect, parse_url, redirects_to_get, resolve_redirect,
};
use crate::{Origin, Response};

const DEFAULT_MAX_IDLE_PER_ORIGIN: usize = 8;
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(300);
const DEFAULT_MAX_REDIRECTS: usize = 10;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Async HTTP/1.1 client with bounded redirects and connection reuse.
pub struct HttpClient {
    tls: TlsConnector,
    pool: IdlePool<Origin, Conn>,
    max_idle_per_host: usize,
    idle_timeout: Duration,
    max_redirects: usize,
    request_timeout: Duration,
    max_response_bytes: usize,
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpClient {
    /// Create a client using Mozilla roots and bounded default policies.
    ///
    /// Defaults are eight idle connections per origin, a five-minute idle
    /// lifetime, ten followed redirects, a 30-second logical-request deadline,
    /// and the response limit documented by [`DEFAULT_MAX_RESPONSE_BYTES`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_tls(TlsConnector::with_webpki_roots())
    }

    /// Create a client with a caller-supplied TLS connector.
    #[must_use]
    pub fn with_tls(tls: TlsConnector) -> Self {
        Self {
            tls,
            pool: IdlePool::default(),
            max_idle_per_host: DEFAULT_MAX_IDLE_PER_ORIGIN,
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
            max_redirects: DEFAULT_MAX_REDIRECTS,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        }
    }

    /// Set the deadline for the complete logical request.
    ///
    /// The deadline covers connection acquisition, a stale pooled-connection
    /// retry, every redirect hop, request writes, and response reads.
    pub fn set_timeout(&mut self, duration: Duration) {
        self.request_timeout = duration;
    }

    /// Set the maximum bytes buffered while parsing one response.
    ///
    /// The limit includes response headers and body. A peer exceeding it is
    /// rejected with [`io::ErrorKind::InvalidData`].
    pub fn set_max_response_bytes(&mut self, bytes: usize) {
        self.max_response_bytes = bytes;
    }

    /// Set the maximum idle connections retained per origin.
    ///
    /// Zero disables retention without creating empty origin buckets.
    pub fn set_max_idle_per_host(&mut self, connections: usize) {
        self.max_idle_per_host = connections;
    }

    /// Set how long a pooled connection remains eligible for reuse.
    ///
    /// Expiry is checked when the pool is accessed, avoiding a background task.
    /// Zero disables reuse of connections subsequently returned to the pool.
    pub fn set_idle_timeout(&mut self, duration: Duration) {
        self.idle_timeout = duration;
    }

    /// Set the maximum redirect hops followed by one logical request.
    ///
    /// Zero rejects the first redirect that carries a `Location` field.
    pub fn set_max_redirects(&mut self, redirects: usize) {
        self.max_redirects = redirects;
    }

    /// Perform a `GET` request.
    ///
    /// # Errors
    ///
    /// Returns URL, connection, timeout, redirect-policy, or protocol errors.
    pub async fn get(&self, url: &str, headers: &[(&str, &str)]) -> io::Result<Response> {
        self.request("GET", url, headers, None).await
    }

    /// Perform a `HEAD` request.
    ///
    /// # Errors
    ///
    /// Returns URL, connection, timeout, redirect-policy, or protocol errors.
    pub async fn head(&self, url: &str, headers: &[(&str, &str)]) -> io::Result<Response> {
        self.request("HEAD", url, headers, None).await
    }

    /// Perform `method url`, following bounded HTTP redirects.
    ///
    /// `301` and `302` rewrite POST to a bodyless GET for deployed compatibility;
    /// `303` rewrites every method except HEAD to GET; `307` and `308` preserve
    /// method and body. Destination-specific and hop-by-hop fields are removed,
    /// and credentials are not forwarded across origins. A redirect response
    /// without `Location` is returned to the caller unchanged.
    ///
    /// A failed pooled exchange is retried once on a fresh connection only for
    /// idempotent methods. The single configured timeout encloses retries and
    /// redirects rather than restarting at each hop.
    ///
    /// # Errors
    ///
    /// Returns URL, connection, timeout, redirect-limit, redirect-location, or
    /// protocol errors.
    pub async fn request(
        &self,
        method: &str,
        url: &str,
        headers: &[(&str, &str)],
        body: Option<&[u8]>,
    ) -> io::Result<Response> {
        match timeout(
            self.request_timeout,
            self.request_following_redirects(method, url, headers, body),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "logical HTTP request timed out",
            )),
        }
    }

    async fn request_following_redirects<'a>(
        &self,
        method: &'a str,
        url: &'a str,
        headers: &'a [(&'a str, &'a str)],
        body: Option<&'a [u8]>,
    ) -> io::Result<Response> {
        let mut current_method = Cow::Borrowed(method);
        let mut current_url = Cow::Borrowed(url);
        let mut current_headers: Option<Vec<(&str, &str)>> = None;
        let mut current_body = body;
        let mut redirects_followed = 0usize;

        loop {
            let request_headers = current_headers.as_deref().unwrap_or(headers);
            let response = self
                .request_once_url(
                    current_method.as_ref(),
                    current_url.as_ref(),
                    request_headers,
                    current_body,
                )
                .await?;
            if !is_redirect(response.status) {
                return Ok(response);
            }
            let Some(location) = response.header("location") else {
                return Ok(response);
            };
            if redirects_followed >= self.max_redirects {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("redirect limit of {} exceeded", self.max_redirects),
                ));
            }

            let (origin, _) = parse_url(current_url.as_ref())?;
            let next_url = resolve_redirect(current_url.as_ref(), location)?;
            let (next_origin, _) = parse_url(&next_url)?;
            let body_dropped = redirects_to_get(response.status, current_method.as_ref());
            let next_headers =
                forwarded_headers(request_headers, next_origin != origin, body_dropped);
            if body_dropped {
                current_method = Cow::Borrowed("GET");
                current_body = None;
            }
            current_headers = Some(next_headers);
            current_url = Cow::Owned(next_url);
            redirects_followed = redirects_followed.checked_add(1).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "redirect counter overflow")
            })?;
        }
    }

    async fn request_once_url(
        &self,
        method: &str,
        url: &str,
        headers: &[(&str, &str)],
        body: Option<&[u8]>,
    ) -> io::Result<Response> {
        let (origin, path) = parse_url(url)?;
        if let Some(connection) = self.pool.take(&origin, self.idle_timeout) {
            match self
                .try_once(connection, method, &origin, &path, headers, body)
                .await
            {
                Ok(response) => return Ok(response),
                Err(pooled_error) if !is_idempotent(method) => return Err(pooled_error),
                Err(pooled_error) => {
                    let connection = Conn::connect(&origin, &self.tls)
                        .await
                        .map_err(|error| with_retry_context(error, &pooled_error))?;
                    return self
                        .try_once(connection, method, &origin, &path, headers, body)
                        .await
                        .map_err(|error| with_retry_context(error, &pooled_error));
                }
            }
        }

        let connection = Conn::connect(&origin, &self.tls).await?;
        self.try_once(connection, method, &origin, &path, headers, body)
            .await
    }

    async fn try_once(
        &self,
        mut connection: Conn,
        method: &str,
        origin: &Origin,
        path: &str,
        headers: &[(&str, &str)],
        body: Option<&[u8]>,
    ) -> io::Result<Response> {
        let host = origin.host_header();
        write_request(&mut connection, method, &host, path, headers, body).await?;
        let response = read_response(
            &mut connection,
            method.eq_ignore_ascii_case("HEAD"),
            self.max_response_bytes,
        )
        .await?;
        if response.keep_alive {
            self.pool.put(origin, connection, self.max_idle_per_host);
        }
        Ok(response)
    }
}

fn is_idempotent(method: &str) -> bool {
    ["GET", "HEAD", "OPTIONS", "TRACE", "PUT", "DELETE"]
        .iter()
        .any(|candidate| method.eq_ignore_ascii_case(candidate))
}

fn with_retry_context(error: io::Error, pooled_error: &io::Error) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("{error} (retry after stale pooled connection failed with: {pooled_error})"),
    )
}

#[cfg(test)]
mod tests;
