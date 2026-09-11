//! Typestate connections and bounded request bodies.

use std::io;

use moirai_async::io::AsyncWriteExt;
use moirai_async::net::TcpStream;
use moirai_async::timer::timeout;

use super::config::ServerConfig;
use super::response::{HttpResponse, encode_response_head};
use crate::request::{HttpRequestHead, read_request_head};

/// A connection that has not read its request yet.
#[derive(Debug)]
pub struct AwaitingRequest;

/// A connection that has read one request and may write one response.
#[derive(Debug)]
pub struct AwaitingResponse {
    suppress_body: bool,
}

/// A one-shot HTTP connection whose typestate controls its lifecycle.
pub struct HttpConnection<State> {
    pub(super) stream: TcpStream,
    pub(super) config: ServerConfig,
    pub(super) prefix: Vec<u8>,
    pub(super) state: State,
}

impl HttpConnection<AwaitingRequest> {
    /// Read one bounded request and transition to [`AwaitingResponse`].
    ///
    /// The connection is consumed so a failed or timed-out read cannot be
    /// retried at an ambiguous wire position.
    ///
    /// # Errors
    /// Returns malformed, oversized, transfer-encoded, truncated, timed-out,
    /// or transport failures.
    pub async fn read_request(
        mut self,
    ) -> io::Result<(HttpRequest, HttpConnection<AwaitingResponse>)> {
        let deadline = self.config.request_timeout;
        match timeout(deadline, self.read_request_inner()).await {
            Ok(Ok(request)) => {
                let suppress_body = request.method() == "HEAD";
                let connection = HttpConnection {
                    stream: self.stream,
                    config: self.config,
                    prefix: Vec::new(),
                    state: AwaitingResponse { suppress_body },
                };
                Ok((request, connection))
            }
            Ok(Err(error)) => Err(error),
            Err(_) => Err(timed_out("HTTP request read")),
        }
    }

    async fn read_request_inner(&mut self) -> io::Result<HttpRequest> {
        let (head, prefix) = read_request_head(
            &mut self.stream,
            self.config.max_header_bytes,
            self.config.max_header_count,
        )
        .await?;
        self.prefix = prefix;
        let body_length = request_body_length(&head, self.config.max_body_bytes)?;
        let mut body = Vec::new();
        body.try_reserve(body_length)
            .map_err(|_| invalid_data("HTTP request body allocation exceeds available memory"))?;

        if self.prefix.len() > body_length {
            return Err(invalid_data(
                "HTTP request contains bytes beyond Content-Length",
            ));
        }
        if !self.prefix.is_empty() {
            body.extend_from_slice(&self.prefix);
        }

        let mut remaining = body_length
            .checked_sub(self.prefix.len())
            .ok_or_else(|| invalid_data("HTTP request body length underflowed"))?;
        while remaining != 0 {
            let mut chunk = [0_u8; 8192];
            let read_length = remaining.min(chunk.len());
            let target = chunk
                .get_mut(..read_length)
                .ok_or_else(|| invalid_data("HTTP request read window exceeded its bound"))?;
            let count = self.stream.read(target).await?;
            if count == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "HTTP request body ended before Content-Length",
                ));
            }
            let bytes = chunk
                .get(..count)
                .ok_or_else(|| invalid_data("HTTP request read exceeded its bound"))?;
            body.extend_from_slice(bytes);
            remaining = remaining
                .checked_sub(count)
                .ok_or_else(|| invalid_data("HTTP request body length underflowed"))?;
        }
        self.prefix.clear();
        Ok(HttpRequest { head, body })
    }
}

impl HttpConnection<AwaitingResponse> {
    /// Write one bounded response, flush it, and close the connection.
    ///
    /// The connection is consumed so a partial write or deadline cannot be
    /// retried. A `HEAD` request keeps the declared `Content-Length` but omits
    /// the response body on the wire.
    ///
    /// # Errors
    /// Returns invalid response, oversized response, timed-out, or transport
    /// failures.
    pub async fn write_response(self, response: HttpResponse) -> io::Result<()> {
        let deadline = self.config.request_timeout;
        match timeout(deadline, self.write_response_inner(response)).await {
            Ok(result) => result,
            Err(_) => Err(timed_out("HTTP response write")),
        }
    }

    async fn write_response_inner(self, response: HttpResponse) -> io::Result<()> {
        let HttpConnection {
            mut stream,
            config,
            state,
            prefix: _,
        } = self;
        if response.headers.len() > config.max_header_count.saturating_sub(2) {
            return Err(invalid_data(
                "HTTP response header count exceeds configured bound",
            ));
        }
        let head = encode_response_head(&response, config.max_response_bytes)?;
        let total_size = head
            .len()
            .checked_add(response.body.len())
            .ok_or_else(|| invalid_data("HTTP response size overflows its bound"))?;
        if total_size > config.max_response_bytes {
            return Err(invalid_data("HTTP response exceeds configured byte bound"));
        }

        stream.write_all(&head).await?;
        if !state.suppress_body && !response.body.is_empty() {
            stream.write_all(&response.body).await?;
        }
        stream.flush().await?;
        stream.shutdown().await
    }
}

/// A validated HTTP request with an owned bounded body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    head: HttpRequestHead,
    body: Vec<u8>,
}

impl HttpRequest {
    /// Return the request method.
    #[must_use]
    pub fn method(&self) -> &str {
        self.head.method()
    }

    /// Return the origin-form request target.
    #[must_use]
    pub fn target(&self) -> &str {
        self.head.target()
    }

    /// Return the HTTP version token.
    #[must_use]
    pub fn version(&self) -> &str {
        self.head.version()
    }

    /// Return the first header value matching `name` case-insensitively.
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.head.header(name)
    }

    /// Return all request headers in receive order.
    #[must_use]
    pub fn headers(&self) -> &[(String, String)] {
        self.head.headers()
    }

    /// Return the complete request body.
    #[must_use]
    pub fn body(&self) -> &[u8] {
        &self.body
    }
}

fn request_body_length(head: &HttpRequestHead, max_body_bytes: usize) -> io::Result<usize> {
    if head.header("transfer-encoding").is_some() {
        return Err(invalid_data(
            "transfer-encoded HTTP request bodies are not supported",
        ));
    }
    let Some(value) = head.header("content-length") else {
        return Ok(0);
    };
    let length = value
        .parse::<usize>()
        .map_err(|_| invalid_data("HTTP Content-Length is not a decimal byte count"))?;
    if length > max_body_bytes {
        return Err(invalid_data(
            "HTTP request body exceeds configured byte bound",
        ));
    }
    Ok(length)
}

fn invalid_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn timed_out(operation: &str) -> io::Error {
    io::Error::new(io::ErrorKind::TimedOut, operation)
}
