//! HTTP/1.1 request serialization and response parsing (status line, headers,
//! and body framing: Content-Length, chunked transfer-encoding, or EOF-delimited).

use std::io;

use moirai_async::io::{AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Default ceiling on the total bytes buffered while parsing one response
/// (status line + headers + body). A response — or a malicious/compromised peer
/// trickling bytes — that exceeds this is rejected rather than allowed to drive
/// unbounded allocation. Matches the transport-layer frame cap (16 MiB) scaled
/// up for whole HTTP responses.
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

/// A parsed HTTP response.
#[derive(Debug, Clone)]
pub struct Response {
    /// HTTP status code.
    pub status: u16,
    /// Response headers in receive order (name lowercased).
    pub headers: Vec<(String, String)>,
    /// Fully-read response body.
    pub body: Vec<u8>,
    /// Whether the connection may be reused (keep-alive, framed body).
    pub keep_alive: bool,
}

impl Response {
    /// First header value matching `name` (case-insensitive).
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// Serialize and write an HTTP/1.1 request, then flush.
///
/// `headers` are sent verbatim; `Host` and `Content-Length` are added when absent.
///
/// # Errors
/// Propagates write/flush failures.
pub async fn write_request<S: AsyncWrite + Unpin>(
    stream: &mut S,
    method: &str,
    host: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&[u8]>,
) -> io::Result<()> {
    let mut req = Vec::with_capacity(256);
    req.extend_from_slice(method.as_bytes());
    req.push(b' ');
    req.extend_from_slice(path.as_bytes());
    req.extend_from_slice(b" HTTP/1.1\r\n");

    let has = |n: &str| headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(n));
    if !has("host") {
        req.extend_from_slice(format!("Host: {host}\r\n").as_bytes());
    }
    for (k, v) in headers {
        req.extend_from_slice(format!("{k}: {v}\r\n").as_bytes());
    }
    if let Some(b) = body {
        if !has("content-length") {
            req.extend_from_slice(format!("Content-Length: {}\r\n", b.len()).as_bytes());
        }
    }
    req.extend_from_slice(b"\r\n");
    if let Some(b) = body {
        req.extend_from_slice(b);
    }

    stream.write_all(&req).await?;
    stream.flush().await
}

/// Buffered reader over a connection for response parsing.
struct Buffered<'a, S> {
    stream: &'a mut S,
    buf: Vec<u8>,
    pos: usize,
    /// Hard ceiling on `buf.len()`. Every read funnels through `fill`, so this is
    /// the single chokepoint bounding total allocation regardless of which body
    /// framing (Content-Length, chunked, EOF) or header stream a peer sends.
    limit: usize,
}

impl<'a, S: AsyncReadExt + Unpin> Buffered<'a, S> {
    fn new(stream: &'a mut S, limit: usize) -> Self {
        Self {
            stream,
            buf: Vec::with_capacity(8192.min(limit.max(1))),
            pos: 0,
            limit,
        }
    }

    fn available(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Read more bytes from the stream into the buffer. Returns bytes read (0 = EOF).
    ///
    /// Enforces the response-size budget: a peer cannot grow `buf` past `limit`
    /// by trickling bytes (slowloris) or by advertising a huge Content-Length /
    /// chunk size, since every byte the parser buffers passes through here.
    async fn fill(&mut self) -> io::Result<usize> {
        if self.buf.len() >= self.limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "response exceeds maximum size",
            ));
        }
        let mut tmp = [0u8; 8192];
        let n = self.stream.read(&mut tmp).await?;
        self.buf.extend_from_slice(&tmp[..n]);
        if self.buf.len() > self.limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "response exceeds maximum size",
            ));
        }
        Ok(n)
    }

    /// Read and consume one CRLF-terminated line (without the CRLF).
    async fn read_crlf_line(&mut self) -> io::Result<String> {
        loop {
            if let Some(rel) = find_crlf(&self.buf[self.pos..]) {
                let line = self.buf[self.pos..self.pos + rel].to_vec();
                self.pos += rel + 2;
                return String::from_utf8(line).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "non-UTF8 header line")
                });
            }
            if self.fill().await? == 0 {
                return Err(eof("CRLF line"));
            }
        }
    }

    /// Consume exactly `n` bytes, reading more as needed.
    async fn read_n(&mut self, n: usize) -> io::Result<Vec<u8>> {
        while self.available() < n {
            if self.fill().await? == 0 {
                return Err(eof("body"));
            }
        }
        let out = self.buf[self.pos..self.pos + n].to_vec();
        self.pos += n;
        Ok(out)
    }

    /// Read everything until the peer closes the connection.
    async fn read_to_eof(&mut self) -> io::Result<Vec<u8>> {
        while self.fill().await? != 0 {}
        Ok(self.buf[self.pos..].to_vec())
    }

    /// Decode a chunked transfer-encoded body.
    async fn read_chunked(&mut self) -> io::Result<Vec<u8>> {
        let mut body = Vec::new();
        loop {
            let line = self.read_crlf_line().await?;
            let size_field = line.split(';').next().unwrap_or("").trim();
            let size = usize::from_str_radix(size_field, 16)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad chunk size"))?;
            if size == 0 {
                // Consume optional trailers until the terminating blank line.
                while !self.read_crlf_line().await?.is_empty() {}
                break;
            }
            body.extend_from_slice(&self.read_n(size).await?);
            // Consume the CRLF that terminates the chunk data.
            let crlf = self.read_n(2).await?;
            if crlf != b"\r\n" {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "missing chunk CRLF",
                ));
            }
        }
        Ok(body)
    }
}

/// Parse a full HTTP response from `stream`. `is_head` suppresses body reading.
///
/// # Errors
/// Propagates I/O errors and returns `InvalidData` on malformed responses.
pub async fn read_response<S: AsyncReadExt + Unpin>(
    stream: &mut S,
    is_head: bool,
    max_response_bytes: usize,
) -> io::Result<Response> {
    let mut r = Buffered::new(stream, max_response_bytes);

    // Read until the header block is complete.
    let (status, headers) = loop {
        let mut header_storage = [httparse::EMPTY_HEADER; 96];
        let mut resp = httparse::Response::new(&mut header_storage);
        match resp.parse(&r.buf[r.pos..]) {
            Ok(httparse::Status::Complete(consumed)) => {
                let status = resp
                    .code
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no status code"))?;
                let headers: Vec<(String, String)> = resp
                    .headers
                    .iter()
                    .map(|h| {
                        (
                            h.name.to_ascii_lowercase(),
                            String::from_utf8_lossy(h.value).into_owned(),
                        )
                    })
                    .collect();
                r.pos += consumed;
                break (status, headers);
            }
            Ok(httparse::Status::Partial) => {
                if r.fill().await? == 0 {
                    return Err(eof("response headers"));
                }
            }
            Err(e) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("malformed response: {e}"),
                ))
            }
        }
    };

    let find = |n: &str| {
        headers
            .iter()
            .find(|(k, _)| k == n)
            .map(|(_, v)| v.as_str())
    };
    let chunked = find("transfer-encoding")
        .map(|v| v.to_ascii_lowercase().contains("chunked"))
        .unwrap_or(false);
    // RFC 9112 §6.3: a present-but-unparseable Content-Length is a framing
    // error that must fail the message (response-desync hazard) — it must NOT
    // silently degrade to read-to-EOF framing. Absent header => EOF framing.
    let content_length: Option<usize> = match find("content-length") {
        Some(v) => Some(v.trim().parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("malformed Content-Length: {v:?}"),
            )
        })?),
        None => None,
    };
    let conn_close = find("connection")
        .map(|v| v.eq_ignore_ascii_case("close"))
        .unwrap_or(false);
    // Bodyless per RFC 9112: HEAD, 1xx, 204, 304.
    let bodyless = is_head || status == 204 || status == 304 || (100..200).contains(&status);

    let (body, framed) = if bodyless {
        (Vec::new(), true)
    } else if chunked {
        (r.read_chunked().await?, true)
    } else if let Some(len) = content_length {
        // Reject an oversized advertised length up front with a clear error,
        // rather than reading until `fill` trips the buffer ceiling.
        if len > max_response_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Content-Length exceeds maximum response size",
            ));
        }
        (r.read_n(len).await?, true)
    } else {
        // No framing: body runs to EOF; the connection cannot be reused.
        (r.read_to_eof().await?, false)
    };

    Ok(Response {
        status,
        headers,
        body,
        keep_alive: framed && !conn_close,
    })
}

fn find_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\r\n")
}

fn eof(what: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::UnexpectedEof,
        format!("connection closed while reading {what}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use moirai_async::io::AsyncRead;

    /// In-memory reader that yields a fixed byte script, then EOF. Never returns
    /// `Pending`, so the response future resolves on a single poll.
    struct MockReader {
        data: Vec<u8>,
        pos: usize,
    }

    impl MockReader {
        fn new(data: Vec<u8>) -> Self {
            Self { data, pos: 0 }
        }
    }

    impl AsyncRead for MockReader {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            let remaining = self.data.len() - self.pos;
            let n = remaining.min(buf.len());
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Poll::Ready(Ok(n))
        }
    }

    fn read(data: Vec<u8>, max: usize) -> io::Result<Response> {
        moirai::block_on(read_response(&mut MockReader::new(data), false, max))
    }

    #[test]
    fn oversized_content_length_is_rejected_up_front() {
        // A peer advertises a body far larger than the cap; the body is never
        // actually sent. The upfront check must reject without reading it.
        let resp = b"HTTP/1.1 200 OK\r\nContent-Length: 999999999\r\n\r\n".to_vec();
        let err = read(resp, 4096).expect_err("oversized Content-Length must be rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn eof_delimited_body_over_limit_is_rejected() {
        // No Content-Length and no chunked framing => read-to-EOF. A body larger
        // than the cap must trip the buffer ceiling rather than allocate it all.
        let mut resp = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n".to_vec();
        resp.extend(std::iter::repeat_n(b'x', 64 * 1024));
        let err = read(resp, 8 * 1024).expect_err("EOF body over the cap must be rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn chunked_body_over_limit_is_rejected() {
        // Cumulative chunked decoding must be bounded: many chunks summing past
        // the cap are rejected, not accumulated without limit.
        let mut resp = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        for _ in 0..64 {
            resp.extend_from_slice(b"1000\r\n"); // 0x1000 = 4096-byte chunk
            resp.extend(std::iter::repeat_n(b'y', 0x1000));
            resp.extend_from_slice(b"\r\n");
        }
        resp.extend_from_slice(b"0\r\n\r\n");
        let err = read(resp, 16 * 1024).expect_err("chunked body over the cap must be rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn malformed_content_length_is_invalid_data_not_eof_framing() {
        // Adversarial: a present-but-garbage Content-Length must be a typed
        // framing error (RFC 9112 §6.3), never a silent fall-through to
        // read-to-EOF framing (response-desync hazard on reused connections).
        for bad in ["abc", "-5", "18446744073709551616", "12abc", ""] {
            let resp =
                format!("HTTP/1.1 200 OK\r\nContent-Length: {bad}\r\n\r\nhello").into_bytes();
            let err = read(resp, 64 * 1024)
                .expect_err("garbage Content-Length must be rejected, not EOF-framed");
            assert_eq!(err.kind(), io::ErrorKind::InvalidData, "value: {bad:?}");
        }
    }

    #[test]
    fn absent_content_length_still_uses_eof_framing() {
        // Control: with no Content-Length and no chunked framing, the body
        // legitimately runs to EOF and the connection is not reusable.
        let resp = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nstream-until-close".to_vec();
        let parsed = read(resp, 64 * 1024).expect("EOF-framed response must parse");
        assert_eq!(parsed.status, 200);
        assert_eq!(parsed.body, b"stream-until-close");
        assert!(!parsed.keep_alive, "EOF-framed body forbids reuse");
    }

    #[test]
    fn well_framed_response_under_limit_parses() {
        // Control: a legitimate small response within the budget parses cleanly,
        // proving the cap does not reject valid traffic.
        let resp = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello".to_vec();
        let parsed = read(resp, 64 * 1024).expect("valid response must parse");
        assert_eq!(parsed.status, 200);
        assert_eq!(parsed.body, b"hello");
        assert_eq!(parsed.header("content-length"), Some("5"));
    }
}
