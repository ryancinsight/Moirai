//! HTTP/1.1 request serialization and response parsing (status line, headers,
//! and body framing: Content-Length, chunked transfer-encoding, or EOF-delimited).

use std::io;

use moirai_async::io::{AsyncReadExt, AsyncWrite, AsyncWriteExt};

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
}

impl<'a, S: AsyncReadExt + Unpin> Buffered<'a, S> {
    fn new(stream: &'a mut S) -> Self {
        Self {
            stream,
            buf: Vec::with_capacity(8192),
            pos: 0,
        }
    }

    fn available(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Read more bytes from the stream into the buffer. Returns bytes read (0 = EOF).
    async fn fill(&mut self) -> io::Result<usize> {
        let mut tmp = [0u8; 8192];
        let n = self.stream.read(&mut tmp).await?;
        self.buf.extend_from_slice(&tmp[..n]);
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
) -> io::Result<Response> {
    let mut r = Buffered::new(stream);

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
    let content_length: Option<usize> = find("content-length").and_then(|v| v.trim().parse().ok());
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
