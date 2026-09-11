//! Bounded HTTP/1.1 request-head parsing for server protocols.

use std::io;

use moirai_async::io::AsyncReadExt;

pub(crate) const MAX_HEADER_SLOTS: usize = 128;

/// A validated HTTP request line and header block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequestHead {
    method: String,
    target: String,
    version: String,
    headers: Vec<(String, String)>,
}

impl HttpRequestHead {
    /// Returns the request method.
    #[must_use]
    pub fn method(&self) -> &str {
        &self.method
    }

    /// Returns the origin-form request target.
    #[must_use]
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Returns the HTTP version token.
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Returns the first header value matching `name` case-insensitively.
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(key, _)| key == &name)
            .map(|(_, value)| value.as_str())
    }

    /// Returns all validated headers in receive order.
    #[must_use]
    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    /// Returns the browser page origin, when the request supplied one.
    #[must_use]
    pub fn origin(&self) -> Option<&str> {
        self.header("origin")
    }
}

/// Read one bounded HTTP request head and preserve bytes after its delimiter.
pub(crate) async fn read_request_head<S: AsyncReadExt + Unpin>(
    stream: &mut S,
    max_header_bytes: usize,
    max_header_count: usize,
) -> io::Result<(HttpRequestHead, Vec<u8>)> {
    if max_header_bytes == 0 || max_header_count == 0 || max_header_count > MAX_HEADER_SLOTS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "HTTP request limits are outside the supported bounds",
        ));
    }

    let mut bytes = Vec::with_capacity(max_header_bytes.min(4096));
    loop {
        if let Some(consumed) = header_end(&bytes) {
            let head = bytes
                .get(..consumed)
                .ok_or_else(|| io::Error::other("request delimiter exceeded buffered bytes"))?;
            let remainder = bytes
                .get(consumed..)
                .ok_or_else(|| io::Error::other("request remainder exceeded buffered bytes"))?
                .to_vec();
            return parse_request_head(head, remainder, max_header_count);
        }
        if bytes.len() >= max_header_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HTTP request headers exceed configured byte bound",
            ));
        }

        let available = max_header_bytes
            .checked_sub(bytes.len())
            .ok_or_else(|| io::Error::other("HTTP request buffer exceeded its configured bound"))?;
        let mut chunk = [0u8; 1024];
        let read_len = chunk.len().min(available);
        let target = chunk
            .get_mut(..read_len)
            .ok_or_else(|| io::Error::other("HTTP read slice exceeded its buffer"))?;
        let count = stream.read(target).await?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed before HTTP request headers completed",
            ));
        }
        if count > available {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HTTP request headers exceed configured byte bound",
            ));
        }
        let chunk = chunk
            .get(..count)
            .ok_or_else(|| io::Error::other("HTTP read count exceeded buffer"))?;
        bytes.extend_from_slice(chunk);
    }
}

fn parse_request_head(
    bytes: &[u8],
    remainder: Vec<u8>,
    max_header_count: usize,
) -> io::Result<(HttpRequestHead, Vec<u8>)> {
    let mut storage = [httparse::EMPTY_HEADER; MAX_HEADER_SLOTS];
    let mut request = httparse::Request::new(&mut storage);
    let parsed = request.parse(bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("malformed HTTP request head: {error}"),
        )
    })?;
    let consumed = match parsed {
        httparse::Status::Complete(consumed) => consumed,
        httparse::Status::Partial => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HTTP request delimiter was not parsed",
            ));
        }
    };
    if consumed != bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "HTTP request parser did not consume the complete head",
        ));
    }
    if request.headers.len() > max_header_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "HTTP request header count exceeds configured bound",
        ));
    }

    let method = request
        .method
        .ok_or_else(|| invalid_request("HTTP request has no method"))?;
    let target = request
        .path
        .ok_or_else(|| invalid_request("HTTP request has no target"))?;
    let version = request
        .version
        .ok_or_else(|| invalid_request("HTTP request has no version"))?;
    let version = format!("HTTP/1.{version}");
    if !target.starts_with('/') {
        return Err(invalid_request("HTTP request target must be origin-form"));
    }
    validate_token(method, "HTTP method")?;
    validate_target(target)?;

    let mut headers = Vec::with_capacity(request.headers.len());
    for header in request.headers {
        validate_header_name(header.name)?;
        let value = std::str::from_utf8(header.value)
            .map_err(|_| invalid_request("HTTP header value is not ASCII UTF-8"))?;
        if value
            .bytes()
            .any(|byte| byte < 0x20 && byte != b'\t' || byte == 0x7f)
        {
            return Err(invalid_request("HTTP header value contains a control byte"));
        }
        let name = header.name.to_ascii_lowercase();
        if headers.iter().any(|(existing, _)| existing == &name) {
            return Err(invalid_request("duplicate HTTP header is not accepted"));
        }
        headers.push((name, value.trim().to_owned()));
    }

    Ok((
        HttpRequestHead {
            method: method.to_owned(),
            target: target.to_owned(),
            version,
            headers,
        },
        remainder,
    ))
}

fn validate_token(value: &str, what: &str) -> io::Result<()> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte))
    {
        return Err(invalid_request(what));
    }
    Ok(())
}

fn validate_header_name(name: &str) -> io::Result<()> {
    validate_token(name, "HTTP header name is not a token")
}

fn validate_target(target: &str) -> io::Result<()> {
    if target
        .bytes()
        .any(|byte| byte < 0x20 || byte == 0x7f || byte == b' ')
    {
        return Err(invalid_request(
            "HTTP request target contains a control byte",
        ));
    }
    Ok(())
}

fn invalid_request(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn header_end(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .and_then(|position| position.checked_add(4))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    struct Input {
        bytes: VecDeque<u8>,
    }

    impl Input {
        fn new(bytes: &[u8]) -> Self {
            Self {
                bytes: bytes.iter().copied().collect(),
            }
        }
    }

    impl moirai_async::io::AsyncRead for Input {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            output: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            let count = output.len().min(self.bytes.len()).min(3);
            for slot in output.iter_mut().take(count) {
                let Some(byte) = self.bytes.pop_front() else {
                    return Poll::Ready(Ok(0));
                };
                *slot = byte;
            }
            Poll::Ready(Ok(count))
        }
    }

    #[test]
    fn parser_preserves_buffered_bytes_after_request_head() {
        let mut input = Input::new(b"GET /socket HTTP/1.1\r\nUpgrade: websocket\r\n\r\nframe");
        let (head, remainder) = moirai::block_on(read_request_head(&mut input, 512, 8))
            .expect("request head must parse");
        assert_eq!(head.method(), "GET");
        assert_eq!(head.target(), "/socket");
        assert_eq!(head.version(), "HTTP/1.1");
        assert_eq!(head.header("upgrade"), Some("websocket"));
        assert_eq!(remainder, b"f");
    }

    #[test]
    fn parser_rejects_duplicate_headers_and_oversized_heads() {
        for input in [
            b"GET / HTTP/1.1\r\nX-Test: one\r\nx-test: two\r\n\r\n".as_slice(),
            b"GET / HTTP/1.1\r\nX-Test: one".as_slice(),
        ] {
            let mut reader = Input::new(input);
            let error = moirai::block_on(read_request_head(&mut reader, 32, 8))
                .expect_err("invalid head must fail");
            assert!(matches!(
                error.kind(),
                io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof
            ));
        }
    }

    #[test]
    fn parser_accepts_near_limit_head_with_pipelined_bytes() {
        let prefix = b"GET / HTTP/1.1\r\nX-Pad: ";
        let suffix = b"\r\n\r\n";
        let target_head_length: usize = 1024;
        let padding = target_head_length
            .checked_sub(prefix.len() + suffix.len())
            .expect("test head target exceeds fixed prefix");
        let mut input = Vec::with_capacity(target_head_length + 4);
        input.extend_from_slice(prefix);
        input.extend(std::iter::repeat_n(b'x', padding));
        input.extend_from_slice(suffix);
        input.extend_from_slice(b"next");

        let mut reader = Input::new(&input);
        let (head, remainder) = moirai::block_on(read_request_head(&mut reader, 1024, 8))
            .expect("near-limit head must parse");
        assert_eq!(head.target(), "/");
        assert!(remainder.is_empty());
    }
}
