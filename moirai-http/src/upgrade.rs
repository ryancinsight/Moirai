//! HTTP/1.1 WebSocket upgrade validation and response serialization.

use std::io;
use std::time::Duration;

use moirai_async::io::AsyncWriteExt;
use moirai_async::timer::timeout;
use moirai_crypto::{base64_decode, base64_encode, sha1};

use crate::request::{read_request_head, HttpRequestHead};
use crate::websocket::WebSocketStream;

const WEBSOCKET_GUID: &[u8] = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const DEFAULT_MAX_HEADER_BYTES: usize = 16 * 1024;
const DEFAULT_MAX_HEADER_COUNT: usize = 32;
const DEFAULT_MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_FRAME_TIMEOUT: Duration = Duration::from_secs(30);

/// Resource and deadline policy for one WebSocket connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebSocketConfig {
    /// Maximum bytes read for the HTTP request head.
    pub max_header_bytes: usize,
    /// Maximum number of HTTP request headers.
    pub max_header_count: usize,
    /// Maximum payload bytes in one binary message.
    pub max_message_bytes: usize,
    /// Deadline for reading and answering the HTTP upgrade.
    pub handshake_timeout: Duration,
    /// Deadline for each WebSocket frame operation.
    pub frame_timeout: Duration,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            max_header_bytes: DEFAULT_MAX_HEADER_BYTES,
            max_header_count: DEFAULT_MAX_HEADER_COUNT,
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
            handshake_timeout: DEFAULT_HANDSHAKE_TIMEOUT,
            frame_timeout: DEFAULT_FRAME_TIMEOUT,
        }
    }
}

impl WebSocketConfig {
    /// Construct a WebSocket policy with explicit bounds and deadlines.
    #[must_use]
    pub const fn new(
        max_header_bytes: usize,
        max_header_count: usize,
        max_message_bytes: usize,
        handshake_timeout: Duration,
        frame_timeout: Duration,
    ) -> Self {
        Self {
            max_header_bytes,
            max_header_count,
            max_message_bytes,
            handshake_timeout,
            frame_timeout,
        }
    }

    pub(crate) fn validate(self) -> io::Result<()> {
        if self.max_header_bytes == 0
            || self.max_header_count == 0
            || self.max_header_count > crate::request::MAX_HEADER_SLOTS
            || self.max_message_bytes == 0
            || self.handshake_timeout.is_zero()
            || self.frame_timeout.is_zero()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebSocket limits and deadlines must be non-zero",
            ));
        }
        Ok(())
    }
}

/// Validated request metadata returned by a successful upgrade.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebSocketUpgrade {
    request: HttpRequestHead,
    origin: Option<String>,
}

impl WebSocketUpgrade {
    /// Returns the validated HTTP request head.
    #[must_use]
    pub const fn request(&self) -> &HttpRequestHead {
        &self.request
    }

    /// Returns the browser origin, when the peer sent an `Origin` header.
    #[must_use]
    pub fn origin(&self) -> Option<&str> {
        self.origin.as_deref()
    }
}

/// Perform a bounded HTTP/1.1 WebSocket upgrade on an async byte stream.
///
/// The returned stream preserves any bytes read after the HTTP delimiter, so a
/// peer may pipeline its first WebSocket frame in the same TCP packet.
///
/// # Errors
/// Returns invalid-input/data, timeout, or transport errors when the request
/// does not satisfy the WebSocket upgrade contract or I/O fails.
pub async fn accept_websocket<S>(
    mut stream: S,
    config: WebSocketConfig,
) -> io::Result<(WebSocketStream<S>, WebSocketUpgrade)>
where
    S: moirai_async::io::AsyncRead + moirai_async::io::AsyncWrite + Unpin,
{
    config.validate()?;
    let (request, remainder) = timeout(
        config.handshake_timeout,
        read_request_head(
            &mut stream,
            config.max_header_bytes,
            config.max_header_count,
        ),
    )
    .await
    .map_err(|_| timed_out("WebSocket handshake read"))??;
    let key = validate_upgrade(&request)?;
    let accept = websocket_accept_key(key);
    let response = format!(
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
    );
    timeout(
        config.handshake_timeout,
        stream.write_all(response.as_bytes()),
    )
    .await
    .map_err(|_| timed_out("WebSocket handshake write"))??;
    timeout(config.handshake_timeout, stream.flush())
        .await
        .map_err(|_| timed_out("WebSocket handshake flush"))??;

    let origin = request.origin().map(str::to_owned);
    let upgrade = WebSocketUpgrade { request, origin };
    Ok((WebSocketStream::new(stream, config, remainder), upgrade))
}

fn validate_upgrade(request: &HttpRequestHead) -> io::Result<&str> {
    if request.method() != "GET" {
        return Err(invalid_upgrade("WebSocket upgrade requires GET"));
    }
    if request.version() != "HTTP/1.1" {
        return Err(invalid_upgrade("WebSocket upgrade requires HTTP/1.1"));
    }
    if request
        .header("upgrade")
        .is_none_or(|value| !value.eq_ignore_ascii_case("websocket"))
    {
        return Err(invalid_upgrade("Upgrade header must be websocket"));
    }
    let connection = request
        .header("connection")
        .ok_or_else(|| invalid_upgrade("Connection header is required"))?;
    if !connection
        .split(',')
        .any(|token| token.trim().eq_ignore_ascii_case("upgrade"))
    {
        return Err(invalid_upgrade("Connection header must include Upgrade"));
    }
    if request.header("sec-websocket-version") != Some("13") {
        return Err(invalid_upgrade("Sec-WebSocket-Version must be 13"));
    }
    if request.header("content-length").is_some() || request.header("transfer-encoding").is_some() {
        return Err(invalid_upgrade(
            "WebSocket upgrade must not carry an HTTP body",
        ));
    }
    let key = request
        .header("sec-websocket-key")
        .ok_or_else(|| invalid_upgrade("Sec-WebSocket-Key is required"))?;
    let decoded = base64_decode(key.as_bytes())
        .filter(|bytes| bytes.len() == 16)
        .ok_or_else(|| invalid_upgrade("Sec-WebSocket-Key must encode 16 bytes"))?;
    if decoded.len() != 16 {
        return Err(invalid_upgrade("Sec-WebSocket-Key length is invalid"));
    }
    Ok(key)
}

fn websocket_accept_key(key: &str) -> String {
    let mut input = Vec::with_capacity(key.len());
    input.extend_from_slice(key.as_bytes());
    input.extend_from_slice(WEBSOCKET_GUID);
    base64_encode(&sha1(&input))
}

fn invalid_upgrade(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn timed_out(operation: &str) -> io::Error {
    io::Error::new(io::ErrorKind::TimedOut, operation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::websocket::WebSocketStream;
    use moirai_async::io::{AsyncRead, AsyncWrite};
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::task::Waker;
    use std::task::{Context, Poll};
    use std::time::Duration;

    struct MemoryStream {
        input: VecDeque<u8>,
        output: Vec<u8>,
    }

    impl MemoryStream {
        fn new(input: &[u8]) -> Self {
            Self {
                input: input.iter().copied().collect(),
                output: Vec::new(),
            }
        }
    }

    impl AsyncRead for MemoryStream {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            output: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            let count = output.len().min(self.input.len());
            for slot in output.iter_mut().take(count) {
                let Some(byte) = self.input.pop_front() else {
                    return Poll::Ready(Ok(0));
                };
                *slot = byte;
            }
            Poll::Ready(Ok(count))
        }
    }

    impl AsyncWrite for MemoryStream {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            input: &[u8],
        ) -> Poll<io::Result<usize>> {
            let count = input.len().min(5);
            let input = input
                .get(..count)
                .expect("invariant: test writer count is within input length");
            self.output.extend_from_slice(input);
            Poll::Ready(Ok(count))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    impl crate::websocket::OutputBytes for MemoryStream {
        fn output_bytes(&self) -> &[u8] {
            &self.output
        }
    }

    struct PendingStream {
        dropped: Option<Arc<AtomicBool>>,
    }

    impl AsyncRead for PendingStream {
        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            _output: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            let _ = self;
            Poll::Pending
        }
    }

    impl AsyncWrite for PendingStream {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            _input: &[u8],
        ) -> Poll<io::Result<usize>> {
            let _ = self;
            Poll::Pending
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            let _ = self;
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            let _ = self;
            Poll::Ready(Ok(()))
        }
    }

    impl Drop for PendingStream {
        fn drop(&mut self) {
            if let Some(dropped) = self.dropped.take() {
                dropped.store(true, Ordering::Relaxed);
            }
        }
    }

    fn request(extra: &str) -> Vec<u8> {
        let mut request = String::from(
            "GET /metis HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: keep-alive, Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nOrigin: http://127.0.0.1:8765\r\n",
        );
        if !extra.is_empty() {
            request.push_str(extra);
            request.push_str("\r\n");
        }
        request.push_str("\r\n");
        request.into_bytes()
    }

    #[test]
    fn valid_upgrade_emits_rfc_response_and_preserves_origin() {
        let mut bytes = request("");
        bytes.extend_from_slice(b"first-frame");
        let stream = MemoryStream::new(&bytes);
        let (stream, upgrade): (WebSocketStream<MemoryStream>, WebSocketUpgrade) =
            moirai::block_on(accept_websocket(stream, WebSocketConfig::default()))
                .expect("upgrade must succeed");
        assert_eq!(upgrade.origin(), Some("http://127.0.0.1:8765"));
        assert_eq!(stream.initial_bytes(), b"first-frame");
        assert_eq!(
            stream.output_bytes(),
            b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n"
        );
    }

    #[test]
    fn invalid_upgrade_headers_are_rejected() {
        for replacement in [
            "Upgrade: http",
            "Connection: keep-alive",
            "Sec-WebSocket-Version: 12",
            "Sec-WebSocket-Key: bad",
            "Content-Length: 0",
        ] {
            let input = request(replacement);
            let stream = MemoryStream::new(&input);
            let error = match moirai::block_on(accept_websocket(stream, WebSocketConfig::default()))
            {
                Ok(_) => panic!("invalid upgrade must fail"),
                Err(error) => error,
            };
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        }
    }

    #[test]
    fn handshake_deadline_terminates_a_pending_peer() {
        let config = WebSocketConfig::new(
            1024,
            8,
            1024,
            Duration::from_millis(10),
            Duration::from_millis(10),
        );
        let error =
            match moirai::block_on(accept_websocket(PendingStream { dropped: None }, config)) {
                Ok(_) => panic!("pending handshake must time out"),
                Err(error) => error,
            };
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    }

    #[test]
    fn dropping_handshake_future_drops_the_owned_stream() {
        let dropped = Arc::new(AtomicBool::new(false));
        let config = WebSocketConfig::new(
            1024,
            8,
            1024,
            Duration::from_secs(1),
            Duration::from_secs(1),
        );
        let mut future = Box::pin(accept_websocket(
            PendingStream {
                dropped: Some(Arc::clone(&dropped)),
            },
            config,
        ));
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        assert!(future.as_mut().poll(&mut context).is_pending());
        drop(future);
        assert!(dropped.load(Ordering::Relaxed));
    }
}
