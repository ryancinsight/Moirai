//! Bounded RFC 6455 message framing over a Moirai async byte stream.

use std::io;

use moirai_async::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use moirai_async::timer::timeout;

use crate::upgrade::WebSocketConfig;

const FIN: u8 = 0x80;
const RSV_MASK: u8 = 0x70;
const OPCODE_MASK: u8 = 0x0f;
const MASK: u8 = 0x80;
const CLOSE: u8 = 0x8;
const PING: u8 = 0x9;
const PONG: u8 = 0xa;
const BINARY: u8 = 0x2;

/// A bounded, message-oriented RFC 6455 stream.
pub struct WebSocketStream<S> {
    stream: S,
    config: WebSocketConfig,
    prefix: Vec<u8>,
    prefix_position: usize,
    closed: bool,
}

impl<S> WebSocketStream<S> {
    pub(crate) fn new(stream: S, config: WebSocketConfig, prefix: Vec<u8>) -> Self {
        Self {
            stream,
            config,
            prefix,
            prefix_position: 0,
            closed: false,
        }
    }
}

impl<S: AsyncRead + AsyncWrite + Unpin> WebSocketStream<S> {
    /// Receive the next complete binary message.
    ///
    /// Ping frames are answered with pong frames and are not returned. Text,
    /// continuation, fragmented, reserved-bit and unmasked client frames are
    /// rejected. A close frame transitions the stream to a terminal state.
    ///
    /// # Errors
    /// Returns malformed-frame, timeout, connection, or write failures.
    pub async fn recv_message(&mut self) -> io::Result<Vec<u8>> {
        if self.closed {
            return Err(closed_error());
        }
        let timeout_duration = self.config.frame_timeout;
        timeout(timeout_duration, self.recv_message_inner())
            .await
            .map_err(|_| timed_out("WebSocket frame receive"))?
    }

    /// Send one unfragmented binary message to the peer.
    ///
    /// # Errors
    /// Returns a size, timeout, connection, or write failure.
    pub async fn send_binary(&mut self, payload: &[u8]) -> io::Result<()> {
        if self.closed {
            return Err(closed_error());
        }
        if payload.len() > self.config.max_message_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "WebSocket message exceeds configured byte bound",
            ));
        }
        let timeout_duration = self.config.frame_timeout;
        timeout(timeout_duration, self.send_frame(BINARY, payload))
            .await
            .map_err(|_| timed_out("WebSocket binary send"))?
    }

    /// Send a close frame and transition the stream to a terminal state.
    ///
    /// # Errors
    /// Returns invalid-input, timeout, connection, or write failures.
    pub async fn close(&mut self, code: u16, reason: &[u8]) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        if reason.len() > 123 || std::str::from_utf8(reason).is_err() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebSocket close reason must be valid UTF-8 and at most 123 bytes",
            ));
        }
        let capacity = reason.len().checked_add(2).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "close reason is too large")
        })?;
        let mut payload = Vec::with_capacity(capacity);
        payload.extend_from_slice(&code.to_be_bytes());
        payload.extend_from_slice(reason);
        validate_close_payload(&payload)?;
        let timeout_duration = self.config.frame_timeout;
        let result = timeout(timeout_duration, self.send_frame(CLOSE, &payload))
            .await
            .map_err(|_| timed_out("WebSocket close send"))?;
        if result.is_ok() {
            self.closed = true;
        }
        result
    }

    async fn recv_message_inner(&mut self) -> io::Result<Vec<u8>> {
        loop {
            let mut header = [0u8; 2];
            self.read_exact(&mut header).await?;
            let first = header[0];
            let second = header[1];
            if first & RSV_MASK != 0 {
                return Err(protocol_error("WebSocket reserved bits are not supported"));
            }
            let opcode = first & OPCODE_MASK;
            let final_frame = first & FIN != 0;
            let masked = second & MASK != 0;
            let length_code = second & 0x7f;
            let payload_length = self.read_length(length_code).await?;
            let control = opcode >= CLOSE;
            if control {
                if !final_frame || payload_length > 125 {
                    return Err(protocol_error("WebSocket control frame is invalid"));
                }
            } else if !final_frame {
                return Err(protocol_error(
                    "Fragmented WebSocket messages are not supported",
                ));
            }
            if (!control && opcode != BINARY) || (control && !matches!(opcode, CLOSE | PING | PONG))
            {
                return Err(protocol_error(
                    "WebSocket frame opcode is not a supported message or control frame",
                ));
            }
            if !masked {
                return Err(protocol_error("Client WebSocket frames must be masked"));
            }
            let mask = self.read_mask().await?;
            if opcode == BINARY {
                if payload_length > self.config.max_message_bytes {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "WebSocket message exceeds configured byte bound",
                    ));
                }
                return self.read_payload(payload_length, mask).await;
            }
            if opcode == PING {
                let payload = self.read_payload(payload_length, mask).await?;
                self.send_frame(PONG, &payload).await?;
                continue;
            }
            if opcode == PONG {
                let _ = self.read_payload(payload_length, mask).await?;
                continue;
            }
            if opcode == CLOSE {
                let payload = self.read_payload(payload_length, mask).await?;
                validate_close_payload(&payload)?;
                self.closed = true;
                self.send_frame(CLOSE, &payload).await?;
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "WebSocket peer closed the connection",
                ));
            }
            return Err(protocol_error("WebSocket frame opcode is not supported"));
        }
    }

    async fn read_length(&mut self, length_code: u8) -> io::Result<usize> {
        match length_code {
            0..=125 => Ok(usize::from(length_code)),
            126 => {
                let mut bytes = [0u8; 2];
                self.read_exact(&mut bytes).await?;
                Ok(usize::from(u16::from_be_bytes(bytes)))
            }
            127 => {
                let mut bytes = [0u8; 8];
                self.read_exact(&mut bytes).await?;
                let length = u64::from_be_bytes(bytes);
                if length & (1u64 << 63) != 0 {
                    return Err(protocol_error(
                        "WebSocket payload length has its high bit set",
                    ));
                }
                usize::try_from(length).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "WebSocket payload length cannot be represented",
                    )
                })
            }
            _ => Err(protocol_error("WebSocket payload length code is invalid")),
        }
    }

    async fn read_mask(&mut self) -> io::Result<[u8; 4]> {
        let mut mask = [0u8; 4];
        self.read_exact(&mut mask).await?;
        Ok(mask)
    }

    async fn read_payload(&mut self, length: usize, mask: [u8; 4]) -> io::Result<Vec<u8>> {
        let mut payload = vec![0u8; length];
        self.read_exact(&mut payload).await?;
        for (byte, mask_byte) in payload.iter_mut().zip(mask.iter().cycle()) {
            *byte ^= *mask_byte;
        }
        Ok(payload)
    }

    async fn read_exact(&mut self, output: &mut [u8]) -> io::Result<()> {
        let prefix_available = self.prefix.len().saturating_sub(self.prefix_position);
        let from_prefix = output.len().min(prefix_available);
        if from_prefix != 0 {
            let start = self.prefix_position;
            let end = start.checked_add(from_prefix).ok_or_else(|| {
                io::Error::other("WebSocket prefix position arithmetic overflowed")
            })?;
            let source = self
                .prefix
                .get(start..end)
                .ok_or_else(|| io::Error::other("WebSocket prefix bounds are inconsistent"))?;
            let destination = output
                .get_mut(..from_prefix)
                .ok_or_else(|| io::Error::other("WebSocket output bounds are inconsistent"))?;
            destination.copy_from_slice(source);
            self.prefix_position = end;
        }
        let mut filled = from_prefix;
        while filled < output.len() {
            let destination = output
                .get_mut(filled..)
                .ok_or_else(|| io::Error::other("WebSocket output bounds are inconsistent"))?;
            let count = self.stream.read(destination).await?;
            if count == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "connection closed inside a WebSocket frame",
                ));
            }
            filled = filled
                .checked_add(count)
                .ok_or_else(|| io::Error::other("WebSocket read length overflow"))?;
        }
        Ok(())
    }

    async fn send_frame(&mut self, opcode: u8, payload: &[u8]) -> io::Result<()> {
        let capacity = payload
            .len()
            .checked_add(10)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "message is too large"))?;
        let mut frame = Vec::with_capacity(capacity);
        frame.push(FIN | opcode);
        match payload.len() {
            length @ 0..=125 => frame.push(u8::try_from(length).expect("invariant: length <= 125")),
            length @ 126..=65_535 => {
                frame.push(126);
                let length =
                    u16::try_from(length).expect("invariant: extended WebSocket length fits u16");
                frame.extend_from_slice(&length.to_be_bytes());
            }
            length => {
                frame.push(127);
                let length = u64::try_from(length).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "message is too large")
                })?;
                frame.extend_from_slice(&length.to_be_bytes());
            }
        }
        frame.extend_from_slice(payload);
        self.stream.write_all(&frame).await?;
        self.stream.flush().await
    }

    #[cfg(test)]
    pub(crate) fn initial_bytes(&self) -> &[u8] {
        self.prefix
            .get(self.prefix_position..)
            .expect("invariant: WebSocket prefix position stays within the prefix")
    }

    #[cfg(test)]
    pub(crate) fn output_bytes(&self) -> &[u8]
    where
        S: OutputBytes,
    {
        self.stream.output_bytes()
    }
}

fn validate_close_payload(payload: &[u8]) -> io::Result<()> {
    if payload.len() == 1 {
        return Err(protocol_error("WebSocket close payload has one byte"));
    }
    if payload.len() >= 2 {
        let code = payload
            .get(..2)
            .and_then(|bytes| bytes.try_into().ok())
            .map(u16::from_be_bytes)
            .ok_or_else(|| protocol_error("WebSocket close code is truncated"))?;
        let valid_range = (1000..=2999).contains(&code) || (3000..=4999).contains(&code);
        if !valid_range || matches!(code, 1004 | 1005 | 1006 | 1015) {
            return Err(protocol_error("WebSocket close code is reserved"));
        }
        std::str::from_utf8(payload.get(2..).unwrap_or_default())
            .map_err(|_| protocol_error("WebSocket close reason is not UTF-8"))?;
    }
    Ok(())
}

fn protocol_error(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn timed_out(operation: &str) -> io::Error {
    io::Error::new(io::ErrorKind::TimedOut, operation)
}

fn closed_error() -> io::Error {
    io::Error::new(io::ErrorKind::BrokenPipe, "WebSocket is closed")
}

#[cfg(test)]
pub(crate) trait OutputBytes {
    fn output_bytes(&self) -> &[u8];
}

#[cfg(test)]
mod tests {
    use super::*;
    use moirai_async::io::{AsyncRead, AsyncWrite};
    use std::collections::VecDeque;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use std::time::Duration;

    struct MemoryStream {
        input: VecDeque<u8>,
        output: Vec<u8>,
    }

    impl MemoryStream {
        fn new(input: Vec<u8>) -> Self {
            Self {
                input: input.into_iter().collect(),
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
            let count = output.len().min(self.input.len()).min(3);
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
            let count = input.len().min(2);
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

    impl OutputBytes for MemoryStream {
        fn output_bytes(&self) -> &[u8] {
            &self.output
        }
    }

    fn masked_frame(opcode: u8, payload: &[u8], mask: [u8; 4]) -> Vec<u8> {
        let mut frame = vec![
            FIN | opcode,
            MASK | u8::try_from(payload.len()).expect("test payload"),
        ];
        frame.extend_from_slice(&mask);
        frame.extend(
            payload
                .iter()
                .zip(mask.iter().cycle())
                .map(|(byte, mask_byte)| byte ^ mask_byte),
        );
        frame
    }

    fn stream(input: Vec<u8>) -> WebSocketStream<MemoryStream> {
        WebSocketStream::new(
            MemoryStream::new(input),
            WebSocketConfig::default(),
            Vec::new(),
        )
    }

    #[test]
    fn masked_binary_message_round_trips_with_partial_io() {
        let mut stream = stream(masked_frame(BINARY, b"hello", [1, 2, 3, 4]));
        let payload = moirai::block_on(stream.recv_message()).expect("binary message");
        assert_eq!(payload, b"hello");
    }

    #[test]
    fn unmasked_and_fragmented_frames_are_rejected() {
        for frame in [
            vec![FIN | BINARY, 1, b'x'],
            vec![BINARY, MASK | 1, 0, 0, 0, 0, b'x'],
        ] {
            let mut stream = stream(frame);
            let error = moirai::block_on(stream.recv_message()).expect_err("invalid frame");
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        }
    }

    #[test]
    fn ping_is_answered_and_next_binary_message_is_returned() {
        let mut input = masked_frame(PING, b"p", [4, 3, 2, 1]);
        input.extend(masked_frame(BINARY, b"ok", [8, 7, 6, 5]));
        let mut stream = stream(input);
        let payload = moirai::block_on(stream.recv_message()).expect("message after ping");
        assert_eq!(payload, b"ok");
        let output = stream.output_bytes();
        assert_eq!(output, &[FIN | PONG, 1, b'p']);
    }

    #[test]
    fn oversized_binary_message_is_rejected_before_allocation() {
        let mut frame = vec![FIN | BINARY, MASK | 0x7e, 0, 5];
        frame.extend_from_slice(&[0, 0, 0, 0]);
        let mut stream = WebSocketStream::new(
            MemoryStream::new(frame),
            WebSocketConfig::new(1024, 8, 4, Duration::from_secs(1), Duration::from_secs(1)),
            Vec::new(),
        );
        let error = moirai::block_on(stream.recv_message()).expect_err("oversize must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn unsupported_data_opcode_is_rejected_before_payload_allocation() {
        let frame = vec![FIN | 0x1, MASK | 0x7f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut stream = stream(frame);
        let error = moirai::block_on(stream.recv_message()).expect_err("text frame must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn outgoing_close_rejects_reserved_code() {
        let mut stream = stream(Vec::new());
        let error = moirai::block_on(stream.close(1005, &[])).expect_err("reserved code");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(stream.output_bytes().is_empty());
    }

    #[test]
    fn outgoing_close_accepts_application_code() {
        let mut stream = stream(Vec::new());
        moirai::block_on(stream.close(4000, b"application")).expect("application code");
        assert_eq!(stream.output_bytes().get(..2), Some(&[FIN | CLOSE, 13][..]));
    }

    #[test]
    fn close_frame_is_terminal_and_echoed() {
        let input = masked_frame(CLOSE, &1000u16.to_be_bytes(), [1, 1, 1, 1]);
        let mut stream = stream(input);
        let error = moirai::block_on(stream.recv_message()).expect_err("close is terminal");
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        assert_eq!(stream.output_bytes(), &[FIN | CLOSE, 2, 3, 232]);
        let error = moirai::block_on(stream.recv_message()).expect_err("closed stream");
        assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
    }
}
