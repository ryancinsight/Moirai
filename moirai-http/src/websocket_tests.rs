use super::*;
use moirai_async::io::{AsyncRead, AsyncWrite};
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

struct MemoryStream {
    input: VecDeque<u8>,
    output: Vec<u8>,
    read_limit: usize,
    write_limit: usize,
    read_error: Option<io::ErrorKind>,
    write_error: Option<io::ErrorKind>,
}

impl MemoryStream {
    fn new(input: Vec<u8>) -> Self {
        Self {
            input: input.into_iter().collect(),
            output: Vec::new(),
            read_limit: usize::MAX,
            write_limit: usize::MAX,
            read_error: None,
            write_error: None,
        }
    }
}

impl AsyncRead for MemoryStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        output: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        if self.read_limit == 0 {
            if let Some(kind) = self.read_error {
                return Poll::Ready(Err(io::Error::from(kind)));
            }
            return Poll::Pending;
        }
        let count = output
            .len()
            .min(self.input.len())
            .min(3)
            .min(self.read_limit);
        for slot in output.iter_mut().take(count) {
            let Some(byte) = self.input.pop_front() else {
                return Poll::Ready(Ok(0));
            };
            *slot = byte;
        }
        self.read_limit = self.read_limit.saturating_sub(count);
        Poll::Ready(Ok(count))
    }
}

impl AsyncWrite for MemoryStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        input: &[u8],
    ) -> Poll<io::Result<usize>> {
        if self.write_limit == 0 {
            if let Some(kind) = self.write_error {
                return Poll::Ready(Err(io::Error::from(kind)));
            }
            return Poll::Pending;
        }
        let count = input.len().min(2);
        let count = count.min(self.write_limit);
        let input = input
            .get(..count)
            .expect("invariant: test writer count is within input length");
        self.output.extend_from_slice(input);
        self.write_limit = self.write_limit.saturating_sub(count);
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
fn non_minimal_payload_lengths_are_rejected_before_mask_read() {
    for frame in [
        vec![FIN | BINARY, MASK | 0x7e, 0, 5],
        vec![FIN | BINARY, MASK | 0x7f, 0, 0, 0, 0, 0, 0, 0, 5],
    ] {
        let mut stream = stream(frame);
        let error = moirai::block_on(stream.recv_message()).expect_err("non-minimal length");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }
}

#[test]
fn receive_timeout_terminalizes_a_partially_consumed_frame() {
    let mut input = MemoryStream::new(masked_frame(BINARY, b"pending", [1, 2, 3, 4]));
    input.read_limit = 1;
    let mut stream = WebSocketStream::new(
        input,
        WebSocketConfig::new(
            1024,
            8,
            1024,
            Duration::from_secs(1),
            Duration::from_millis(10),
        ),
        Vec::new(),
    );
    let error = moirai::block_on(stream.recv_message()).expect_err("partial frame timeout");
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    let error = moirai::block_on(stream.recv_message()).expect_err("timed out stream is closed");
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
}

#[test]
fn receive_error_terminalizes_a_partially_consumed_frame() {
    let mut input = MemoryStream::new(masked_frame(BINARY, b"broken", [1, 2, 3, 4]));
    input.read_limit = 1;
    input.read_error = Some(io::ErrorKind::ConnectionReset);
    let mut stream = WebSocketStream::new(
        input,
        WebSocketConfig::new(
            1024,
            8,
            1024,
            Duration::from_secs(1),
            Duration::from_secs(1),
        ),
        Vec::new(),
    );
    let error = moirai::block_on(stream.recv_message()).expect_err("partial frame error");
    assert_eq!(error.kind(), io::ErrorKind::ConnectionReset);
    let error = moirai::block_on(stream.recv_message()).expect_err("errored stream is closed");
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
}

#[test]
fn send_timeout_terminalizes_a_partially_written_frame() {
    let mut input = MemoryStream::new(Vec::new());
    input.write_limit = 1;
    let mut stream = WebSocketStream::new(
        input,
        WebSocketConfig::new(
            1024,
            8,
            1024,
            Duration::from_secs(1),
            Duration::from_millis(10),
        ),
        Vec::new(),
    );
    let error = moirai::block_on(stream.send_binary(b"pending")).expect_err("partial send timeout");
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    let error =
        moirai::block_on(stream.send_binary(b"retry")).expect_err("timed out stream is closed");
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
}

#[test]
fn send_error_terminalizes_a_partially_written_frame() {
    let mut input = MemoryStream::new(Vec::new());
    input.write_limit = 1;
    input.write_error = Some(io::ErrorKind::BrokenPipe);
    let mut stream = WebSocketStream::new(
        input,
        WebSocketConfig::new(
            1024,
            8,
            1024,
            Duration::from_secs(1),
            Duration::from_secs(1),
        ),
        Vec::new(),
    );
    let error = moirai::block_on(stream.send_binary(b"broken")).expect_err("partial send error");
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
    let error =
        moirai::block_on(stream.send_binary(b"retry")).expect_err("errored stream is closed");
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
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
