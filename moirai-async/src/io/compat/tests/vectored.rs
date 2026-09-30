//! Vectored-write mapping: `poll_write_vectored` and `is_write_vectored`.

use crate::io::{AsyncWrite, MoiraiCompat, TokioCompat};
use std::io::{self, IoSlice};
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use tokio_dep as tokio;
use tokio_dep::io::AsyncWriteExt as _;

/// Bytes accepted by a recording writer, and the entry points that took them.
#[derive(Default)]
struct Sink {
    bytes: Vec<u8>,
    scalar_calls: usize,
    vectored_calls: usize,
    /// Most bytes accepted per call; `0` accepts everything.
    limit: usize,
}

impl Sink {
    fn accept(&mut self, data: &[u8]) -> usize {
        let count = if self.limit == 0 {
            data.len()
        } else {
            data.len().min(self.limit)
        };
        self.bytes.extend_from_slice(&data[..count]);
        count
    }
}

/// Implements only the required methods, so vectored writes use the default.
#[derive(Default)]
struct Plain(Sink);

impl AsyncWrite for Plain {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.0.scalar_calls += 1;
        Poll::Ready(Ok(self.0.accept(buf)))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// Gathers across slices in one call, up to the sink limit.
#[derive(Default)]
struct Scatter(Sink);

impl AsyncWrite for Scatter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.0.scalar_calls += 1;
        Poll::Ready(Ok(self.0.accept(buf)))
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let sink = &mut self.0;
        sink.vectored_calls += 1;
        let mut written = 0;
        for buf in bufs {
            let room = if sink.limit == 0 {
                buf.len()
            } else {
                sink.limit - written
            };
            written += sink.accept(&buf[..buf.len().min(room)]);
            if sink.limit != 0 && written == sink.limit {
                break;
            }
        }
        Poll::Ready(Ok(written))
    }

    fn is_write_vectored(&self) -> bool {
        true
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

fn slices<'a>(parts: &[&'a [u8]]) -> Vec<IoSlice<'a>> {
    parts.iter().map(|part| IoSlice::new(part)).collect()
}

fn ready<T>(poll: Poll<io::Result<T>>) -> T {
    match poll {
        Poll::Ready(result) => result.expect("in-memory write must succeed"),
        Poll::Pending => panic!("in-memory writer must not be pending"),
    }
}

const PARTS: [&[u8]; 4] = [b"ab", b"", b"cde", b"fg"];

#[test]
fn tokio_vectored_write_reaches_a_vectored_moirai_writer() {
    let mut writer = TokioCompat::new(Scatter(Sink {
        limit: 5,
        ..Sink::default()
    }));
    assert!(tokio::io::AsyncWrite::is_write_vectored(&writer));

    let parts = slices(&PARTS);
    let (first, rest) = futures::executor::block_on(async {
        let first = writer
            .write_vectored(&parts)
            .await
            .expect("Tokio vectored write must succeed");
        let rest = writer
            .write_vectored(&parts[3..])
            .await
            .expect("Tokio vectored write must succeed");
        (first, rest)
    });

    let sink = writer.into_inner().0;
    assert_eq!((first, rest), (5, 2));
    assert_eq!(sink.bytes, b"abcdefg");
    assert_eq!((sink.vectored_calls, sink.scalar_calls), (2, 0));
}

#[test]
fn writers_without_vectored_support_take_the_first_non_empty_slice() {
    let parts = slices(&[b"", b"xy", b"z"]);
    let mut native = Plain::default();
    let direct = ready(AsyncWrite::poll_write_vectored(
        Pin::new(&mut native),
        &mut Context::from_waker(Waker::noop()),
        &parts,
    ));
    assert!(!AsyncWrite::is_write_vectored(&native));

    let mut wrapped = TokioCompat::new(Plain::default());
    assert!(!tokio::io::AsyncWrite::is_write_vectored(&wrapped));
    let through_tokio = futures::executor::block_on(wrapped.write_vectored(&parts))
        .expect("Tokio vectored write must succeed");

    assert_eq!((direct, through_tokio), (2, 2));
    assert_eq!(native.0.bytes, b"xy");
    assert_eq!(wrapped.into_inner().0.bytes, b"xy");

    let mut empty = Plain::default();
    let none = ready(AsyncWrite::poll_write_vectored(
        Pin::new(&mut empty),
        &mut Context::from_waker(Waker::noop()),
        &slices(&[b"", b""]),
    ));
    assert_eq!(none, 0);
    assert_eq!((empty.0.scalar_calls, empty.0.bytes.len()), (1, 0));
}

#[test]
fn moirai_vectored_write_matches_the_native_gather() {
    let parts = slices(&PARTS);

    let mut native = Scatter::default();
    let gathered = ready(AsyncWrite::poll_write_vectored(
        Pin::new(&mut native),
        &mut Context::from_waker(Waker::noop()),
        &parts,
    ));

    let mut wrapped = MoiraiCompat::new(Vec::<u8>::new());
    assert!(AsyncWrite::is_write_vectored(&wrapped));
    let through_moirai = ready(AsyncWrite::poll_write_vectored(
        Pin::new(&mut wrapped),
        &mut Context::from_waker(Waker::noop()),
        &parts,
    ));

    assert_eq!(through_moirai, gathered);
    assert_eq!(wrapped.into_inner(), native.0.bytes);
    assert_eq!(native.0.bytes, b"abcdefg");
}

/// A Tokio writer with only the required methods, so vectored writes use
/// Tokio's default.
#[derive(Default)]
struct TokioScalar(Vec<u8>);

impl tokio::io::AsyncWrite for TokioScalar {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.0.extend_from_slice(buf);
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

#[test]
fn moirai_vectored_capability_follows_the_tokio_writer() {
    let (near, _far) = tokio::io::duplex(8);
    let expected = tokio::io::AsyncWrite::is_write_vectored(&near);
    let wrapped = MoiraiCompat::new(near);
    assert_eq!(AsyncWrite::is_write_vectored(&wrapped), expected);
    assert!(expected);

    let mut scalar = MoiraiCompat::new(TokioScalar::default());
    assert!(!AsyncWrite::is_write_vectored(&scalar));
    let written = ready(AsyncWrite::poll_write_vectored(
        Pin::new(&mut scalar),
        &mut Context::from_waker(Waker::noop()),
        &slices(&[b"", b"xy", b"z"]),
    ));
    assert_eq!(written, 2);
    assert_eq!(scalar.into_inner().0, b"xy");
}
