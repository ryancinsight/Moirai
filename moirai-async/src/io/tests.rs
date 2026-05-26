use super::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
#[cfg(feature = "tokio-compat")]
use super::{MoiraiCompat, TokioCompat};
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

struct ChunkedReader<'a> {
    source: &'a [u8],
    offset: usize,
    max_chunk: usize,
}

impl AsyncRead for ChunkedReader<'_> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        if self.offset == self.source.len() {
            return Poll::Ready(Ok(0));
        }

        let remaining = self.source.len() - self.offset;
        let count = remaining.min(buf.len()).min(self.max_chunk);
        buf[..count].copy_from_slice(&self.source[self.offset..self.offset + count]);
        self.offset += count;
        Poll::Ready(Ok(count))
    }
}

struct PendingAfterFirstRead<'a> {
    source: &'a [u8],
    offset: usize,
    returned_pending: bool,
}

impl AsyncRead for PendingAfterFirstRead<'_> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        if self.offset == 0 {
            let count = 2.min(buf.len()).min(self.source.len());
            buf[..count].copy_from_slice(&self.source[..count]);
            self.offset = count;
            return Poll::Ready(Ok(count));
        }

        if !self.returned_pending {
            self.returned_pending = true;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }

        let remaining = self.source.len() - self.offset;
        let count = remaining.min(buf.len());
        buf[..count].copy_from_slice(&self.source[self.offset..self.offset + count]);
        self.offset += count;
        Poll::Ready(Ok(count))
    }
}

#[derive(Default)]
struct PartialWriter {
    bytes: Vec<u8>,
    max_chunk: usize,
    flushes: usize,
    shutdowns: usize,
}

impl AsyncWrite for PartialWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let count = buf.len().min(self.max_chunk);
        self.bytes.extend_from_slice(&buf[..count]);
        Poll::Ready(Ok(count))
    }

    fn poll_flush(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.flushes += 1;
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.shutdowns += 1;
        Poll::Ready(Ok(()))
    }
}

#[test]
fn read_exact_fills_buffer_across_partial_reads() {
    futures::executor::block_on(async {
        let mut reader = ChunkedReader {
            source: b"abcdef",
            offset: 0,
            max_chunk: 2,
        };
        let mut output = [0_u8; 6];

        reader
            .read_exact(&mut output)
            .await
            .expect("read_exact must fill exact buffer");

        assert_eq!(&output, b"abcdef");
        assert_eq!(reader.offset, 6);
    });
}

#[test]
fn read_exact_reports_unexpected_eof_with_prefix_preserved() {
    futures::executor::block_on(async {
        let mut reader = ChunkedReader {
            source: b"abc",
            offset: 0,
            max_chunk: 8,
        };
        let mut output = [0_u8; 5];

        let error = reader
            .read_exact(&mut output)
            .await
            .expect_err("short source must report EOF");

        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        assert_eq!(&output[..3], b"abc");
        assert_eq!(reader.offset, 3);
    });
}

#[test]
fn read_exact_cancellation_preserves_borrowed_buffer_progress() {
    let mut reader = PendingAfterFirstRead {
        source: b"abcd",
        offset: 0,
        returned_pending: false,
    };
    let mut output = [0_u8; 4];

    {
        let mut future = reader.read_exact(&mut output);
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(
            Pin::new(&mut future).poll(&mut context),
            Poll::Pending
        ));
    }

    assert_eq!(&output[..2], b"ab");
    assert_eq!(reader.offset, 2);
}

#[test]
fn write_all_flush_and_shutdown_use_borrowed_writer_without_boxing() {
    futures::executor::block_on(async {
        let mut writer = PartialWriter {
            max_chunk: 2,
            ..PartialWriter::default()
        };

        writer
            .write_all(b"abcdef")
            .await
            .expect("write_all must retry partial writes");
        writer.flush().await.expect("flush must delegate");
        writer.shutdown().await.expect("shutdown must delegate");

        assert_eq!(&writer.bytes, b"abcdef");
        assert_eq!(writer.flushes, 1);
        assert_eq!(writer.shutdowns, 1);
    });
}

#[cfg(feature = "tokio-compat")]
#[test]
fn tokio_compat_preserves_native_reader_writer_values() {
    futures::executor::block_on(async {
        let reader = ChunkedReader {
            source: b"tokio",
            offset: 0,
            max_chunk: 2,
        };
        let mut reader = TokioCompat::from(reader);
        let mut output = [0_u8; 5];

        tokio_dep::io::AsyncReadExt::read_exact(&mut reader, &mut output)
            .await
            .expect("Tokio read_exact must read through Moirai reader");

        assert_eq!(&output, b"tokio");
        assert_eq!(reader.into_inner().offset, 5);

        let writer = PartialWriter {
            max_chunk: 2,
            ..PartialWriter::default()
        };
        let mut writer = TokioCompat::from(writer);

        tokio_dep::io::AsyncWriteExt::write_all(&mut writer, b"bridge")
            .await
            .expect("Tokio write_all must write through Moirai writer");
        tokio_dep::io::AsyncWriteExt::flush(&mut writer)
            .await
            .expect("Tokio flush must delegate through Moirai writer");
        tokio_dep::io::AsyncWriteExt::shutdown(&mut writer)
            .await
            .expect("Tokio shutdown must delegate through Moirai writer");

        let writer = writer.into_inner();
        assert_eq!(&writer.bytes, b"bridge");
        assert_eq!(writer.flushes, 1);
        assert_eq!(writer.shutdowns, 1);
    });
}

#[cfg(feature = "tokio-compat")]
#[test]
fn moirai_compat_preserves_tokio_duplex_values() {
    futures::executor::block_on(async {
        let (mut tokio_side, moirai_side) = tokio_dep::io::duplex(64);
        let mut moirai_side = MoiraiCompat::from(moirai_side);

        tokio_dep::io::AsyncWriteExt::write_all(&mut tokio_side, b"native")
            .await
            .expect("Tokio duplex write must complete");

        let mut inbound = [0_u8; 6];
        AsyncReadExt::read_exact(&mut moirai_side, &mut inbound)
            .await
            .expect("Moirai read_exact must read from Tokio duplex");
        assert_eq!(&inbound, b"native");

        AsyncWriteExt::write_all(&mut moirai_side, b"reply")
            .await
            .expect("Moirai write_all must write to Tokio duplex");
        AsyncWriteExt::shutdown(&mut moirai_side)
            .await
            .expect("Moirai shutdown must delegate to Tokio duplex");

        let mut reply = [0_u8; 5];
        tokio_dep::io::AsyncReadExt::read_exact(&mut tokio_side, &mut reply)
            .await
            .expect("Tokio read_exact must receive Moirai reply");
        assert_eq!(&reply, b"reply");

        let mut eof = [0_u8; 1];
        let count = tokio_dep::io::AsyncReadExt::read(&mut tokio_side, &mut eof)
            .await
            .expect("Tokio read must observe EOF after Moirai shutdown");
        assert_eq!(count, 0);
    });
}
