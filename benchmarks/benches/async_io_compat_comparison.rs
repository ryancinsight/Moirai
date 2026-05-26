//! Async I/O compatibility comparison for native and Tokio trait adapters.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use moirai_async::io::{
    AsyncRead as MoiraiAsyncRead, AsyncReadExt as MoiraiAsyncReadExt,
    AsyncWrite as MoiraiAsyncWrite, AsyncWriteExt as MoiraiAsyncWriteExt, TokioCompat,
};
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

const PAYLOAD_BYTES: usize = 4096;
const READ_CHUNK: usize = 128;
const WRITE_CHUNK: usize = 128;
const PAYLOAD: [u8; PAYLOAD_BYTES] = [0xA5; PAYLOAD_BYTES];

struct BenchReader {
    offset: usize,
}

impl BenchReader {
    fn new() -> Self {
        Self { offset: 0 }
    }
}

impl MoiraiAsyncRead for BenchReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        if self.offset == PAYLOAD.len() {
            return Poll::Ready(Ok(0));
        }

        let remaining = PAYLOAD.len() - self.offset;
        let count = remaining.min(buf.len()).min(READ_CHUNK);
        buf[..count].copy_from_slice(&PAYLOAD[self.offset..self.offset + count]);
        self.offset += count;
        Poll::Ready(Ok(count))
    }
}

struct BenchWriter {
    bytes: [u8; PAYLOAD_BYTES],
    len: usize,
    shutdowns: usize,
}

impl BenchWriter {
    fn new() -> Self {
        Self {
            bytes: [0; PAYLOAD_BYTES],
            len: 0,
            shutdowns: 0,
        }
    }
}

impl MoiraiAsyncWrite for BenchWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let capacity = self.bytes.len() - self.len;
        let count = capacity.min(buf.len()).min(WRITE_CHUNK);
        let start = self.len;
        let end = start + count;
        self.bytes[start..end].copy_from_slice(&buf[..count]);
        self.len = end;
        Poll::Ready(Ok(count))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.shutdowns += 1;
        Poll::Ready(Ok(()))
    }
}

fn moirai_native_read_exact() -> [u8; PAYLOAD_BYTES] {
    let mut reader = BenchReader::new();
    let mut output = [0_u8; PAYLOAD_BYTES];
    futures::executor::block_on(MoiraiAsyncReadExt::read_exact(&mut reader, &mut output))
        .expect("Moirai read_exact must complete");
    assert_eq!(output, PAYLOAD);
    output
}

fn tokio_compat_read_exact() -> [u8; PAYLOAD_BYTES] {
    let reader = BenchReader::new();
    let mut reader = TokioCompat::from(reader);
    let mut output = [0_u8; PAYLOAD_BYTES];
    futures::executor::block_on(tokio::io::AsyncReadExt::read_exact(
        &mut reader,
        &mut output,
    ))
    .expect("Tokio read_exact must complete through TokioCompat");
    assert_eq!(output, PAYLOAD);
    output
}

fn moirai_native_write_shutdown() -> usize {
    let mut writer = BenchWriter::new();
    futures::executor::block_on(async {
        MoiraiAsyncWriteExt::write_all(&mut writer, &PAYLOAD).await?;
        MoiraiAsyncWriteExt::shutdown(&mut writer).await
    })
    .expect("Moirai write_all and shutdown must complete");

    assert_eq!(&writer.bytes[..writer.len], &PAYLOAD);
    assert_eq!(writer.shutdowns, 1);
    writer.len
}

fn tokio_compat_write_shutdown() -> usize {
    let writer = BenchWriter::new();
    let mut writer = TokioCompat::from(writer);
    futures::executor::block_on(async {
        tokio::io::AsyncWriteExt::write_all(&mut writer, &PAYLOAD).await?;
        tokio::io::AsyncWriteExt::shutdown(&mut writer).await
    })
    .expect("Tokio write_all and shutdown must complete through TokioCompat");

    let writer = writer.into_inner();
    assert_eq!(&writer.bytes[..writer.len], &PAYLOAD);
    assert_eq!(writer.shutdowns, 1);
    writer.len
}

fn async_io_compat_comparison(c: &mut Criterion) {
    let mut read_group = c.benchmark_group("async_io_compat_read_exact");
    read_group.sample_size(20);
    read_group.warm_up_time(Duration::from_millis(300));
    read_group.measurement_time(Duration::from_secs(2));
    read_group.bench_function("moirai_native", |b| {
        b.iter(|| black_box(moirai_native_read_exact()))
    });
    read_group.bench_function("tokio_compat", |b| {
        b.iter(|| black_box(tokio_compat_read_exact()))
    });
    read_group.finish();

    let mut write_group = c.benchmark_group("async_io_compat_write_shutdown");
    write_group.sample_size(20);
    write_group.warm_up_time(Duration::from_millis(300));
    write_group.measurement_time(Duration::from_secs(2));
    write_group.bench_function("moirai_native", |b| {
        b.iter(|| black_box(moirai_native_write_shutdown()))
    });
    write_group.bench_function("tokio_compat", |b| {
        b.iter(|| black_box(tokio_compat_write_shutdown()))
    });
    write_group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().without_plots();
    targets = async_io_compat_comparison
}
criterion_main!(benches);
