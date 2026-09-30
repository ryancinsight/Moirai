//! Value and readiness tests for [`TokioCompat`] and [`MoiraiCompat`].
//!
//! The native path is [`PipeEnd`], an in-memory bounded pipe that stores the
//! waker of the most recent `Pending` poll on each side. Driving the same
//! payload through it directly, through `TokioCompat`, through a Tokio
//! `duplex` behind `MoiraiCompat`, and through both wrappers stacked must
//! yield the same bytes. Counting wakers then show which task context a
//! `Pending` poll registered.

use super::{MoiraiCompat, TokioCompat};
use crate::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use std::collections::VecDeque;
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Wake, Waker};
use tokio_dep as tokio;
use tokio_dep::io::{AsyncReadExt as _, AsyncWriteExt as _};

mod buffered;
mod vectored;

#[derive(Default)]
struct PipeState {
    bytes: VecDeque<u8>,
    capacity: usize,
    closed: bool,
    write_pendings: usize,
    read_waker: Option<Waker>,
    write_waker: Option<Waker>,
}

/// One end of a bounded in-memory byte pipe; both ends share the state.
#[derive(Clone)]
struct PipeEnd(Arc<Mutex<PipeState>>);

fn pipe(capacity: usize) -> (PipeEnd, PipeEnd) {
    let state = Arc::new(Mutex::new(PipeState {
        capacity,
        ..PipeState::default()
    }));
    (PipeEnd(Arc::clone(&state)), PipeEnd(state))
}

impl PipeEnd {
    fn lock(&self) -> MutexGuard<'_, PipeState> {
        self.0
            .lock()
            .expect("invariant: pipe lock is never poisoned")
    }
}

impl AsyncRead for PipeEnd {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }
        let mut state = self.lock();
        if state.bytes.is_empty() {
            if state.closed {
                return Poll::Ready(Ok(0));
            }
            state.read_waker = Some(cx.waker().clone());
            return Poll::Pending;
        }
        let count = state.bytes.len().min(buf.len());
        for (slot, byte) in buf.iter_mut().zip(state.bytes.drain(..count)) {
            *slot = byte;
        }
        let writer = state.write_waker.take();
        drop(state);
        if let Some(waker) = writer {
            waker.wake();
        }
        Poll::Ready(Ok(count))
    }
}

impl AsyncWrite for PipeEnd {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }
        let mut state = self.lock();
        if state.closed {
            return Poll::Ready(Err(io::ErrorKind::BrokenPipe.into()));
        }
        let free = state.capacity - state.bytes.len();
        if free == 0 {
            state.write_pendings += 1;
            state.write_waker = Some(cx.waker().clone());
            return Poll::Pending;
        }
        let count = free.min(buf.len());
        state.bytes.extend(&buf[..count]);
        let reader = state.read_waker.take();
        drop(state);
        if let Some(waker) = reader {
            waker.wake();
        }
        Poll::Ready(Ok(count))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let mut state = self.lock();
        state.closed = true;
        let reader = state.read_waker.take();
        drop(state);
        if let Some(waker) = reader {
            waker.wake();
        }
        Poll::Ready(Ok(()))
    }
}

/// A waker that counts how often it is woken.
#[derive(Default)]
struct Wakes(AtomicUsize);

impl Wake for Wakes {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

impl Wakes {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn count(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }
}

/// Runs one poll with `wakes` as the task context.
fn poll_under<T>(wakes: &Arc<Wakes>, poll: impl FnOnce(&mut Context<'_>) -> T) -> T {
    let waker = Waker::from(Arc::clone(wakes));
    poll(&mut Context::from_waker(&waker))
}

/// Deterministic xorshift bytes, long enough to span many pipe refills.
fn payload() -> Vec<u8> {
    let mut state = 0x9E37_79B9_u32;
    (0..65_549)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state.to_le_bytes()[0]
        })
        .collect()
}

async fn send_moirai<W: AsyncWrite + Unpin>(mut writer: W, payload: &[u8]) {
    writer
        .write_all(payload)
        .await
        .expect("Moirai write_all must retry partial writes");
    writer.shutdown().await.expect("Moirai shutdown must close");
}

/// Reads to EOF in 11-byte steps so the reader also makes partial progress.
async fn collect_moirai<R: AsyncRead + Unpin>(mut reader: R) -> Vec<u8> {
    let mut output = Vec::new();
    let mut chunk = [0_u8; 11];
    loop {
        let count = reader
            .read(&mut chunk)
            .await
            .expect("Moirai read must succeed");
        if count == 0 {
            return output;
        }
        output.extend_from_slice(&chunk[..count]);
    }
}

async fn send_tokio<W: tokio::io::AsyncWrite + Unpin>(mut writer: W, payload: &[u8]) {
    writer
        .write_all(payload)
        .await
        .expect("Tokio write_all must retry partial writes");
    writer.shutdown().await.expect("Tokio shutdown must close");
}

async fn collect_tokio<R: tokio::io::AsyncRead + Unpin>(mut reader: R) -> Vec<u8> {
    let mut output = Vec::new();
    reader
        .read_to_end(&mut output)
        .await
        .expect("Tokio read_to_end must reach EOF");
    output
}

#[test]
fn chunked_transfer_is_byte_identical_on_native_and_wrapped_paths() {
    const CAPACITY: usize = 7;
    let payload = payload();

    let (writer, reader) = pipe(CAPACITY);
    let state = Arc::clone(&writer.0);
    let (_, native) = futures::executor::block_on(async {
        futures::join!(send_moirai(writer, &payload), collect_moirai(reader))
    });
    assert!(native == payload, "native pipe must deliver the payload");
    assert!(
        state
            .lock()
            .expect("invariant: pipe lock is never poisoned")
            .write_pendings
            > 0,
        "a 7-byte pipe must exert backpressure on a 64 KiB payload"
    );

    let (writer, reader) = pipe(CAPACITY);
    let (_, through_tokio) = futures::executor::block_on(async {
        futures::join!(
            send_tokio(TokioCompat::new(writer), &payload),
            collect_tokio(TokioCompat::new(reader))
        )
    });
    assert!(through_tokio == native, "TokioCompat must not change bytes");

    let (near, far) = tokio::io::duplex(CAPACITY);
    let (_, through_moirai) = futures::executor::block_on(async {
        futures::join!(
            send_moirai(MoiraiCompat::new(near), &payload),
            collect_moirai(MoiraiCompat::new(far))
        )
    });
    assert!(
        through_moirai == native,
        "MoiraiCompat must not change bytes"
    );

    let (writer, reader) = pipe(CAPACITY);
    let (_, stacked) = futures::executor::block_on(async {
        futures::join!(
            send_moirai(MoiraiCompat::new(TokioCompat::new(writer)), &payload),
            collect_moirai(MoiraiCompat::new(TokioCompat::new(reader)))
        )
    });
    assert!(
        stacked == native,
        "Moirai over Tokio over Moirai must be identity"
    );

    let (near, far) = tokio::io::duplex(CAPACITY);
    let (_, stacked) = futures::executor::block_on(async {
        futures::join!(
            send_tokio(TokioCompat::new(MoiraiCompat::new(near)), &payload),
            collect_tokio(TokioCompat::new(MoiraiCompat::new(far)))
        )
    });
    assert!(
        stacked == native,
        "Tokio over Moirai over Tokio must be identity"
    );
}

#[test]
fn tokio_read_registers_the_polling_context_and_repoll_replaces_it() {
    let (mut writer, reader) = pipe(8);
    let mut reader = TokioCompat::new(reader);
    let (first, second) = (Wakes::new(), Wakes::new());
    let mut storage = [0_u8; 4];
    let mut buf = tokio::io::ReadBuf::new(&mut storage);

    for wakes in [&first, &second] {
        let poll = poll_under(wakes, |cx| {
            tokio::io::AsyncRead::poll_read(Pin::new(&mut reader), cx, &mut buf)
        });
        assert!(matches!(poll, Poll::Pending));
        assert!(buf.filled().is_empty());
    }

    futures::executor::block_on(writer.write_all(b"xy")).expect("pipe write must succeed");
    assert_eq!((first.count(), second.count()), (0, 1));

    let poll = poll_under(&first, |cx| {
        tokio::io::AsyncRead::poll_read(Pin::new(&mut reader), cx, &mut buf)
    });
    assert!(matches!(poll, Poll::Ready(Ok(()))));
    assert_eq!(buf.filled(), b"xy");
}

#[test]
fn moirai_read_registers_the_polling_context_and_repoll_replaces_it() {
    let (mut far, near) = tokio::io::duplex(8);
    let mut near = MoiraiCompat::new(near);
    let (first, second) = (Wakes::new(), Wakes::new());
    let mut buf = [0_u8; 4];

    for wakes in [&first, &second] {
        let poll = poll_under(wakes, |cx| {
            AsyncRead::poll_read(Pin::new(&mut near), cx, &mut buf)
        });
        assert!(matches!(poll, Poll::Pending));
    }

    futures::executor::block_on(far.write_all(b"xy")).expect("duplex write must succeed");
    assert_eq!((first.count(), second.count()), (0, 1));

    let poll = poll_under(&first, |cx| {
        AsyncRead::poll_read(Pin::new(&mut near), cx, &mut buf)
    });
    assert!(matches!(poll, Poll::Ready(Ok(2))));
    assert_eq!(&buf[..2], b"xy");
}

#[test]
fn tokio_write_backpressure_wakes_the_latest_polling_context() {
    let (writer, mut reader) = pipe(3);
    let mut writer = TokioCompat::new(writer);
    let (first, second) = (Wakes::new(), Wakes::new());

    let poll = poll_under(&first, |cx| {
        tokio::io::AsyncWrite::poll_write(Pin::new(&mut writer), cx, b"abcde")
    });
    assert!(matches!(poll, Poll::Ready(Ok(3))));

    for wakes in [&first, &second] {
        let poll = poll_under(wakes, |cx| {
            tokio::io::AsyncWrite::poll_write(Pin::new(&mut writer), cx, b"de")
        });
        assert!(matches!(poll, Poll::Pending));
    }

    let mut drained = [0_u8; 2];
    futures::executor::block_on(reader.read_exact(&mut drained)).expect("pipe read must succeed");
    assert_eq!(&drained, b"ab");
    assert_eq!((first.count(), second.count()), (0, 1));

    let poll = poll_under(&first, |cx| {
        tokio::io::AsyncWrite::poll_write(Pin::new(&mut writer), cx, b"de")
    });
    assert!(matches!(poll, Poll::Ready(Ok(2))));
}

#[test]
fn moirai_write_backpressure_wakes_the_latest_polling_context() {
    let (near, mut far) = tokio::io::duplex(3);
    let mut near = MoiraiCompat::new(near);
    let (first, second) = (Wakes::new(), Wakes::new());

    let poll = poll_under(&first, |cx| {
        AsyncWrite::poll_write(Pin::new(&mut near), cx, b"abcde")
    });
    assert!(matches!(poll, Poll::Ready(Ok(3))));

    for wakes in [&first, &second] {
        let poll = poll_under(wakes, |cx| {
            AsyncWrite::poll_write(Pin::new(&mut near), cx, b"de")
        });
        assert!(matches!(poll, Poll::Pending));
    }

    let mut drained = [0_u8; 2];
    futures::executor::block_on(far.read_exact(&mut drained)).expect("duplex read must succeed");
    assert_eq!(&drained, b"ab");
    assert_eq!((first.count(), second.count()), (0, 1));

    let poll = poll_under(&first, |cx| {
        AsyncWrite::poll_write(Pin::new(&mut near), cx, b"de")
    });
    assert!(matches!(poll, Poll::Ready(Ok(2))));
}

#[test]
fn tokio_shutdown_reaches_the_reader_as_eof_and_wakes_it() {
    let (writer, mut reader) = pipe(8);
    let mut writer = TokioCompat::new(writer);
    let waiting = Wakes::new();

    futures::executor::block_on(async {
        writer
            .write_all(b"ab")
            .await
            .expect("Tokio write must succeed");
        let mut head = [0_u8; 2];
        reader
            .read_exact(&mut head)
            .await
            .expect("Moirai read must succeed");
        assert_eq!(&head, b"ab");
    });

    let mut byte = [0_u8; 1];
    let poll = poll_under(&waiting, |cx| {
        AsyncRead::poll_read(Pin::new(&mut reader), cx, &mut byte)
    });
    assert!(matches!(poll, Poll::Pending));

    futures::executor::block_on(writer.shutdown()).expect("Tokio shutdown must delegate");
    assert_eq!(waiting.count(), 1);

    futures::executor::block_on(async {
        for _ in 0..2 {
            let count = reader.read(&mut byte).await.expect("EOF read must succeed");
            assert_eq!(count, 0);
        }
        let error = writer
            .write(b"c")
            .await
            .expect_err("write after shutdown must fail");
        assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
    });
}

#[test]
fn moirai_observes_tokio_peer_eof_and_broken_pipe() {
    let (mut far, near) = tokio::io::duplex(8);
    let mut near = MoiraiCompat::new(near);

    futures::executor::block_on(async {
        far.write_all(b"ab")
            .await
            .expect("Tokio write must succeed");
        far.shutdown().await.expect("Tokio shutdown must close");

        let mut head = [0_u8; 2];
        near.read_exact(&mut head)
            .await
            .expect("Moirai read must drain");
        assert_eq!(&head, b"ab");
        assert_eq!(
            near.read(&mut head).await.expect("EOF read must succeed"),
            0
        );

        drop(far);
        let error = near
            .write(b"c")
            .await
            .expect_err("write to a dropped peer must fail");
        assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
    });
}

#[test]
fn zero_length_operations_transfer_nothing_and_keep_pending_data() {
    let (writer, reader) = pipe(8);
    let (mut writer, mut reader) = (TokioCompat::new(writer), TokioCompat::new(reader));
    futures::executor::block_on(async {
        assert_eq!(writer.write(&[]).await.expect("empty Tokio write"), 0);
        writer
            .write_all(b"ab")
            .await
            .expect("Tokio write must succeed");
        assert_eq!(reader.read(&mut []).await.expect("empty Tokio read"), 0);
        let mut both = [0_u8; 2];
        reader
            .read_exact(&mut both)
            .await
            .expect("data must survive empty read");
        assert_eq!(&both, b"ab");
    });

    let (far, near) = tokio::io::duplex(8);
    let (mut far, mut near) = (far, MoiraiCompat::new(near));
    futures::executor::block_on(async {
        assert_eq!(near.write(&[]).await.expect("empty Moirai write"), 0);
        far.write_all(b"ab")
            .await
            .expect("Tokio write must succeed");
        assert_eq!(near.read(&mut []).await.expect("empty Moirai read"), 0);
        let mut both = [0_u8; 2];
        near.read_exact(&mut both)
            .await
            .expect("data must survive empty read");
        assert_eq!(&both, b"ab");
    });
}
