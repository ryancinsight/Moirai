//! Buffered-read mapping: `poll_fill_buf` and `consume` in both directions.

use super::{Wakes, poll_under};
use crate::io::{AsyncBufRead, AsyncRead, MoiraiCompat, TokioCompat};
use futures::future::poll_fn;
use std::io;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};
use tokio_dep as tokio;
use tokio_dep::io::{AsyncBufReadExt as _, AsyncWriteExt as _};

#[derive(Default)]
struct GateState {
    open: bool,
    waker: Option<Waker>,
}

/// Holds a [`Staged`] reader back until the test opens it.
#[derive(Clone, Default)]
struct Gate(Arc<Mutex<GateState>>);

impl Gate {
    fn state(&self) -> MutexGuard<'_, GateState> {
        self.0
            .lock()
            .expect("invariant: gate lock is never poisoned")
    }

    fn open(&self) {
        let mut state = self.state();
        state.open = true;
        let waker = state.waker.take();
        drop(state);
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// A Moirai buffered reader that exposes `data` in `chunk`-byte windows once
/// its gate is open, and registers the polling waker while it is closed.
struct Staged {
    data: Vec<u8>,
    position: usize,
    chunk: usize,
    gate: Gate,
}

impl Staged {
    fn new(data: &[u8], chunk: usize, gate: Gate) -> Self {
        Self {
            data: data.to_vec(),
            position: 0,
            chunk,
            gate,
        }
    }

    fn window(&self, cx: &Context<'_>) -> Poll<&[u8]> {
        let mut gate = self.gate.state();
        if !gate.open {
            gate.waker = Some(cx.waker().clone());
            return Poll::Pending;
        }
        drop(gate);
        let end = (self.position + self.chunk).min(self.data.len());
        Poll::Ready(&self.data[self.position..end])
    }
}

impl AsyncRead for Staged {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let Poll::Ready(window) = this.window(cx) else {
            return Poll::Pending;
        };
        let count = window.len().min(buf.len());
        buf[..count].copy_from_slice(&window[..count]);
        this.position += count;
        Poll::Ready(Ok(count))
    }
}

impl AsyncBufRead for Staged {
    fn poll_fill_buf(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<&[u8]>> {
        self.get_mut().window(cx).map(Ok)
    }

    fn consume(self: Pin<&mut Self>, amt: usize) {
        let this = self.get_mut();
        this.position += amt;
        assert!(this.position <= this.data.len(), "consumed past the data");
    }
}

/// Every window a Moirai buffered reader exposes until EOF, consuming each whole.
async fn moirai_windows<R: AsyncBufRead + Unpin>(mut reader: R) -> Vec<Vec<u8>> {
    let mut windows = Vec::new();
    loop {
        let window = poll_fn(|cx| {
            AsyncBufRead::poll_fill_buf(Pin::new(&mut reader), cx).map_ok(<[u8]>::to_vec)
        })
        .await
        .expect("Moirai fill_buf must succeed");
        if window.is_empty() {
            return windows;
        }
        AsyncBufRead::consume(Pin::new(&mut reader), window.len());
        windows.push(window);
    }
}

/// Every window a Tokio buffered reader exposes until EOF, consuming each whole.
async fn tokio_windows<R: tokio::io::AsyncBufRead + Unpin>(mut reader: R) -> Vec<Vec<u8>> {
    let mut windows = Vec::new();
    loop {
        let window = reader
            .fill_buf()
            .await
            .expect("Tokio fill_buf must succeed")
            .to_vec();
        if window.is_empty() {
            return windows;
        }
        tokio::io::AsyncBufRead::consume(Pin::new(&mut reader), window.len());
        windows.push(window);
    }
}

fn fill_tokio<R: tokio::io::AsyncBufRead + Unpin>(
    reader: &mut R,
    wakes: &Arc<Wakes>,
) -> Poll<io::Result<Vec<u8>>> {
    poll_under(wakes, |cx| {
        tokio::io::AsyncBufRead::poll_fill_buf(Pin::new(reader), cx).map_ok(<[u8]>::to_vec)
    })
}

fn fill_moirai<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    wakes: &Arc<Wakes>,
) -> Poll<io::Result<Vec<u8>>> {
    poll_under(wakes, |cx| {
        AsyncBufRead::poll_fill_buf(Pin::new(reader), cx).map_ok(<[u8]>::to_vec)
    })
}

const TEXT: &[u8] = b"alpha\nbeta\ngamma\n";

fn open_gate() -> Gate {
    let gate = Gate::default();
    gate.open();
    gate
}

#[test]
fn buffered_windows_match_across_native_and_wrapped_paths() {
    let expected: Vec<Vec<u8>> = TEXT.chunks(4).map(<[u8]>::to_vec).collect();

    let native = futures::executor::block_on(moirai_windows(Staged::new(TEXT, 4, open_gate())));
    assert_eq!(native, expected);

    let through_tokio = futures::executor::block_on(tokio_windows(TokioCompat::new(Staged::new(
        TEXT,
        4,
        open_gate(),
    ))));
    assert_eq!(through_tokio, native);

    let buffered = tokio::io::BufReader::with_capacity(4, TEXT);
    let through_moirai = futures::executor::block_on(moirai_windows(MoiraiCompat::new(buffered)));
    assert_eq!(through_moirai, native);
}

#[test]
fn tokio_line_reads_over_a_moirai_buffered_reader_keep_line_boundaries() {
    let mut reader = TokioCompat::new(Staged::new(TEXT, 4, open_gate()));

    let mut lines = Vec::new();
    futures::executor::block_on(async {
        for _ in 0..3 {
            let mut line = Vec::new();
            reader
                .read_until(b'\n', &mut line)
                .await
                .expect("Tokio read_until must succeed");
            lines.push(line);
        }
        let mut rest = Vec::new();
        let count = reader
            .read_until(b'\n', &mut rest)
            .await
            .expect("EOF read_until must succeed");
        assert_eq!(count, 0);
    });

    assert_eq!(
        lines,
        [b"alpha\n".to_vec(), b"beta\n".to_vec(), b"gamma\n".to_vec()]
    );
    assert_eq!(reader.into_inner().position, TEXT.len());
}

#[test]
fn tokio_fill_buf_registers_the_polling_context_and_consume_advances() {
    let gate = Gate::default();
    let mut reader = TokioCompat::new(Staged::new(TEXT, 4, gate.clone()));
    let (first, second) = (Wakes::new(), Wakes::new());

    for wakes in [&first, &second] {
        assert!(matches!(fill_tokio(&mut reader, wakes), Poll::Pending));
    }

    gate.open();
    assert_eq!((first.count(), second.count()), (0, 1));

    assert!(matches!(
        fill_tokio(&mut reader, &first),
        Poll::Ready(Ok(window)) if window == b"alph"
    ));
    tokio::io::AsyncBufRead::consume(Pin::new(&mut reader), 2);
    assert!(matches!(
        fill_tokio(&mut reader, &first),
        Poll::Ready(Ok(window)) if window == b"pha\n"
    ));
}

#[test]
fn moirai_fill_buf_registers_the_polling_context_and_consume_advances() {
    let (mut far, near) = tokio::io::duplex(16);
    let mut reader = MoiraiCompat::new(tokio::io::BufReader::with_capacity(4, near));
    let (first, second) = (Wakes::new(), Wakes::new());

    for wakes in [&first, &second] {
        assert!(matches!(fill_moirai(&mut reader, wakes), Poll::Pending));
    }

    futures::executor::block_on(far.write_all(b"alphabet")).expect("duplex write must succeed");
    assert_eq!((first.count(), second.count()), (0, 1));

    assert!(matches!(
        fill_moirai(&mut reader, &first),
        Poll::Ready(Ok(window)) if window == b"alph"
    ));
    AsyncBufRead::consume(Pin::new(&mut reader), 3);
    assert!(matches!(
        fill_moirai(&mut reader, &first),
        Poll::Ready(Ok(window)) if window == b"h"
    ));
}
