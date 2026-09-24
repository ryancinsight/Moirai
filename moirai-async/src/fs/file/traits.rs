//! The crate's async I/O traits for [`File`], and its positioned reads.

use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use moirai_pal::fs::File as Handle;

use super::File;
use crate::blocking::Abandoned;
use crate::fs::pool;
use crate::io::{AsyncLength, AsyncRead, AsyncReadAt, AsyncWrite};

impl AsyncReadAt for File {
    async fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        if buf.is_empty() {
            return Ok(());
        }

        let requested = u64::try_from(buf.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "read length does not fit u64")
        })?;
        offset.checked_add(requested).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "read range overflows u64")
        })?;

        self.fence.settled().await;
        let handle = Arc::clone(&self.handle);
        let len = buf.len();
        let (data, filled) = pool()
            .run(Abandoned::Skip, move || read_exact_at(&handle, len, offset))
            .await?;
        // Bytes read before a failure still reach the caller, as with a
        // positioned read loop run in place.
        buf[..data.len()].copy_from_slice(&data);
        filled
    }
}

/// Fill `len` bytes from `offset` with positioned reads, which leave the
/// stream cursor alone. Returns the bytes read, and whether all `len` were.
fn read_exact_at(handle: &Handle, len: usize, offset: u64) -> (Vec<u8>, io::Result<()>) {
    let mut data = vec![0; len];
    let mut filled = 0;
    let result = loop {
        if filled == len {
            break Ok(());
        }
        let Some(position) = u64::try_from(filled)
            .ok()
            .and_then(|filled| offset.checked_add(filled))
        else {
            break Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read range overflows u64",
            ));
        };
        match handle.read_at(&mut data[filled..], position) {
            Ok(0) => {
                break Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "positioned read reached end of file before filling buffer",
                ));
            }
            Ok(count) => filled += count,
            Err(error) => break Err(error),
        }
    };
    data.truncate(filled);
    (data, result)
}

impl AsyncLength for File {
    async fn len(&self) -> io::Result<u64> {
        Ok(self.metadata().await?.len())
    }
}

impl AsyncRead for File {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let n = ready!(self.poll_read_into(cx, buf))?;
        self.stats.bytes_read += n as u64;
        self.stats.read_operations += 1;
        Poll::Ready(Ok(n))
    }
}

impl AsyncWrite for File {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let n = ready!(self.poll_write_behind(cx, buf))?;
        self.stats.bytes_written += n as u64;
        self.stats.write_operations += 1;
        Poll::Ready(Ok(n))
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.poll_settled(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.poll_settled(cx)
    }
}
