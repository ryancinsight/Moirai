//! Platform-agnostic async file I/O operations.

use std::fs::{File as StdFile, OpenOptions as StdOpenOptions};
use std::future::Future;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::pin::Pin;
use std::task::{Context, Poll};

pub struct YieldFuture {
    yielded: bool,
}

impl Future for YieldFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

pub fn yield_now() -> YieldFuture {
    YieldFuture { yielded: false }
}

/// Copy a file through the platform file-copy implementation.
pub async fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<u64> {
    yield_now().await;
    std::fs::copy(from, to)
}

/// High-performance cooperative async file handle
pub struct AsyncFile {
    inner: StdFile,
}

impl AsyncFile {
    pub async fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        yield_now().await;
        let inner = StdFile::open(path)?;
        Ok(Self { inner })
    }

    pub async fn open_with_options<P: AsRef<Path>>(
        path: P,
        read: bool,
        write: bool,
        create: bool,
        append: bool,
        truncate: bool,
    ) -> io::Result<Self> {
        yield_now().await;
        let mut opts = StdOpenOptions::new();
        opts.read(read)
            .write(write)
            .create(create)
            .append(append)
            .truncate(truncate);
        let inner = opts.open(path)?;
        Ok(Self { inner })
    }

    pub fn poll_read(&mut self, _cx: &mut Context<'_>, buf: &mut [u8]) -> Poll<io::Result<usize>> {
        Poll::Ready((&self.inner).read(buf))
    }

    pub fn poll_write(&mut self, _cx: &mut Context<'_>, buf: &[u8]) -> Poll<io::Result<usize>> {
        Poll::Ready((&self.inner).write(buf))
    }

    pub fn poll_flush(&mut self, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(self.inner.flush())
    }

    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        yield_now().await;
        (&self.inner).read(buf)
    }

    pub async fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        yield_now().await;
        (&self.inner).read_to_string(buf)
    }

    pub async fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        yield_now().await;
        (&self.inner).read_to_end(buf)
    }

    pub async fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        yield_now().await;
        (&self.inner).write(buf)
    }

    pub async fn flush(&mut self) -> io::Result<()> {
        yield_now().await;
        self.inner.flush()
    }

    pub async fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        yield_now().await;
        self.inner.seek(pos)
    }

    pub async fn sync_all(&mut self) -> io::Result<()> {
        yield_now().await;
        self.inner.sync_all()
    }

    pub async fn sync_data(&mut self) -> io::Result<()> {
        yield_now().await;
        self.inner.sync_data()
    }

    pub async fn metadata(&self) -> io::Result<std::fs::Metadata> {
        yield_now().await;
        self.inner.metadata()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "moirai_pal_async_file_{name}_{}_{}",
            std::process::id(),
            nonce
        ))
    }

    #[test]
    fn async_file_roundtrip_seek_and_metadata_are_value_semantic() {
        let path = test_path("roundtrip.bin");
        block_on(async {
            let mut file = AsyncFile::open_with_options(&path, true, true, true, false, true)
                .await
                .expect("file create must succeed");
            let written = file.write(b"alpha-beta").await.expect("write must succeed");
            assert_eq!(written, 10);
            file.flush().await.expect("flush must succeed");

            let position = file
                .seek(SeekFrom::Start(6))
                .await
                .expect("seek must succeed");
            assert_eq!(position, 6);

            let mut suffix = [0_u8; 4];
            let read = file.read(&mut suffix).await.expect("read must succeed");
            assert_eq!(read, 4);
            assert_eq!(&suffix, b"beta");

            let metadata = file.metadata().await.expect("metadata must succeed");
            assert_eq!(metadata.len(), 10);
        });
        std::fs::remove_file(&path).expect("test file cleanup must succeed");
    }

    #[test]
    fn async_file_read_to_end_preserves_source_bytes() {
        let path = test_path("source.bin");
        let expected: Vec<u8> = (0_u8..=31).map(|value| value.wrapping_mul(3)).collect();
        std::fs::write(&path, &expected).expect("source write must succeed");

        block_on(async {
            let mut file = AsyncFile::open(&path).await.expect("open must succeed");
            let mut actual = Vec::new();
            let read = file
                .read_to_end(&mut actual)
                .await
                .expect("read_to_end must succeed");
            assert_eq!(read, expected.len());
            assert_eq!(actual, expected);
        });

        std::fs::remove_file(&path).expect("test file cleanup must succeed");
    }

    #[test]
    fn async_file_copy_preserves_source_bytes() {
        let source = test_path("copy-source.bin");
        let dest = test_path("copy-dest.bin");
        let expected: Vec<u8> = (0_u8..=63).map(|value| value.wrapping_mul(5)).collect();
        std::fs::write(&source, &expected).expect("source write must succeed");

        block_on(async {
            let copied = copy(&source, &dest).await.expect("copy must succeed");
            assert_eq!(copied, expected.len() as u64);
            let actual = std::fs::read(&dest).expect("dest read must succeed");
            assert_eq!(actual, expected);
        });

        std::fs::remove_file(&source).expect("source cleanup must succeed");
        std::fs::remove_file(&dest).expect("dest cleanup must succeed");
    }
}
