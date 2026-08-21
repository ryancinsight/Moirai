//! Positioned asynchronous read primitives.
//!
//! These contracts complement [`super::AsyncRead`], whose stateful stream
//! cursor is not sufficient for format readers that issue concurrent reads at
//! explicit file offsets. They intentionally use `std::io::Result` so the
//! runtime remains independent of any consumer's error hierarchy.

use std::future::Future;
use std::io;

/// Read exactly the requested bytes at an absolute offset without changing a
/// shared cursor.
pub trait AsyncReadAt: Send + Sync {
    /// Read exactly `buf.len()` bytes beginning at `offset`.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> impl Future<Output = io::Result<()>> + Send;
}

/// Query the byte length of an asynchronous positioned-read source.
pub trait AsyncLength: Send + Sync {
    /// Return the total number of readable bytes.
    fn len(&self) -> impl Future<Output = io::Result<u64>> + Send;

    /// Whether the source has no readable bytes.
    ///
    /// Provided rather than required: a zero length is the only way to be
    /// empty, so an implementor that answers `len` has already answered this.
    /// It exists because a caller checking for an empty source should not have
    /// to compare against zero itself, and because a `len` without an
    /// `is_empty` is a trait-shape defect that `clippy::len_without_is_empty`
    /// correctly rejects.
    fn is_empty(&self) -> impl Future<Output = io::Result<bool>> + Send {
        async move { Ok(self.len().await? == 0) }
    }
}

/// A cloneable, read-only in-memory positioned source for tests and examples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsyncMemReader {
    data: Vec<u8>,
}

impl AsyncMemReader {
    /// Construct a reader from owned bytes.
    #[must_use]
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Construct an empty reader.
    #[must_use]
    pub fn new() -> Self {
        Self::from_bytes(Vec::new())
    }

    /// Borrow the complete source contents.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume the reader and return its source contents.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }
}

impl Default for AsyncMemReader {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncReadAt for AsyncMemReader {
    async fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        if buf.is_empty() {
            return Ok(());
        }

        let offset = usize::try_from(offset).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "read offset does not fit usize",
            )
        })?;
        let end = offset.checked_add(buf.len()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "read range overflows usize")
        })?;
        if end > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "positioned read needs {end} bytes but source contains {}",
                    self.data.len()
                ),
            ));
        }

        buf.copy_from_slice(&self.data[offset..end]);
        Ok(())
    }
}

impl AsyncLength for AsyncMemReader {
    async fn len(&self) -> io::Result<u64> {
        u64::try_from(self.data.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "source length does not fit u64")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn positioned_reader_returns_exact_bytes_and_length() {
        let reader = AsyncMemReader::from_bytes(vec![10, 20, 30, 40]);
        let mut output = [0; 2];

        block_on(async {
            reader
                .read_at(1, &mut output)
                .await
                .expect("read must succeed");
            assert_eq!(output, [20, 30]);
            assert_eq!(reader.len().await.expect("length must succeed"), 4);
        });
    }

    #[test]
    fn positioned_reader_rejects_short_reads() {
        let reader = AsyncMemReader::from_bytes(vec![1, 2]);
        let mut output = [0; 2];

        let error = block_on(reader.read_at(1, &mut output)).expect_err("read must fail");
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn zero_length_reads_do_not_require_source_bytes() {
        let reader = AsyncMemReader::new();
        let mut output = [];

        block_on(reader.read_at(u64::MAX, &mut output)).expect("empty read must succeed");
    }

    #[test]
    fn empty_source_reports_is_empty() {
        // The provided `is_empty` must agree with `len` at the boundary that
        // matters: a source with no bytes, and one with exactly one.
        block_on(async {
            let empty = AsyncMemReader::new();
            assert_eq!(empty.len().await.expect("len"), 0);
            assert!(empty.is_empty().await.expect("is_empty"));

            let one = AsyncMemReader::from_bytes(vec![0u8; 1]);
            assert_eq!(one.len().await.expect("len"), 1);
            assert!(!one.is_empty().await.expect("is_empty"));
        });
    }
}
