//! Bounded asynchronous access to a browser-selected file.

use std::io;

use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use crate::drop_validation::parse_size;
use crate::file_policy::{advance_cursor, validate_buffer_length};

/// Maximum number of bytes copied by one browser file read.
pub const MAX_READ_BYTES: usize = crate::file_policy::MAX_READ_BYTES;

/// Owns a browser `File` and a validated sequential read cursor.
pub struct WebFile {
    file_handle: web_sys::File,
    position: u64,
}

impl WebFile {
    /// Creates a reader from a browser `File` object.
    pub fn from_js_file(file: web_sys::File) -> Self {
        Self {
            file_handle: file,
            position: 0,
        }
    }

    /// Reads the next bounded chunk into the caller-provided buffer.
    ///
    /// The browser file remains outside Rust-owned storage. A read larger than
    /// [`MAX_READ_BYTES`] is rejected before a `Blob` or stream scratch buffer is
    /// created, so an untrusted file cannot force a large provider allocation.
    ///
    /// # Errors
    /// Returns an I/O error when the buffer is over the provider bound, the
    /// browser reports an invalid size, cursor arithmetic overflows, or the
    /// browser rejects the bounded `Blob.stream` operation.
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let requested = validate_buffer_length(buf.len())?;
        if buf.is_empty() {
            return Ok(0);
        }

        let size = checked_size(&self.file_handle)?;
        if self.position > size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser file cursor is beyond the file size",
            ));
        }
        let length = requested.min(size - self.position);
        if length == 0 {
            return Ok(0);
        }
        let end_position = advance_cursor(self.position, length)?;
        let blob = self
            .file_handle
            .slice_with_f64_and_f64(self.position as f64, end_position as f64)
            .map_err(|_| io::Error::other("browser rejected the file slice"))?;
        let target_len = usize::try_from(length).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser file read length cannot be represented",
            )
        })?;
        let copied = read_blob_stream(blob, &mut buf[..target_len]).await?;
        let copied = u64::try_from(copied).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser file result length cannot be represented",
            )
        })?;
        self.position = advance_cursor(self.position, copied)?;
        usize::try_from(copied).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser file result length cannot be represented",
            )
        })
    }

    /// Returns the validated browser-reported file size in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        let size = self.file_handle.size();
        debug_assert!(
            size.is_finite() && !size.is_sign_negative() && size.fract() == 0.0,
            "invariant: browser File.size is a finite non-negative integer"
        );
        size as u64
    }

    /// Returns the current sequential read cursor.
    #[must_use]
    pub const fn position(&self) -> u64 {
        self.position
    }

    /// Moves the read cursor within the browser-reported file size.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when `pos` is beyond the file.
    pub fn seek(&mut self, pos: u64) -> io::Result<()> {
        let size = checked_size(&self.file_handle)?;
        if pos > size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser file cursor is beyond the file size",
            ));
        }
        self.position = pos;
        Ok(())
    }
}

fn checked_size(file: &web_sys::File) -> io::Result<u64> {
    parse_size(file.size())
}

struct StreamReader {
    reader: web_sys::ReadableStreamDefaultReader,
    released: bool,
}

impl StreamReader {
    fn release(&mut self) {
        if !self.released {
            self.reader.release_lock();
            self.released = true;
        }
    }
}

impl Drop for StreamReader {
    fn drop(&mut self) {
        self.release();
    }
}

async fn read_blob_stream(blob: web_sys::Blob, buffer: &mut [u8]) -> io::Result<usize> {
    let stream = blob.stream();
    let reader = stream
        .get_reader()
        .dyn_into::<web_sys::ReadableStreamDefaultReader>()
        .map_err(|_| io::Error::other("browser rejected the bounded file stream reader"))?;
    let mut reader = StreamReader {
        reader,
        released: false,
    };
    let mut copied = 0usize;

    while copied < buffer.len() {
        let result = JsFuture::from(reader.reader.read())
            .await
            .map_err(|_| io::Error::other("browser rejected the bounded file stream read"))?
            .unchecked_into::<web_sys::ReadableStreamReadResult>();
        let done = result
            .get_done()
            .ok_or_else(|| io::Error::other("browser file stream omitted its completion flag"))?;
        let value = result.get_value();
        let chunk =
            if value.is_undefined() || value.is_null() {
                if done {
                    None
                } else {
                    return Err(io::Error::other(
                        "browser file stream omitted a chunk before completion",
                    ));
                }
            } else {
                Some(value.dyn_into::<js_sys::Uint8Array>().map_err(|_| {
                    io::Error::other("browser file stream returned a non-byte chunk")
                })?)
            };
        let chunk_length = chunk.as_ref().map_or(0, js_sys::Uint8Array::length);
        let chunk_length = usize::try_from(chunk_length).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser file chunk length cannot be represented",
            )
        })?;
        let remaining = buffer.len() - copied;
        if chunk_length > remaining {
            return Err(io::Error::other(
                "browser file stream exceeded the bounded read request",
            ));
        }
        if chunk_length == 0 && !done {
            return Err(io::Error::other(
                "browser file stream returned an empty incomplete chunk",
            ));
        }
        if let Some(chunk) = chunk {
            chunk.copy_to(&mut buffer[copied..copied + chunk_length]);
            copied += chunk_length;
        }
        if done {
            break;
        }
    }

    reader.release();
    Ok(copied)
}
