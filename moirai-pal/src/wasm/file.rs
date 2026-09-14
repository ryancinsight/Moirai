//! Bounded asynchronous access to a browser-selected file.

use std::io;

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
    /// [`MAX_READ_BYTES`] is rejected before a `Blob` or `ArrayBuffer` is
    /// created, so an untrusted file cannot force a large provider allocation.
    ///
    /// # Errors
    /// Returns an I/O error when the buffer is over the provider bound, the
    /// browser reports an invalid size, cursor arithmetic overflows, or the
    /// browser rejects the bounded `Blob.arrayBuffer` operation.
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
        let array_buffer = JsFuture::from(blob.array_buffer())
            .await
            .map_err(|_| io::Error::other("browser rejected the bounded file read"))?;
        let array_buffer = js_sys::ArrayBuffer::from(array_buffer);
        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        let available = usize::try_from(uint8_array.length()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser file result length cannot be represented",
            )
        })?;
        let copied = available.min(buf.len());
        uint8_array.copy_to(&mut buf[..copied]);
        let copied = u64::try_from(copied).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser file result length cannot be represented",
            )
        })?;
        self.position = advance_cursor(self.position, copied)?;
        Ok(usize::try_from(copied).expect("invariant: copied length came from a Rust slice"))
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
