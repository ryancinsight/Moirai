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
    size_bytes: u64,
    position: u64,
}

impl WebFile {
    /// Creates a reader from a browser `File` object.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when the browser-reported size
    /// is not a finite, non-negative integer representable by `u64`.
    pub fn from_js_file(file: web_sys::File) -> io::Result<Self> {
        let size_bytes = parse_size(file.size())?;
        Ok(Self {
            file_handle: file,
            size_bytes,
            position: 0,
        })
    }

    /// Reads the next bounded chunk into the caller-provided buffer.
    ///
    /// The browser file remains outside Rust-owned storage. A read larger than
    /// [`MAX_READ_BYTES`] is rejected before a `Blob` or stream scratch buffer is
    /// created, so an untrusted file cannot force a large provider allocation.
    /// A first read that covers a file no larger than the same bound uses the
    /// browser `File.arrayBuffer()` operation; later or larger reads use the
    /// bounded object-URL response stream.
    ///
    /// # Errors
    /// Returns an I/O error when the buffer is over the provider bound, cursor
    /// arithmetic overflows, or the browser rejects the bounded object-URL
    /// response stream.
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let requested = validate_buffer_length(buf.len())?;
        if buf.is_empty() {
            return Ok(0);
        }

        let size = self.size_bytes;
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
        let target_len = usize::try_from(length).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser file read length cannot be represented",
            )
        })?;
        let copied = if self.position == 0 && length == size && target_len <= MAX_READ_BYTES {
            read_small_file_array_buffer(&self.file_handle, &mut buf[..target_len]).await?
        } else {
            let end_position = advance_cursor(self.position, length)?;
            let blob = self
                .file_handle
                .slice_with_f64_and_f64(self.position as f64, end_position as f64)
                .map_err(|_| io::Error::other("browser rejected the file slice"))?;
            read_blob_via_object_url(blob, &mut buf[..target_len]).await?
        };
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

    /// Returns the validated browser file size in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size_bytes
    }

    /// Returns the current sequential read cursor.
    #[must_use]
    pub const fn position(&self) -> u64 {
        self.position
    }

    /// Moves the read cursor within the validated browser file size.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when `pos` is beyond the file.
    pub fn seek(&mut self, pos: u64) -> io::Result<()> {
        let size = self.size_bytes;
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

async fn read_small_file_array_buffer(
    file: &web_sys::File,
    buffer: &mut [u8],
) -> io::Result<usize> {
    let value = JsFuture::from(file.array_buffer())
        .await
        .map_err(|_| io::Error::other("browser rejected the bounded whole-file read"))?;
    let array_buffer = js_sys::ArrayBuffer::from(value);
    let bytes = js_sys::Uint8Array::new(&array_buffer);
    let available = usize::try_from(bytes.length()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "browser whole-file result length cannot be represented",
        )
    })?;
    if available != buffer.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "browser whole-file result length disagrees with its declared size",
        ));
    }
    bytes.copy_to(buffer);
    Ok(available)
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

async fn read_blob_via_object_url(blob: web_sys::Blob, buffer: &mut [u8]) -> io::Result<usize> {
    let mut object_url = ObjectUrl::from_blob(&blob)?;
    let result = read_object_url_stream(object_url.as_str(), buffer).await;
    let cleanup = object_url.revoke();
    match (result, cleanup) {
        (Ok(copied), Ok(())) => Ok(copied),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(read_error), Err(cleanup_error)) => Err(io::Error::new(
            read_error.kind(),
            format!("{read_error}; browser object URL cleanup failed: {cleanup_error}"),
        )),
    }
}

struct ObjectUrl {
    value: String,
    revoked: bool,
}

impl ObjectUrl {
    fn from_blob(blob: &web_sys::Blob) -> io::Result<Self> {
        let value = web_sys::Url::create_object_url_with_blob(blob)
            .map_err(|_| io::Error::other("browser rejected the file object URL"))?;
        Ok(Self {
            value,
            revoked: false,
        })
    }

    fn as_str(&self) -> &str {
        &self.value
    }

    fn revoke(&mut self) -> io::Result<()> {
        if self.revoked {
            return Ok(());
        }
        web_sys::Url::revoke_object_url(&self.value)
            .map_err(|_| io::Error::other("browser rejected object URL revocation"))?;
        self.revoked = true;
        Ok(())
    }
}

impl Drop for ObjectUrl {
    fn drop(&mut self) {
        if !self.revoked
            && let Err(error) = web_sys::Url::revoke_object_url(&self.value)
        {
            web_sys::console::error_1(&error);
        }
    }
}

async fn read_object_url_stream(url: &str, buffer: &mut [u8]) -> io::Result<usize> {
    let window = web_sys::window()
        .ok_or_else(|| io::Error::other("browser window is unavailable for file access"))?;
    let response = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|_| io::Error::other("browser rejected the file object URL request"))?
        .dyn_into::<web_sys::Response>()
        .map_err(|_| io::Error::other("browser returned an invalid file object URL response"))?;
    if !response.ok() {
        return Err(io::Error::other(
            "browser file object URL response was not successful",
        ));
    }
    let stream = response
        .body()
        .ok_or_else(|| io::Error::other("browser file object URL response has no body"))?;
    read_stream(stream, buffer).await
}

async fn read_stream(stream: web_sys::ReadableStream, buffer: &mut [u8]) -> io::Result<usize> {
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
