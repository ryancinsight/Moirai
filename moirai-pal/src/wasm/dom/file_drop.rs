//! Browser file-drop metadata with bounded, validated ownership.

use super::super::WebFile;
use super::WebEvent;
use crate::drop_validation::{MAX_FILE_COUNT, file_name, media_type, parse_size};
use std::io;
use wasm_bindgen::JsCast;
use web_sys::{DragEvent, File, FileList, MouseEvent};

/// Metadata for one file supplied by a browser drop event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DroppedFile {
    name: String,
    media_type: String,
    size_bytes: u64,
}

impl DroppedFile {
    fn from_browser_file(file: &File) -> io::Result<Self> {
        let name = file_name(file.name())?;
        let media_type = media_type(file.type_())?;
        let size_bytes = parse_size(file.size())?;
        Ok(Self {
            name,
            media_type,
            size_bytes,
        })
    }

    /// Returns the browser-provided file name as display metadata.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the browser-provided media type, which may be empty.
    #[must_use]
    pub fn media_type(&self) -> &str {
        &self.media_type
    }

    /// Returns the browser-provided file size in bytes.
    #[must_use]
    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }
}

/// Owns one browser-selected file and its validated display metadata.
pub struct DroppedFileAccess {
    metadata: DroppedFile,
    reader: WebFile,
}

impl DroppedFileAccess {
    /// Returns the validated display metadata for this file.
    #[must_use]
    pub fn metadata(&self) -> &DroppedFile {
        &self.metadata
    }

    /// Returns the browser-provided file name as display metadata.
    #[must_use]
    pub fn name(&self) -> &str {
        self.metadata.name()
    }

    /// Returns the browser-provided media type, which may be empty.
    #[must_use]
    pub fn media_type(&self) -> &str {
        self.metadata.media_type()
    }

    /// Returns the browser-provided file size in bytes.
    #[must_use]
    pub const fn size_bytes(&self) -> u64 {
        self.metadata.size_bytes()
    }

    /// Reads the next bounded chunk into a caller-provided buffer.
    ///
    /// The browser `File` remains owned by this entry and is never converted
    /// into a filesystem path. Dropping the returned future cancels the
    /// provider's `FileReader` callbacks.
    pub async fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.reader.read(buffer).await
    }

    /// Returns the current read cursor in bytes.
    #[must_use]
    pub const fn position(&self) -> u64 {
        self.reader.position()
    }

    /// Seeks within the validated browser file size.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when `position` is beyond the
    /// selected file.
    pub fn seek(&mut self, position: u64) -> io::Result<()> {
        self.reader.seek(position)
    }
}

/// Owns the bounded file entries captured by one browser drop event.
pub struct DropFiles {
    client_x: i32,
    client_y: i32,
    files: Box<[DroppedFileAccess]>,
}

impl DropFiles {
    /// Returns the viewport-relative horizontal drop coordinate in CSS pixels.
    #[must_use]
    pub const fn client_x(&self) -> i32 {
        self.client_x
    }

    /// Returns the viewport-relative vertical drop coordinate in CSS pixels.
    #[must_use]
    pub const fn client_y(&self) -> i32 {
        self.client_y
    }

    /// Returns the owned file entries captured by the event.
    #[must_use]
    pub fn files(&self) -> &[DroppedFileAccess] {
        &self.files
    }

    /// Returns mutable access to the owned file entries for bounded reads.
    pub fn files_mut(&mut self) -> &mut [DroppedFileAccess] {
        &mut self.files
    }
}

/// Validated metadata for one browser file-drop event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DropMetadata {
    client_x: i32,
    client_y: i32,
    files: Box<[DroppedFile]>,
}

impl DropMetadata {
    /// Returns the viewport-relative horizontal drop coordinate in CSS pixels.
    #[must_use]
    pub const fn client_x(&self) -> i32 {
        self.client_x
    }

    /// Returns the viewport-relative vertical drop coordinate in CSS pixels.
    #[must_use]
    pub const fn client_y(&self) -> i32 {
        self.client_y
    }

    /// Returns the validated files supplied by the event.
    #[must_use]
    pub fn files(&self) -> &[DroppedFile] {
        &self.files
    }
}

impl WebEvent {
    /// Reads bounded file metadata from a browser drag event.
    ///
    /// Non-drag events return `Ok(None)`. Drag events return an owned metadata
    /// snapshot. Browser-provided counts, names, media types and sizes are
    /// validated before any metadata is returned to the application.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when the browser omits the
    /// transfer or file list, an indexed file is absent, or metadata violates
    /// the provider bounds.
    pub fn drop_metadata(&self) -> io::Result<Option<DropMetadata>> {
        let Some((client_x, client_y, files)) = drop_context(&self.event)? else {
            return Ok(None);
        };
        let entries = collect_files(&files)?;
        Ok(Some(DropMetadata {
            client_x,
            client_y,
            files: entries,
        }))
    }

    /// Captures bounded browser file entries with asynchronous byte access.
    ///
    /// Non-drag events return `Ok(None)`. The returned entries own browser
    /// `File` objects and expose only validated metadata plus bounded reads;
    /// they never expose browser bindings or filesystem paths to consumers.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when the browser omits the
    /// transfer or file list, an indexed file is absent, or metadata violates
    /// the provider bounds.
    pub fn drop_files(&self) -> io::Result<Option<DropFiles>> {
        let Some((client_x, client_y, files)) = drop_context(&self.event)? else {
            return Ok(None);
        };
        let entries = collect_file_access(&files)?;
        Ok(Some(DropFiles {
            client_x,
            client_y,
            files: entries,
        }))
    }
}

fn drop_context(event: &web_sys::Event) -> io::Result<Option<(i32, i32, FileList)>> {
    let Some(drag) = event.dyn_ref::<DragEvent>() else {
        return Ok(None);
    };
    let mouse = event.dyn_ref::<MouseEvent>().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Drag event has no mouse coordinates",
        )
    })?;
    let transfer = drag.data_transfer().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Drag event has no data transfer",
        )
    })?;
    let files = transfer.files().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "Drag event has no file list")
    })?;
    Ok(Some((mouse.client_x(), mouse.client_y(), files)))
}

fn collect_files(files: &FileList) -> io::Result<Box<[DroppedFile]>> {
    let length = bounded_file_count(files)?;
    let capacity = usize::try_from(length).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Browser file drop count cannot be represented",
        )
    })?;
    let mut entries = Vec::with_capacity(capacity);
    for index in 0..length {
        let file = files.item(index).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Browser file list changed during metadata capture",
            )
        })?;
        entries.push(DroppedFile::from_browser_file(&file)?);
    }
    Ok(entries.into_boxed_slice())
}

fn collect_file_access(files: &FileList) -> io::Result<Box<[DroppedFileAccess]>> {
    let length = bounded_file_count(files)?;
    let capacity = usize::try_from(length).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Browser file drop count cannot be represented",
        )
    })?;
    let mut entries = Vec::with_capacity(capacity);
    for index in 0..length {
        let file = files.item(index).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Browser file list changed during access capture",
            )
        })?;
        let metadata = DroppedFile::from_browser_file(&file)?;
        entries.push(DroppedFileAccess {
            metadata,
            reader: WebFile::from_js_file(file),
        });
    }
    Ok(entries.into_boxed_slice())
}

fn bounded_file_count(files: &FileList) -> io::Result<u32> {
    let length = files.length();
    if length > MAX_FILE_COUNT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Browser file drop exceeds the bounded file count",
        ));
    }
    Ok(length)
}
