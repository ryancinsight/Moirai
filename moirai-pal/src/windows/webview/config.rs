//! Validated WebView2 URL and resource bounds.

use std::{
    io,
    path::{Path, PathBuf},
    time::Duration,
};

use super::folder::FolderMapping;
use crate::webview_limits::MAX_WEBVIEW_WAIT_MILLISECONDS;

pub(super) mod profile;

/// Maximum UTF-16 code units in one WebView2 URI.
pub const MAX_WEBVIEW_URI_UNITS: usize = 2 * 1024;
/// Validated WebView2 host configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebViewConfig {
    start_uri: String,
    allowed_prefix: String,
    wait: Duration,
    /// `None` serves a packaged `file:///` directory.
    folder: Option<FolderMapping>,
    user_data_folder: Option<PathBuf>,
}

impl WebViewConfig {
    /// Validates a packaged `file:///` entry page with the default finite wait.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a non-file URI, traversal, malformed percent
    /// escapes or a URI outside the configured UTF-16 bound.
    pub fn new(start_uri: impl AsRef<str>) -> io::Result<Self> {
        Self::with_wait(
            start_uri,
            Duration::from_millis(u64::from(MAX_WEBVIEW_WAIT_MILLISECONDS)),
        )
    }

    /// Validates a packaged entry page and finite WebView2 operation wait.
    ///
    /// # Errors
    /// Returns `InvalidInput` when the URI or wait exceeds the provider's
    /// resource bounds.
    pub fn with_wait(start_uri: impl AsRef<str>, wait: Duration) -> io::Result<Self> {
        validate_wait(wait)?;
        let start_uri = start_uri.as_ref();
        validate_file_uri(start_uri)?;
        let slash = start_uri.rfind('/').ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebView2 entry URI has no directory",
            )
        })?;
        let allowed_prefix = slash
            .checked_add(1)
            .and_then(|end| start_uri.get(..end))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "WebView2 entry URI is not UTF-8",
                )
            })?;
        let mut owned_uri = String::new();
        owned_uri
            .try_reserve_exact(start_uri.len())
            .map_err(|_| allocation_error())?;
        owned_uri.push_str(start_uri);
        let mut owned_prefix = String::new();
        owned_prefix
            .try_reserve_exact(allowed_prefix.len())
            .map_err(|_| allocation_error())?;
        owned_prefix.push_str(allowed_prefix);
        Ok(Self {
            start_uri: owned_uri,
            allowed_prefix: owned_prefix,
            wait,
            folder: None,
            user_data_folder: None,
        })
    }

    /// Serves `folder` as `https://{host}/` and opens `entry` within it.
    ///
    /// A `file:///` page cannot load ES modules or stream-compile
    /// WebAssembly; a mapped host is a secure origin, so a page built for a
    /// web server runs unchanged. `host` must be a lowercase name under
    /// `.example`, `.invalid`, `.localhost` or `.test` (RFC 2606, RFC 6761),
    /// so the mapping cannot shadow a real site; navigation is confined to
    /// it, and other origins cannot read its resources.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a host outside the reserved domains, a
    /// relative or non-directory folder, an entry with traversal, query or
    /// fragment syntax, or a wait beyond the provider bound, and the
    /// filesystem error when `folder` cannot be inspected.
    pub fn folder(
        host: &str,
        folder: impl AsRef<Path>,
        entry: &str,
        wait: Duration,
    ) -> io::Result<Self> {
        validate_wait(wait)?;
        let folder = FolderMapping::new(host, folder.as_ref())?;
        validate_mapped_path(entry)?;
        let allowed_prefix = format!("https://{host}/");
        let start_uri = format!("{allowed_prefix}{entry}");
        if start_uri.encode_utf16().count() > MAX_WEBVIEW_URI_UNITS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebView2 URI exceeds the bounded UTF-16 limit",
            ));
        }
        Ok(Self {
            start_uri,
            allowed_prefix,
            wait,
            folder: Some(folder),
            user_data_folder: None,
        })
    }

    /// Uses an existing empty local directory as this host's fresh profile.
    ///
    /// The caller must create and exclusively own this directory until the
    /// controller and environment close. Shared profiles cannot provide a
    /// per-request permission evidence contract. The provider does not remove
    /// the caller's data. Omission preserves WebView2's default profile.
    ///
    /// # Errors
    /// Rejects relative, remote, reparse, non-directory or non-empty paths and
    /// a profile overlapping mapped package resources. The host revalidates
    /// the directory before calling COM to detect intervening filesystem changes.
    pub fn with_user_data_folder(mut self, folder: impl AsRef<Path>) -> io::Result<Self> {
        self.user_data_folder = Some(profile::validate(folder.as_ref())?);
        self.validate_directories()?;
        Ok(self)
    }

    pub(super) fn validate_directories(&self) -> io::Result<()> {
        if let Some(mapping) = self.folder_mapping() {
            FolderMapping::new(mapping.host(), mapping.folder())?;
        }
        if let Some(profile) = self.user_data_folder.as_deref() {
            profile::validate(profile)?;
            if let Some(mapping) = self.folder_mapping() {
                profile::reject_overlap(profile, mapping.folder())?;
            }
        }
        Ok(())
    }

    pub(super) fn user_data_folder(&self) -> Option<&Path> {
        self.user_data_folder.as_deref()
    }

    /// Returns the validated entry URI.
    #[must_use]
    pub fn start_uri(&self) -> &str {
        &self.start_uri
    }

    /// Returns the finite operation wait.
    #[must_use]
    pub const fn wait(&self) -> Duration {
        self.wait
    }

    pub(super) fn allows(&self, uri: &str) -> bool {
        match &self.folder {
            None => validate_file_uri(uri).is_ok() && uri.starts_with(&self.allowed_prefix),
            Some(_) => uri
                .strip_prefix(&self.allowed_prefix)
                .is_some_and(|path| validate_mapped_path(path).is_ok()),
        }
    }

    /// The host-to-folder mapping the host installs before navigating.
    pub(super) const fn folder_mapping(&self) -> Option<&FolderMapping> {
        self.folder.as_ref()
    }
}

fn validate_wait(wait: Duration) -> io::Result<()> {
    let milliseconds = u32::try_from(wait.as_millis()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "WebView2 wait exceeds the 30 second bound",
        )
    })?;
    if milliseconds > MAX_WEBVIEW_WAIT_MILLISECONDS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "WebView2 wait exceeds the 30 second bound",
        ));
    }
    Ok(())
}

fn validate_file_uri(uri: &str) -> io::Result<()> {
    if uri.is_empty()
        || uri
            .chars()
            .any(|character| matches!(character, '\0' | '\\' | '?' | '#'))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "WebView2 URI must be a non-empty NUL-free file URI",
        ));
    }
    if uri.encode_utf16().count() > MAX_WEBVIEW_URI_UNITS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "WebView2 URI exceeds the bounded UTF-16 limit",
        ));
    }
    let Some(rest) = uri
        .strip_prefix("file:///")
        .or_else(|| uri.strip_prefix("FILE:///"))
    else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "WebView2 navigation is restricted to file:/// resources",
        ));
    };
    if rest.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "WebView2 file URI has an empty path",
        ));
    }
    validate_percent_escapes(uri)?;
    for segment in rest.split('/') {
        if segment == "." || segment == ".." {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebView2 file URI contains a traversal segment",
            ));
        }
    }
    Ok(())
}

/// A path below a mapped host: non-empty segments, no traversal, and no
/// query, fragment, backslash or NUL.
fn validate_mapped_path(path: &str) -> io::Result<()> {
    if path.is_empty()
        || path
            .chars()
            .any(|character| matches!(character, '\0' | '\\' | '?' | '#'))
        || path
            .split('/')
            .any(|segment| segment.is_empty() || segment == "." || segment == "..")
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "WebView2 mapped path must be relative segments without traversal or query",
        ));
    }
    validate_percent_escapes(path)
}

fn validate_percent_escapes(uri: &str) -> io::Result<()> {
    let bytes = uri.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            index += 1;
            continue;
        }
        let pair = bytes.get(index + 1..index + 3).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebView2 URI contains an incomplete percent escape",
            )
        })?;
        let high = hex(pair[0]).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebView2 URI contains a non-hex percent escape",
            )
        })?;
        let low = hex(pair[1]).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebView2 URI contains a non-hex percent escape",
            )
        })?;
        let value = (high << 4) | low;
        if matches!(value, b'.' | b'/' | b'\\') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebView2 URI contains an encoded path separator",
            ));
        }
        index += 3;
    }
    Ok(())
}

fn hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn allocation_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::OutOfMemory,
        "WebView2 configuration reservation failed",
    )
}
