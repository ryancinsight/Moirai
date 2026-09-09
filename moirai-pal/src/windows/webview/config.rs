//! Validated WebView2 URL and resource bounds.

use std::{io, time::Duration};

/// Maximum retained WebView2 events for one host.
pub const MAX_WEBVIEW_EVENTS: usize = 256;
/// Maximum UTF-8 bytes in one outbound or inbound JSON message.
pub const MAX_WEBVIEW_MESSAGE_BYTES: usize = 64 * 1024;
/// Maximum UTF-16 code units in one JSON message.
pub const MAX_WEBVIEW_MESSAGE_UNITS: usize = 64 * 1024;
/// Maximum UTF-16 code units in one WebView2 URI.
pub const MAX_WEBVIEW_URI_UNITS: usize = 2 * 1024;
/// Maximum finite wait for WebView2 creation or navigation.
pub const MAX_WEBVIEW_WAIT_MILLISECONDS: u32 = 30_000;

/// Validated WebView2 host configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebViewConfig {
    start_uri: String,
    allowed_prefix: String,
    wait: Duration,
}

impl WebViewConfig {
    /// Validates a packaged `file:///` entry page with the default finite wait.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a non-file URI, traversal, malformed percent
    /// escapes or a URI outside the configured UTF-16 bound.
    pub fn new(start_uri: impl AsRef<str>) -> io::Result<Self> {
        Self::with_wait(start_uri, Duration::from_secs(30))
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
        })
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
        validate_file_uri(uri).is_ok() && uri.starts_with(&self.allowed_prefix)
    }
}

pub(super) fn validate_message(bytes: &[u8]) -> io::Result<()> {
    if bytes.len() > MAX_WEBVIEW_MESSAGE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "WebView2 JSON message exceeds the bounded byte limit",
        ));
    }
    if bytes.contains(&0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "WebView2 JSON message contains NUL",
        ));
    }
    Ok(())
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
