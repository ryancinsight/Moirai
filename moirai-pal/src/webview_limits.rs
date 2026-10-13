//! One native/browser WebView resource law, checked before retained copies.

use std::io;

/// Maximum retained native events for one WebView2 host.
#[cfg(any(windows, test))]
pub const MAX_WEBVIEW_EVENTS: usize = 256;
/// Maximum UTF8 bytes retained by one WebView2 JSON message.
pub const MAX_WEBVIEW_MESSAGE_BYTES: usize = 64 * 1024;
/// Maximum UTF16 code units admitted by one WebView2 JSON message.
pub const MAX_WEBVIEW_MESSAGE_UNITS: usize = 64 * 1024;
/// Maximum finite WebView operation wait, including browser permission probes.
pub const MAX_WEBVIEW_WAIT_MILLISECONDS: u32 = 30_000;

/// Admits a borrowed JSON representation before any native or JS copy.
pub(crate) fn validate_message(bytes: &[u8]) -> io::Result<()> {
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

/// Computes the exact UTF8 allocation without retaining a UTF16 copy.
///
/// For a valid scalar `c`, UTF8 uses `c.len_utf8()` bytes. Summing those
/// lengths gives the allocation oracle; UTF16 length is checked first and
/// both inclusive resource bounds hold before a caller reserves storage.
/// Surrogates and raw NUL are rejected instead of altered by a lossy copy.
pub(crate) fn message_utf8_length(units: impl ExactSizeIterator<Item = u16>) -> io::Result<usize> {
    if units.len() > MAX_WEBVIEW_MESSAGE_UNITS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "WebView2 UTF-16 payload exceeds its bound",
        ));
    }
    let mut bytes = 0;
    for character in char::decode_utf16(units) {
        let character = character.map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "WebView2 payload contains unmatched UTF-16",
            )
        })?;
        if character == '\0' {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebView2 JSON message contains NUL",
            ));
        }
        // At most 65,536 admitted code units each contribute four bytes;
        // this sum fits usize on every supported native/WASM target.
        bytes += character.len_utf8();
        if bytes > MAX_WEBVIEW_MESSAGE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebView2 JSON message exceeds the bounded byte limit",
            ));
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests;
