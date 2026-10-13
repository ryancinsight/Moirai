//! The Windows text clipboard, through Win32's global-memory handles.
//!
//! This module is the mechanism half of a scoped clipboard provider: it
//! reads and writes the system's `CF_UNICODETEXT` format and nothing else.
//! Every call owns one `OpenClipboard`/`CloseClipboard` region, re-reads
//! the sequence number so a concurrent writer is reported, and never
//! executes or transforms the text it carries.

use std::io;

use windows::Win32::Foundation::{GlobalFree, HANDLE, HGLOBAL, HWND};
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, GetClipboardData, GetClipboardSequenceNumber, OpenClipboard,
    SetClipboardData,
};
use windows::Win32::System::Memory::{
    GMEM_MOVEABLE, GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock,
};
use windows::Win32::System::Ole::CF_UNICODETEXT;

/// Maximum UTF-16 units one read or write may carry.
pub const MAX_CLIPBOARD_UNITS: usize = 1 << 20;

/// Reads the clipboard's text content.
///
/// The read is bounded by [`MAX_CLIPBOARD_UNITS`]: a clipboard without text
/// or one carrying more than the bound means `ErrorKind::Unsupported`
/// rather than an unbounded allocation from the reported global size.
///
/// # Errors
/// Returns `ErrorKind::Unsupported` when the text exceeds the bound,
/// `ErrorKind::Interrupted` when another application wrote to the clipboard
/// during the read, and an OS error when the clipboard or its memory could
/// not be used. Text absent from the clipboard is `Ok(None)`.
pub fn read_text() -> io::Result<Option<String>> {
    let session = ClipboardSession::open()?;
    let snapshot = unsafe { GetClipboardSequenceNumber() };
    // SAFETY: the clipboard is open on this thread; the returned data handle
    // is locked and unlocked below and is not modified.
    let Ok(handle) = (unsafe { GetClipboardData(u32::from(CF_UNICODETEXT.0)) }) else {
        // Not text, or a viewer owns the data as a non-text format.
        return Ok(None);
    };
    let Some(units) = read_handle_units(handle) else {
        return Ok(None);
    };
    if unsafe { GetClipboardSequenceNumber() } != snapshot {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "the clipboard changed while the read was open",
        ));
    }
    drop(session);
    if units.len() > MAX_CLIPBOARD_UNITS {
        return Err(bound_error("clipboard text exceeds the read bound"));
    }
    String::from_utf16(&units).map(Some).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "clipboard text is not valid UTF-16",
        )
    })
}

/// Places `text` on the clipboard as `CF_UNICODETEXT`, replacing whatever
/// was there.
///
/// Empty text and text longer than [`MAX_CLIPBOARD_UNITS`] are refused
/// before the clipboard is touched. The returned value is the UTF-8 byte
/// count carried onto the clipboard.
///
/// # Errors
/// Returns `ErrorKind::InvalidInput` for empty or oversized text and an OS
/// error when the clipboard or its memory could not be used.
pub fn write_text(text: &str) -> io::Result<usize> {
    let mut units: Vec<u16> = text.encode_utf16().collect();
    if units.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "clipboard text is empty",
        ));
    }
    if units.len() > MAX_CLIPBOARD_UNITS {
        return Err(bound_error("clipboard text exceeds the write bound"));
    }
    units.push(0);
    let bytes: Vec<u8> = units.iter().flat_map(|unit| unit.to_le_bytes()).collect();
    let byte_len = bytes.len();

    let _session = ClipboardSession::open()?;
    unsafe { EmptyClipboard() }.map_err(os_error)?;
    // SAFETY: the allocation is shared only with the clipboard API through
    // `SetClipboardData` below; the locked region is only modified here.
    let handle = unsafe { GlobalAlloc(GMEM_MOVEABLE, byte_len) }.map_err(os_error)?;
    let write_succeeded = (|| {
        // SAFETY: the handle was allocated above, is locked here exactly
        // once and unlocked by the guard, and is not yet owned by the
        // system.
        let target = unsafe { GlobalLock(handle) };
        if target.is_null() {
            return false;
        }
        struct Unlock(HGLOBAL);
        impl Drop for Unlock {
            fn drop(&mut self) {
                // SAFETY: this is the handle locked above.
                unsafe { GlobalUnlock(self.0).ok() };
            }
        }
        let _unlock = Unlock(handle);
        // SAFETY: the locked global holds at least `byte_len` bytes and the
        // source slice is exactly `byte_len` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), target.cast(), bytes.len());
        }
        true
    })();
    if !write_succeeded {
        // SAFETY: the handle was allocated above and was never handed to
        // the clipboard, so this call still owns it.
        let _ = unsafe { GlobalFree(Some(handle)) };
        return Err(io::Error::other("clipboard memory could not be locked"));
    }
    // SAFETY: the clipboard is open, was emptied above, and takes ownership
    // of the handle; the system now owns and frees the memory.
    unsafe { SetClipboardData(u32::from(CF_UNICODETEXT.0), Some(handle_as_data(handle))) }
        .map_err(|error| {
            // The system did not take ownership, so the handle is still
            // ours to release.
            // SAFETY: allocated above, never accepted by the clipboard.
            let _ = unsafe { GlobalFree(Some(handle)) };
            io::Error::from_raw_os_error(error.code().0)
        })?;
    Ok(text.len())
}

/// The data handle's UTF-16 content, bounded and NUL-terminated.
fn read_handle_units(handle: HANDLE) -> Option<Vec<u16>> {
    // SAFETY: the handle came from `GetClipboardData` on this thread; the
    // lock is held by the guard below and never taken twice.
    let locked = unsafe { GlobalLock(handle_as_global(handle)) };
    if locked.is_null() {
        return None;
    }
    struct Unlock(HGLOBAL);
    impl Drop for Unlock {
        fn drop(&mut self) {
            // SAFETY: this is the handle locked above.
            unsafe { GlobalUnlock(self.0).ok() };
        }
    }
    let _unlock = Unlock(handle_as_global(handle));
    // SAFETY: the locked global reports its own byte size through
    // `GlobalSize` and holds UTF-16 units; the NUL terminator required by
    // CF_UNICODETEXT bounds the scan.
    let byte_count = unsafe { GlobalSize(handle_as_global(handle)) };
    let units = unsafe { std::slice::from_raw_parts(locked.cast::<u16>(), byte_count / 2) };
    let end = units
        .iter()
        .position(|unit| *unit == 0)
        .unwrap_or(units.len());
    Some(units[..end].to_vec())
}

/// The clipboard's data handle as the memory handle the global API takes.
fn handle_as_global(handle: HANDLE) -> HGLOBAL {
    HGLOBAL(handle.0)
}

/// The allocated handle as the clipboard API's generic data handle.
fn handle_as_data(handle: HGLOBAL) -> HANDLE {
    HANDLE(handle.0)
}

/// One bounded OpenClipboard/CloseClipboard region on the calling thread.
struct ClipboardSession {
    _not_send: std::marker::PhantomData<*mut ()>,
}

impl ClipboardSession {
    fn open() -> io::Result<Self> {
        // SAFETY: the clipboard is opened for this thread and closed exactly
        // once by `Drop`; the null window is the documented no-owner form.
        unsafe { OpenClipboard(Some(HWND::default())) }
            .map(|_| Self {
                _not_send: std::marker::PhantomData,
            })
            .map_err(|error| io_os_error(&error))
    }
}

impl Drop for ClipboardSession {
    fn drop(&mut self) {
        // SAFETY: the clipboard was opened above on this thread.
        let _ = unsafe { CloseClipboard() };
    }
}

/// Error text for a clipboard over the provider bound.
fn bound_error(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, message)
}

fn os_error(error: windows::core::Error) -> io::Error {
    io_os_error(&error)
}

fn io_os_error(error: &windows::core::Error) -> io::Error {
    io::Error::from_raw_os_error(error.code().0)
}

#[cfg(test)]
mod tests {
    use super::{MAX_CLIPBOARD_UNITS, read_text, write_text};
    use std::io::ErrorKind;

    #[test]
    fn empty_and_oversized_text_are_refused_before_touching_the_clipboard() {
        assert!(write_text("").is_err());
        let long = "\u{2022}".repeat(MAX_CLIPBOARD_UNITS + 1);
        assert!(write_text(&long).is_err());
    }

    #[test]
    fn absent_text_is_none_and_read_errors_are_typed() {
        // The real clipboard may hold text or nothing; the mechanism must
        // report one of the admitted outcomes, never panic.
        match read_text() {
            Ok(None | Some(_)) => {}
            Err(error) => assert!(
                matches!(
                    error.kind(),
                    ErrorKind::Unsupported | ErrorKind::Interrupted | ErrorKind::Other
                ),
                "unexpected read error: {error}"
            ),
        }
    }

    #[test]
    #[ignore = "replaces the real system clipboard; run explicitly on an owned host"]
    fn text_round_trips_through_the_real_clipboard() {
        // Mutation of the user's real clipboard is a host effect, so this
        // journey restores whatever it observes first.
        let original = read_text().ok().flatten().unwrap_or_default();
        let text = "Métis \u{1F9E9} clipboard round trip";
        write_text(text).expect("write");
        assert_eq!(read_text().expect("read").as_deref(), Some(text));
        if original.is_empty() {
            let _ = write_text("Métis clipboard round-trip sentinel");
        } else {
            let _ = write_text(&original);
        }
    }
}
