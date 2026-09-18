//! Win32 scan-code LPARAM encoding shared by keyboard-message tests.

use windows::Win32::Foundation::LPARAM;

pub(super) fn key_lparam(scan_code: u8, extended: bool, context: bool) -> LPARAM {
    let mut raw = u64::from(scan_code) << 16;
    if extended {
        raw |= 1 << 24;
    }
    if context {
        raw |= 1 << 29;
    }
    LPARAM(raw as isize)
}
