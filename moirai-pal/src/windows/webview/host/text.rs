//! Bounded WebView2 UTF-16 conversion.

use std::io;

use webview2_com::CoTaskMemPWSTR;
use windows::core::{PCWSTR, PWSTR};

use super::super::config::MAX_WEBVIEW_URI_UNITS;
use super::super::config::profile::MAX_PROFILE_PATH_UNITS;
use super::error::callback_error;
use crate::webview_limits::{MAX_WEBVIEW_MESSAGE_UNITS, message_utf8_length};

/// Decodes a WebView2 task-memory string and frees it.
///
/// # Safety
///
/// `uri` is null or points to a NUL-terminated UTF-16 buffer allocated with
/// `CoTaskMemAlloc` that the caller owns and never touches again: the
/// returned value's owner frees it with `CoTaskMemFree`.
pub(super) unsafe fn read_task_mem_uri(uri: PWSTR) -> windows::core::Result<String> {
    let value = CoTaskMemPWSTR::from(uri);
    // SAFETY: the caller's contract supplies a NUL-terminated buffer, owned by
    // `value` until this function returns; the bounded scan stops at that
    // terminator and the slice remains borrowed.
    unsafe { read_bounded_pcwstr::<UriText>(value.as_ref().as_pcwstr()) }
}

/// Decodes a WebView2 JSON message from task memory and frees it.
///
/// # Safety
///
/// `uri` is null or points to a NUL-terminated UTF-16 buffer allocated with
/// `CoTaskMemAlloc` that the caller owns and never touches again: the
/// returned value's owner frees it with `CoTaskMemFree`.
pub(super) unsafe fn read_task_mem_message(uri: PWSTR) -> windows::core::Result<String> {
    let value = CoTaskMemPWSTR::from(uri);
    // SAFETY: the caller's contract supplies a NUL-terminated buffer, owned by
    // `value` until this function returns; the bounded scan stops at that
    // terminator and the slice remains borrowed.
    unsafe { read_bounded_pcwstr::<MessageText>(value.as_ref().as_pcwstr()) }
}

/// Reads the actual SDK user-data folder under the native path bound.
///
/// # Safety
/// The pointer transfers the same CoTaskMem-owned terminated buffer ownership
/// as `read_task_mem_uri`; this function owns and frees it exactly once.
pub(super) unsafe fn read_task_mem_profile(path: PWSTR) -> windows::core::Result<String> {
    let value = CoTaskMemPWSTR::from(path);
    // SAFETY: this getter buffer stays owned until bounded conversion returns.
    unsafe { read_bounded_pcwstr::<ProfileText>(value.as_ref().as_pcwstr()) }
}

unsafe fn read_bounded_pcwstr<P: TextBudget>(uri: &PCWSTR) -> windows::core::Result<String> {
    if uri.0.is_null() {
        return Err(callback_error("WebView2 returned a null UTF-16 pointer"));
    }
    // Read at most budget+1 units, stopping at the actual terminator. The SDK
    // buffer is terminated and owned by the caller; this bounds scanning work
    // as well as the later retained copy without reading beyond that buffer.
    let mut length = 0;
    loop {
        // SAFETY: earlier units are non-terminating, so the SDK's terminated
        // allocation contains this next unit. No pointer escapes the call.
        if unsafe { uri.0.add(length).read() } == 0 {
            break;
        }
        if length == P::MAX_UNITS {
            return Err(callback_error("WebView2 UTF-16 payload exceeds its bound"));
        }
        length += 1;
    }
    // SAFETY: the provider's callback contract supplies at least `length` code
    // units followed by NUL; no pointer escapes this synchronous conversion.
    let units = unsafe { std::slice::from_raw_parts(uri.0, length) };
    // The data is owned by `CoTaskMemPWSTR` in the caller and remains valid
    // until this function returns.
    decode_utf16::<P>(units)
}

/// Reads a borrowed native completion payload before making a retained copy.
///
/// # Safety
/// `value` is null or points to an SDK-owned terminated buffer valid until
/// the callback returns. No pointer or borrow survives this call.
pub(super) unsafe fn read_borrowed_message(value: &PCWSTR) -> windows::core::Result<String> {
    // SAFETY: the callback supplies the documented borrowed terminated buffer.
    unsafe { read_bounded_pcwstr::<MessageText>(value) }
}

pub(super) fn encode_utf16(value: &str, max_units: usize) -> io::Result<Vec<u16>> {
    let units = value.encode_utf16().count();
    if units > max_units {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "WebView2 UTF-16 payload exceeds its bound",
        ));
    }
    let capacity = units.checked_add(1).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "WebView2 UTF-16 length overflow",
        )
    })?;
    let mut encoded = Vec::new();
    encoded.try_reserve_exact(capacity).map_err(|_| {
        io::Error::new(
            io::ErrorKind::OutOfMemory,
            "WebView2 UTF-16 reservation failed",
        )
    })?;
    encoded.extend(value.encode_utf16());
    encoded.push(0);
    Ok(encoded)
}

trait TextBudget {
    const MAX_UNITS: usize;
    fn capacity(units: &[u16]) -> windows::core::Result<usize>;
}

struct UriText;
struct MessageText;
struct ProfileText;

impl TextBudget for UriText {
    const MAX_UNITS: usize = MAX_WEBVIEW_URI_UNITS;

    fn capacity(units: &[u16]) -> windows::core::Result<usize> {
        units
            .len()
            .checked_mul(3)
            .ok_or_else(|| callback_error("WebView2 UTF-16 payload capacity overflow"))
    }
}

impl TextBudget for MessageText {
    const MAX_UNITS: usize = MAX_WEBVIEW_MESSAGE_UNITS;

    fn capacity(units: &[u16]) -> windows::core::Result<usize> {
        message_utf8_length(units.iter().copied()).map_err(|error| {
            windows::core::Error::new(windows::Win32::Foundation::E_ABORT, error.to_string())
        })
    }
}

impl TextBudget for ProfileText {
    const MAX_UNITS: usize = MAX_PROFILE_PATH_UNITS;

    fn capacity(units: &[u16]) -> windows::core::Result<usize> {
        units
            .len()
            .checked_mul(3)
            .ok_or_else(|| callback_error("WebView2 profile text capacity overflow"))
    }
}

fn decode_utf16<P: TextBudget>(units: &[u16]) -> windows::core::Result<String> {
    if units.len() > P::MAX_UNITS {
        return Err(callback_error("WebView2 UTF-16 payload exceeds its bound"));
    }
    let capacity = P::capacity(units)?;
    let mut value = String::new();
    value
        .try_reserve_exact(capacity)
        .map_err(|_| callback_error("WebView2 UTF-16 payload reservation failed"))?;
    for character in char::decode_utf16(units.iter().copied()) {
        value.push(
            character.map_err(|_| callback_error("WebView2 payload contains unmatched UTF-16"))?,
        );
    }
    Ok(value)
}

#[cfg(test)]
mod tests;
