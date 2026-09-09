//! Bounded WebView2 UTF-16 conversion.

use std::io;

use webview2_com::CoTaskMemPWSTR;
use windows::{
    Win32::Globalization::lstrlenW,
    core::{PCWSTR, PWSTR},
};

use super::super::config::{MAX_WEBVIEW_MESSAGE_UNITS, MAX_WEBVIEW_URI_UNITS};
use super::error::callback_error;

pub(super) fn read_task_mem_uri(uri: PWSTR) -> windows::core::Result<String> {
    let value = CoTaskMemPWSTR::from(uri);
    // SAFETY: WebView2 provides a NUL-terminated PCWSTR valid for this callback;
    // lstrlenW only reads until that terminator and the slice remains borrowed.
    unsafe { read_bounded_pcwstr(value.as_ref().as_pcwstr(), MAX_WEBVIEW_URI_UNITS) }
}

pub(super) fn read_task_mem_message(uri: PWSTR) -> windows::core::Result<String> {
    let value = CoTaskMemPWSTR::from(uri);
    // SAFETY: WebView2 provides a NUL-terminated PCWSTR valid for this callback;
    // lstrlenW only reads until that terminator and the slice remains borrowed.
    unsafe { read_bounded_pcwstr(value.as_ref().as_pcwstr(), MAX_WEBVIEW_MESSAGE_UNITS) }
}

unsafe fn read_bounded_pcwstr(uri: &PCWSTR, max_units: usize) -> windows::core::Result<String> {
    if uri.0.is_null() {
        return Err(callback_error("WebView2 returned a null UTF-16 pointer"));
    }
    // SAFETY: the caller holds the COM-owned NUL-terminated buffer for this
    // synchronous conversion; lstrlenW does not retain the pointer.
    let length = unsafe { lstrlenW(*uri) };
    if length < 0 {
        return Err(callback_error("WebView2 returned a negative UTF-16 length"));
    }
    let length =
        usize::try_from(length).map_err(|_| callback_error("WebView2 URI length overflow"))?;
    if length > max_units {
        return Err(callback_error("WebView2 UTF-16 payload exceeds its bound"));
    }
    // SAFETY: the provider's callback contract supplies at least `length` code
    // units followed by NUL; no pointer escapes this synchronous conversion.
    let units = unsafe { std::slice::from_raw_parts(uri.0, length) };
    // The data is owned by `CoTaskMemPWSTR` in the caller and remains valid
    // until this function returns.
    decode_utf16(units, max_units)
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

fn decode_utf16(units: &[u16], max_units: usize) -> windows::core::Result<String> {
    if units.len() > max_units {
        return Err(callback_error("WebView2 UTF-16 payload exceeds its bound"));
    }
    let capacity = units
        .len()
        .checked_mul(3)
        .ok_or_else(|| callback_error("WebView2 UTF-16 payload capacity overflow"))?;
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
