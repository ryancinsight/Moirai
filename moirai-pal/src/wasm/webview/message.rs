//! Lossless bounded message extraction from real browser values.

use std::io;

use js_sys::{JSON, JsString};
use wasm_bindgen::{JsCast, JsValue};

use crate::webview_limits::message_utf8_length;

use crate::wasm::browser_error::javascript_error;

pub(super) fn receive(value: JsValue) -> io::Result<String> {
    if let Some(text) = value.dyn_ref::<JsString>() {
        let text = copy_string(text)?;
        JSON::parse(&text).map_err(|error| {
            javascript_error(
                "WebView2 received JSON.parse",
                io::ErrorKind::InvalidData,
                error,
            )
        })?;
        return Ok(text);
    }
    if !value.is_object() || value.is_null() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "WebView2 message must contain a JSON object or JSON string",
        ));
    }
    serialize(&value)
}

pub(super) fn serialize(value: &JsValue) -> io::Result<String> {
    copy_string(&representation(value)?)
}

pub(super) fn validate_serialization(value: &JsValue) -> io::Result<usize> {
    message_utf8_length(representation(value)?.iter())
}

fn representation(value: &JsValue) -> io::Result<JsString> {
    let text = JSON::stringify(value).map_err(|error| {
        javascript_error("WebView2 JSON.stringify", io::ErrorKind::InvalidData, error)
    })?;
    if !<JsString as AsRef<JsValue>>::as_ref(&text).is_string() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "WebView2 JSON.stringify returned no JSON representation",
        ));
    }
    Ok(text)
}

/// Each pass borrows immutable JS UTF16. No JS-to-Rust string conversion
/// occurs until its exact UTF8 allocation and both native bounds are proven.
pub(super) fn copy_string(text: &JsString) -> io::Result<String> {
    let bytes = message_utf8_length(text.iter())?;
    let mut value = String::new();
    value.try_reserve_exact(bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::OutOfMemory,
            format!("WebView2 message reservation: {error}"),
        )
    })?;
    for character in char::decode_utf16(text.iter()) {
        value.push(character.map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("WebView2 message UTF16 changed: {error}"),
            )
        })?);
    }
    Ok(value)
}
