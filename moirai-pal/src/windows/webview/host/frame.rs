//! Renderer-frame synchronization after a visibility transition.

use std::{
    io::{self, Cursor},
    sync::mpsc,
    time::Duration,
};

use webview2_com::{
    CallDevToolsProtocolMethodCompletedHandler, Microsoft::Web::WebView2::Win32::ICoreWebView2,
};
use windows::core::HSTRING;

use super::super::config::MAX_WEBVIEW_CAPTURE_BYTES;
use super::super::pump::wait_for;
use super::error::{callback_error, windows_error};

pub(super) fn wait_for_presented_frame(
    webview: &ICoreWebView2,
    timeout: Duration,
) -> io::Result<()> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let handler = CallDevToolsProtocolMethodCompletedHandler::create(Box::new(
        move |error_code, response| {
            sender
                .send(error_code.map(|()| response))
                .map_err(|_| callback_error("WebView2 frame waiter was dropped"))
        },
    ));
    let method = HSTRING::from("Page.captureScreenshot");
    let parameters = HSTRING::from(r#"{"format":"png","fromSurface":true}"#);
    // SAFETY: inputs outlive the call; WebView2 retains the handler.
    unsafe { webview.CallDevToolsProtocolMethod(&method, &parameters, &handler) }
        .map_err(windows_error)?;
    let response = wait_for(receiver, timeout)?;
    validate_presented_frame(&response)?;
    Ok(())
}

fn validate_presented_frame(response: &str) -> io::Result<()> {
    let response: serde_json::Value = serde_json::from_str(response)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let object = response.as_object().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "WebView2 presented-frame response is not an object",
        )
    })?;
    if object.contains_key("error") || object.contains_key("exceptionDetails") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "WebView2 presented-frame response reports an error",
        ));
    }
    let data = object
        .get("data")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid_frame("WebView2 presented-frame response has no image data"))?;
    let encoded = decode_base64(data)?;
    let mut decoder = png::Decoder::new_with_limits(
        Cursor::new(encoded),
        png::Limits {
            bytes: MAX_WEBVIEW_CAPTURE_BYTES,
        },
    );
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder
        .read_info()
        .map_err(|error| invalid_frame(format!("WebView2 presented-frame PNG header: {error}")))?;
    let size = reader.output_buffer_size().ok_or_else(|| {
        invalid_frame("WebView2 presented-frame PNG dimensions are not representable")
    })?;
    if size > MAX_WEBVIEW_CAPTURE_BYTES {
        return Err(invalid_frame(
            "WebView2 presented-frame PNG exceeds the bounded image budget",
        ));
    }
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(size)
        .map_err(|_| io::Error::new(io::ErrorKind::OutOfMemory, "presented-frame allocation"))?;
    pixels.resize(size, 0);
    let output = reader
        .next_frame(&mut pixels)
        .map_err(|error| invalid_frame(format!("WebView2 presented-frame PNG data: {error}")))?;
    if output.buffer_size() == 0 {
        return Err(invalid_frame(
            "WebView2 presented-frame PNG has no decoded pixels",
        ));
    }
    reader
        .finish()
        .map_err(|error| invalid_frame(format!("WebView2 presented-frame PNG trailer: {error}")))?;
    Ok(())
}

fn decode_base64(data: &str) -> io::Result<Vec<u8>> {
    if data.is_empty() || !data.len().is_multiple_of(4) {
        return Err(invalid_frame("WebView2 presented-frame data is not base64"));
    }
    let capacity = data
        .len()
        .checked_mul(3)
        .and_then(|length| length.checked_div(4))
        .ok_or_else(|| invalid_frame("WebView2 presented-frame data is too large"))?;
    if capacity > MAX_WEBVIEW_CAPTURE_BYTES {
        return Err(invalid_frame(
            "WebView2 presented-frame data exceeds the bounded image budget",
        ));
    }
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(capacity)
        .map_err(|_| io::Error::new(io::ErrorKind::OutOfMemory, "presented-frame allocation"))?;
    for (index, chunk) in data.as_bytes().chunks_exact(4).enumerate() {
        let last = index == data.len() / 4 - 1;
        let first = base64_value(chunk[0]).ok_or_else(|| invalid_frame("invalid base64"))?;
        let second = base64_value(chunk[1]).ok_or_else(|| invalid_frame("invalid base64"))?;
        let third = match chunk[2] {
            b'=' => None,
            value => Some(base64_value(value).ok_or_else(|| invalid_frame("invalid base64"))?),
        };
        let fourth = match chunk[3] {
            b'=' => None,
            value => Some(base64_value(value).ok_or_else(|| invalid_frame("invalid base64"))?),
        };
        if (third.is_none() && fourth.is_some()) || (!last && (third.is_none() || fourth.is_none()))
        {
            return Err(invalid_frame("invalid base64 padding"));
        }
        let third = third.unwrap_or(0);
        let fourth = fourth.unwrap_or(0);
        if (chunk[2] == b'=' && second & 0x0f != 0) || (chunk[3] == b'=' && third & 0x03 != 0) {
            return Err(invalid_frame("nonzero base64 padding bits"));
        }
        decoded.push(first << 2 | second >> 4);
        if chunk[2] != b'=' {
            decoded.push(second << 4 | third >> 2);
        }
        if chunk[3] != b'=' {
            decoded.push(third << 6 | fourth);
        }
    }
    Ok(decoded)
}

fn base64_value(value: u8) -> Option<u8> {
    match value {
        b'A'..=b'Z' => Some(value - b'A'),
        b'a'..=b'z' => Some(value - b'a' + 26),
        b'0'..=b'9' => Some(value - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn invalid_frame(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::validate_presented_frame;

    const VALID_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

    #[test]
    fn presented_frame_requires_complete_decodable_image() {
        validate_presented_frame(&format!(r#"{{"data":"{VALID_PNG}"}}"#))
            .expect("complete PNG image");

        for response in [
            r#"{"data":""}"#,
            r#"{"data":"not-base64"}"#,
            r#"{"data":"iVBORw0KGgo="}"#,
            r#"{"data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAA"}"#,
            r#"{"data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYA=="}"#,
            r#"{"data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYIM="}"#,
            r#"{"error":{"message":"missing data"}}"#,
            r#"{"exceptionDetails":{"text":"data"}}"#,
            r#"{"data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=","exceptionDetails":{"text":"failure"}}"#,
            "[]",
            "not JSON",
        ] {
            assert_eq!(
                validate_presented_frame(response)
                    .expect_err("response without image data")
                    .kind(),
                io::ErrorKind::InvalidData
            );
        }
    }
}
