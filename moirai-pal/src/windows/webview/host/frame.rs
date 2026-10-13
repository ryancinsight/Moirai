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

use super::super::capture::{MAX_WEBVIEW_CAPTURE_BYTES, base64};
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
    let encoded = base64::decode::<MAX_WEBVIEW_CAPTURE_BYTES>(data)?;
    let mut decode_options = png::DecodeOptions::default();
    decode_options.set_skip_ancillary_crc_failures(false);
    let mut decoder = png::Decoder::new_with_options(Cursor::new(encoded), decode_options);
    decoder.set_limits(png::Limits {
        bytes: MAX_WEBVIEW_CAPTURE_BYTES,
    });
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
            r#"{"data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAADdEVYdGsAdgAAAAAAAAAASUVORK5CYII="}"#,
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
