//! Renderer-frame synchronization after a visibility transition.

use std::{io, sync::mpsc, time::Duration};

use webview2_com::{
    CallDevToolsProtocolMethodCompletedHandler, Microsoft::Web::WebView2::Win32::ICoreWebView2,
};
use windows::core::HSTRING;

use super::super::pump::wait_for;
use super::error::{callback_error, windows_error};

#[derive(Debug, Eq, PartialEq)]
struct PresentedFrame;

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
    let PresentedFrame = presented_frame(&response)?;
    Ok(())
}

fn presented_frame(response: &str) -> io::Result<PresentedFrame> {
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
    match object.get("data").and_then(serde_json::Value::as_str) {
        Some(data) if !data.is_empty() => Ok(PresentedFrame),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "WebView2 presented-frame response has no image data",
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::{PresentedFrame, presented_frame};

    #[test]
    fn presented_frame_requires_top_level_image_data() {
        assert_eq!(
            presented_frame(r#"{"data":"iVBORw0KGgo="}"#).expect("image data"),
            PresentedFrame
        );

        for response in [
            r#"{"data":""}"#,
            r#"{"error":{"message":"missing data"}}"#,
            r#"{"exceptionDetails":{"text":"data"}}"#,
            r#"{"data":"iVBORw0KGgo=","exceptionDetails":{"text":"failure"}}"#,
            "[]",
            "not JSON",
        ] {
            assert_eq!(
                presented_frame(response)
                    .expect_err("response without image data")
                    .kind(),
                io::ErrorKind::InvalidData
            );
        }
    }
}
