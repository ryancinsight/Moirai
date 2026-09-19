//! Bounded browser text clipboard access.

use super::WebDocument;
use crate::text_validation::text_value;
use js_sys::Reflect;
use std::io;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{Clipboard, Navigator};

/// A browser text clipboard provider owned by one document context.
///
/// The browser remains the authority for permission, user activation and
/// platform clipboard integration. Métis consumers receive bounded UTF-8
/// text only; no browser or native clipboard handle crosses this boundary.
#[derive(Clone)]
pub struct WebClipboard {
    clipboard: Clipboard,
}

pub(super) fn from_document(document: &WebDocument) -> io::Result<WebClipboard> {
    let window = document
        .document
        .default_view()
        .ok_or_else(|| unsupported("Browser document has no Window"))?;
    let navigator: Navigator = window.navigator();
    let value = Reflect::get(navigator.as_ref(), &JsValue::from_str("clipboard"))
        .map_err(|_| unsupported("Browser clipboard API lookup failed"))?;
    let clipboard = value
        .dyn_into::<Clipboard>()
        .map_err(|_| unsupported("Browser clipboard API is unavailable"))?;
    Ok(WebClipboard { clipboard })
}

impl WebClipboard {
    /// Reads UTF-8 text from the browser clipboard.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::Unsupported`] when the browser does not expose
    /// clipboard text, [`io::ErrorKind::InvalidInput`] when the returned text
    /// exceeds the provider bound, or [`io::ErrorKind::Other`] when the
    /// browser rejects the asynchronous read.
    pub async fn read_text(&self) -> io::Result<String> {
        let value = JsFuture::from(self.clipboard.read_text())
            .await
            .map_err(|_| clipboard_error("Browser clipboard read was rejected"))?;
        let text = value
            .as_string()
            .ok_or_else(|| clipboard_error("Browser clipboard returned non-text data"))?;
        text_value(text)
    }

    /// Writes bounded UTF-8 text to the browser clipboard.
    ///
    /// The browser may require a transient user activation and may reject the
    /// operation under its permission policy.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when `text` exceeds the
    /// provider bound, or [`io::ErrorKind::Other`] when the browser rejects
    /// the asynchronous write.
    pub async fn write_text(&self, text: &str) -> io::Result<()> {
        let text = text_value(text.to_owned())?;
        JsFuture::from(self.clipboard.write_text(&text))
            .await
            .map(|_| ())
            .map_err(|_| clipboard_error("Browser clipboard write was rejected"))
    }
}

fn unsupported(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, message)
}

fn clipboard_error(message: &'static str) -> io::Error {
    io::Error::other(message)
}
