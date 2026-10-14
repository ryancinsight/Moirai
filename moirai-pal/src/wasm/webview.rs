//! The actual WebView2 browser bridge; no authored JavaScript forwarding layer.

mod endpoint;
mod listener;
mod message;

use std::{io, time::Duration};

use js_sys::Function;

use crate::webview_limits::{
    MAX_WEBVIEW_MESSAGE_BYTES, MAX_WEBVIEW_WAIT_MILLISECONDS, validate_message,
};

use super::browser_error::javascript_error;
use endpoint::{NativeMessageTarget, callable, property, unavailable};
pub use listener::WebViewMessageListener;

/// A checked handle to this document's actual `window.chrome.webview` methods.
///
/// It is unavailable in an ordinary browser. Native-origin messages are
/// delivered synchronously to one Rust callback; this handle retains no queue.
pub struct WebViewBridge {
    target: NativeMessageTarget,
    post: Function,
}

impl WebViewBridge {
    /// Resolves the installed WebView2 bridge in the current browser realm.
    ///
    /// # Errors
    /// Returns `Unsupported` when the native bridge is absent, `InvalidData`
    /// for an invalid native interface, and contextual JavaScript exceptions.
    pub fn current() -> io::Result<Self> {
        let window = web_sys::window().ok_or_else(|| unavailable("browser Window"))?;
        let chrome = property(window.as_ref(), "chrome")?;
        let webview = property(&chrome, "webview")?;
        let target = NativeMessageTarget::new(webview)?;
        let post = callable(&target.receiver, "postMessage")?;
        Ok(Self { target, post })
    }

    /// The native/browser JSON byte budget, from the provider's resource law.
    #[must_use]
    pub const fn message_byte_limit() -> usize {
        MAX_WEBVIEW_MESSAGE_BYTES
    }

    /// The finite native/browser operation wait, from the shared provider law.
    #[must_use]
    #[inline]
    pub fn operation_wait_limit() -> Duration {
        Duration::from_millis(u64::from(MAX_WEBVIEW_WAIT_MILLISECONDS))
    }

    /// Number of callbacks retained after an unsuccessful browser teardown.
    ///
    /// The count is zero or one for this document. A retained callback blocks
    /// replacement until the browser realm is destroyed.
    #[must_use]
    pub fn retained_cleanup_count() -> usize {
        listener::retained_cleanup_count()
    }

    /// Posts parsed JSON through the native object's `postMessage` method.
    ///
    /// # Errors
    /// Rejects malformed or oversized JSON before posting. Both the input and
    /// the native JSON serialization obey the byte and UTF16 bounds. Exceptions
    /// from parsing, serialization or the actual native call retain context.
    pub fn post_json(&self, json: &str) -> io::Result<()> {
        validate_message(json.as_bytes())?;
        let value = js_sys::JSON::parse(json).map_err(|error| {
            javascript_error("WebView2 JSON.parse", io::ErrorKind::InvalidInput, error)
        })?;
        message::validate_serialization(&value)?;
        self.post
            .call1(&self.target.receiver, &value)
            .map_err(|error| {
                javascript_error("WebView2 postMessage", io::ErrorKind::Other, error)
            })?;
        Ok(())
    }

    /// Attaches the document's single owned message callback.
    ///
    /// # Errors
    /// Returns `WouldBlock` while a listener or failed cleanup guard exists.
    /// Browser registration exceptions retain their JavaScript context.
    pub fn listen<F>(&self, callback: F) -> io::Result<WebViewMessageListener>
    where
        F: FnMut(io::Result<String>) + 'static,
    {
        WebViewMessageListener::attach(self.target.clone(), callback)
    }
}

#[cfg(test)]
mod tests;
