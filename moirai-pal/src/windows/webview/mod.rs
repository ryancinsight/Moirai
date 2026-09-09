//! Thread-owned WebView2 hosting with bounded navigation and message policy.

mod config;
mod event;
mod host;
mod pump;
mod state;

pub use config::{
    MAX_WEBVIEW_EVENTS, MAX_WEBVIEW_MESSAGE_BYTES, MAX_WEBVIEW_MESSAGE_UNITS,
    MAX_WEBVIEW_URI_UNITS, MAX_WEBVIEW_WAIT_MILLISECONDS, WebViewConfig,
};
pub use event::{WebViewEvent, WebViewHostEvent};
pub use host::WebViewHost;

#[cfg(test)]
mod tests;
