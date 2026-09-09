//! Rust-owned WebView2 events.

/// Event delivered by one WebView2 host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebViewEvent {
    /// A navigation was requested, with the resource policy result.
    NavigationStarting {
        /// Requested URI.
        uri: String,
        /// Whether the request was allowed to continue.
        allowed: bool,
    },
    /// A navigation completed.
    NavigationCompleted {
        /// Completed URI.
        uri: String,
        /// Whether WebView2 reported success.
        success: bool,
    },
    /// A new-window request was denied by policy.
    NewWindowDenied {
        /// Requested URI.
        uri: String,
    },
    /// A bounded JSON message received from the page.
    Message {
        /// Source URI reported by WebView2.
        source: String,
        /// JSON payload.
        json: String,
    },
    /// A message from a disallowed source was rejected.
    MessageRejected {
        /// Source URI reported by WebView2.
        source: String,
    },
}

/// Event returned by a WebView2 host's combined native pump.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebViewHostEvent {
    /// Event translated by the parent Win32 window.
    Window(super::super::window::WindowEvent),
    /// Event produced by the WebView2 controller.
    WebView(WebViewEvent),
}
