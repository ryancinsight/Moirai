//! Rust-owned WebView2 events.

/// A WebView2 capability request observed by the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WebViewPermission {
    /// Microphone capture.
    Microphone,
    /// Camera capture.
    Camera,
    /// Geolocation.
    Geolocation,
    /// Browser notifications.
    Notifications,
    /// Ambient or other sensors.
    OtherSensors,
    /// Clipboard read access.
    ClipboardRead,
    /// Multiple automatic downloads.
    MultipleAutomaticDownloads,
    /// File read/write access.
    FileReadWrite,
    /// Media autoplay.
    Autoplay,
    /// Local font access.
    LocalFonts,
    /// MIDI system-exclusive messages.
    MidiSystemExclusiveMessages,
    /// Window-management access.
    WindowManagement,
    /// A permission kind introduced by a newer WebView2 runtime.
    Unknown(i32),
}

impl WebViewPermission {
    pub(super) const fn from_raw(kind: i32) -> Self {
        match kind {
            1 => Self::Microphone,
            2 => Self::Camera,
            3 => Self::Geolocation,
            4 => Self::Notifications,
            5 => Self::OtherSensors,
            6 => Self::ClipboardRead,
            7 => Self::MultipleAutomaticDownloads,
            8 => Self::FileReadWrite,
            9 => Self::Autoplay,
            10 => Self::LocalFonts,
            11 => Self::MidiSystemExclusiveMessages,
            12 => Self::WindowManagement,
            value => Self::Unknown(value),
        }
    }
}

/// Event delivered by one WebView2 host.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum WebViewEvent {
    /// A capability request was synchronously denied by the host policy.
    PermissionDenied {
        /// URI that requested the capability.
        uri: String,
        /// Requested WebView2 capability.
        permission: WebViewPermission,
        /// Whether WebView2 marked the request as user initiated.
        user_initiated: bool,
    },
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
