//! Portable resource law for a WebView2 rendered-frame payload.

/// Maximum PNG bytes retained by one WebView2 preview capture.
///
/// The bound matches the maximum RGBA8 presentation payload for one surface
/// (`16,777,216` pixels × four channels). A compressed preview is rejected
/// before allocation when it exceeds the same memory budget.
pub const MAX_WEBVIEW_CAPTURE_BYTES: usize = crate::frame::MAX_FRAME_PIXELS * 4;

#[path = "capture/base64.rs"]
pub(crate) mod base64;
