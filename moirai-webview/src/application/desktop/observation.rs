//! Native lifecycle values for host diagnostics; page contents never enter logs.

use objc2_app_kit::{NSWindow, NSWindowButton, NSWindowStyleMask};
use objc2_foundation::NSNotification;

pub(super) fn notification(notification: &NSNotification, event: &'static str) {
    match notification
        .object()
        .and_then(|object| object.downcast::<NSWindow>().ok())
    {
        Some(window) => self::window(&window, event),
        None => tracing::error!(event, "AppKit window notification has no native window"),
    }
}

pub(super) fn window(window: &NSWindow, event: &'static str) {
    let frame = window.frame();
    let button = |kind| {
        window
            .standardWindowButton(kind)
            .map(|button| (button.isHidden(), button.isEnabled(), button.frame()))
    };
    tracing::info!(
        event,
        window_number = window.windowNumber(),
        x = frame.origin.x,
        y = frame.origin.y,
        width = frame.size.width,
        height = frame.size.height,
        minimized = window.isMiniaturized(),
        zoomed = window.isZoomed(),
        fullscreen = window.styleMask().contains(NSWindowStyleMask::FullScreen),
        close = ?button(NSWindowButton::CloseButton),
        minimize = ?button(NSWindowButton::MiniaturizeButton),
        zoom = ?button(NSWindowButton::ZoomButton),
        "native Apple window lifecycle"
    );
}
