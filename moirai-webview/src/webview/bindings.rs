//! The shared WebKit class binding missing from objc2-web-kit's iOS surface.
//!
//! Apple's WKWebView has the same selectors on both targets and inherits the
//! native view class. Declaring that ABI once avoids separate WebView logic.

use objc2::{
    MainThreadOnly, extern_class, extern_methods,
    rc::{Allocated, Retained},
    runtime::ProtocolObject,
};
use objc2_foundation::{NSRect, NSURLRequest};
use objc2_web_kit::{WKNavigation, WKNavigationDelegate, WKUIDelegate, WKWebViewConfiguration};

#[cfg(target_os = "macos")]
use objc2_app_kit::NSView as PlatformView;
#[cfg(target_os = "ios")]
use objc2_ui_kit::UIView as PlatformView;

extern_class!(
    // SAFETY: WKWebView subclasses NSView on macOS and UIView on iOS. Its
    // operations are main-thread-only and objc2 owns retain/release.
    #[unsafe(super(PlatformView))]
    #[name = "WKWebView"]
    #[thread_kind = MainThreadOnly]
    pub(crate) struct WebView;
);

impl WebView {
    extern_methods!(
        #[unsafe(method(initWithFrame:configuration:))]
        #[unsafe(method_family = init)]
        pub(crate) fn init(
            this: Allocated<Self>,
            frame: NSRect,
            configuration: &WKWebViewConfiguration,
        ) -> Retained<Self>;

        // Protocol erasure is required by the system's Objective-C boundary.
        #[unsafe(method(setNavigationDelegate:))]
        pub(crate) fn set_delegate(
            &self,
            delegate: Option<&ProtocolObject<dyn WKNavigationDelegate>>,
        );

        // Protocol erasure is required by the system's Objective-C boundary.
        #[unsafe(method(setUIDelegate:))]
        pub(crate) fn set_ui_delegate(&self, delegate: Option<&ProtocolObject<dyn WKUIDelegate>>);

        #[unsafe(method(loadRequest:))]
        pub(crate) fn load(&self, request: &NSURLRequest) -> Option<Retained<WKNavigation>>;
    );
}

#[link(name = "WebKit", kind = "framework")]
unsafe extern "C" {}
