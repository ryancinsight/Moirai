use crate::{ResourceRoot, webview};
use objc2::{MainThreadOnly, define_class, msg_send, rc::Retained, runtime::ProtocolObject};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSApplicationDelegate, NSBackingStoreType,
    NSWindow, NSWindowDelegate, NSWindowStyleMask,
};
use objc2_foundation::{
    NSNotification, NSObject, NSObjectProtocol, NSPoint, NSRect, NSSize, NSString,
};
use std::io;

mod menu;
mod observation;

define_class!(
    // SAFETY: NSObject imposes no subclassing contract; only the main thread
    // constructs or calls this application delegate.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    struct Delegate;
    // SAFETY: No additional NSObjectProtocol invariants.
    unsafe impl NSObjectProtocol for Delegate {}
    // SAFETY: This callback has NSApplicationDelegate's exact signature.
    unsafe impl NSApplicationDelegate for Delegate {
        #[unsafe(method(applicationShouldTerminateAfterLastWindowClosed:))]
        fn close_after_last_window(&self, _app: &NSApplication) -> bool {
            true
        }
    }
    // SAFETY: Each callback has NSWindowDelegate's exact notification ABI.
    unsafe impl NSWindowDelegate for Delegate {
        #[unsafe(method(windowDidResize:))]
        fn resized(&self, notification: &NSNotification) {
            observation::notification(notification, "resized");
        }

        #[unsafe(method(windowDidMiniaturize:))]
        fn minimized(&self, notification: &NSNotification) {
            observation::notification(notification, "minimized");
        }

        #[unsafe(method(windowDidDeminiaturize:))]
        fn restored(&self, notification: &NSNotification) {
            observation::notification(notification, "restored");
        }

        #[unsafe(method(windowDidEnterFullScreen:))]
        fn entered_fullscreen(&self, notification: &NSNotification) {
            observation::notification(notification, "entered_fullscreen");
        }

        #[unsafe(method(windowDidExitFullScreen:))]
        fn exited_fullscreen(&self, notification: &NSNotification) {
            observation::notification(notification, "exited_fullscreen");
        }

        #[unsafe(method(windowWillClose:))]
        fn closing(&self, notification: &NSNotification) {
            observation::notification(notification, "closing");
        }
    }
);

pub(super) fn run(
    resources: ResourceRoot,
    title: &str,
    mtm: objc2::MainThreadMarker,
) -> io::Result<()> {
    let content = webview::Content::new(resources, mtm);
    let rect = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(800.0, 600.0));
    let view = webview::create(&content, rect)?;
    let app = NSApplication::sharedApplication(mtm);
    // SAFETY: The delegate is a valid initialized NSObject and is retained
    // through app.run. The window is main-thread-only and uses AppKit enums.
    let delegate: Retained<Delegate> = unsafe { msg_send![Delegate::alloc(mtm), init] };
    app.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    menu::install(&app, title, mtm);
    // SAFETY: The owned NSWindow disables auto-release on close below.
    let window = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            NSWindow::alloc(mtm),
            rect,
            NSWindowStyleMask::Titled
                | NSWindowStyleMask::Closable
                | NSWindowStyleMask::Miniaturizable
                | NSWindowStyleMask::Resizable,
            NSBackingStoreType::Buffered,
            false,
        )
    };
    // SAFETY: Retained owns release; AppKit must not release the window again.
    unsafe { window.setReleasedWhenClosed(false) };
    window.setTitle(&NSString::from_str(title));
    window.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    window.setContentView(Some(&view));
    window.center();
    window.makeKeyAndOrderFront(None);
    observation::window(&window, "created");
    app.run();
    window.setDelegate(None);
    app.setWindowsMenu(None);
    app.setMainMenu(None);
    app.setDelegate(None);
    Ok(())
}
