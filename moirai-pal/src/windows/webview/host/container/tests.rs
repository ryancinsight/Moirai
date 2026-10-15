//! Real resize and parent-cascade invalidation of a retained native owner.

use crate::windows::{
    webview::{
        WebViewHost,
        tests::{TestPackage, startup::tracing},
    },
    window::{NativeWindow, WindowConfig, WindowVisibility},
};
use std::io;
use windows::Win32::{Foundation::RECT, UI::WindowsAndMessaging::GetClientRect};

#[test]
#[ignore = "requires an installed WebView2 runtime and real native HWND lifecycle"]
fn installed_runtime_owns_private_parent_and_rejects_destroyed_handle_use() {
    let (_trace, _subscriber) = tracing();
    let _test = tracing::info_span!("native_controller_container_lifetime").entered();
    let (package, mut host) = tracing::info_span!("arrange").in_scope(|| {
        let package = TestPackage::create_with_script(br#"<!doctype html><p id="host-status">private parent</p><output id="result">native lifecycle</output>"#);
        let config = WindowConfig::with_visibility("Moirai private controller lifecycle",480,320,WindowVisibility::Visible).expect("window configuration");
        let window = NativeWindow::new(&config).expect("real native window");
        let host = WebViewHost::new(window,package.fresh_config()).expect("installed WebView2");
        (package,host)
    });
    let alive = host
        .container
        .as_ref()
        .expect("owned parent")
        .liveness_probe();
    assert!(alive(), "WM_NCCREATE establishes the native owner");
    tracing::info_span!("act").in_scope(|| host.resize(400, 300).expect("real view resize"));
    tracing::info_span!("assert").in_scope(|| {
        let mut native = RECT::default();
        let container = host.container.as_ref().expect("owned parent");
        unsafe { GetClientRect(container.hwnd, &mut native) }.expect("native client bounds");
        let mut controller = RECT::default();
        unsafe {
            host.controller
                .as_ref()
                .expect("controller")
                .Bounds(&mut controller)
        }
        .expect("actual controller bounds");
        assert_eq!(
            (native.left, native.top, native.right, native.bottom),
            (0, 0, 400, 300)
        );
        assert_eq!(
            (
                controller.left,
                controller.top,
                controller.right,
                controller.bottom
            ),
            (0, 0, 400, 300)
        );
    });
    // The real root destroys its child natively. Keep the Rust owner to test
    // its callback witness without interpreting IsWindow(old numeric HWND).
    let mut retained = host.container.take().expect("private owner transfer");
    retained.deactivate();
    tracing::info_span!("act").in_scope(|| host.close().expect("real controller/root close"));
    tracing::info_span!("assert").in_scope(|| {
        assert!(
            !alive(),
            "WM_NCDESTROY invalidates retained native ownership"
        );
        let error = retained
            .resize(512, 256)
            .expect_err("destroyed native container rejects resize");
        assert_eq!(error.kind(), io::ErrorKind::NotConnected);
        retained
            .close()
            .expect("idempotent close after native cascade");
        package.cleanup().expect("owned profile cleanup");
    });
}
