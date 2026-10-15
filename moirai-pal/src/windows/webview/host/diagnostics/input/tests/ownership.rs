//! Real native sibling/foreign focus remains outside the controller's parent.

use crate::windows::{
    webview::WebViewHost,
    window::{NativeWindow, WindowConfig, WindowVisibility},
};
use std::time::Duration;
use windows::{
    Win32::{
        Foundation::HWND,
        UI::{
            Input::KeyboardAndMouse::{GetFocus, SetFocus},
            WindowsAndMessaging::{
                CreateWindowExW, DestroyWindow, GetForegroundWindow, IsChild, SetForegroundWindow,
                WINDOW_EX_STYLE, WS_CHILD, WS_TABSTOP, WS_VISIBLE,
            },
        },
    },
    core::w,
};

pub(super) fn private_parent(host: &WebViewHost) -> HWND {
    let mut parent = HWND::default();
    unsafe {
        host.controller
            .as_ref()
            .expect("controller")
            .ParentWindow(&mut parent)
    }
    .expect("SDK parent");
    assert_ne!(
        parent, host.window.hwnd,
        "controller has its own exclusive native parent"
    );
    assert!(unsafe { IsChild(host.window.hwnd, parent) }.as_bool());
    parent
}

struct Sibling(Option<HWND>);
impl Sibling {
    fn new(parent: HWND) -> Self {
        // SAFETY: the caller owns this live parent on the creating thread.
        // EDIT is a predefined native system control, not a substitute view.
        let hwnd = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                w!("EDIT"),
                w!("native sibling"),
                WS_CHILD | WS_VISIBLE | WS_TABSTOP,
                0,
                0,
                120,
                24,
                Some(parent),
                None,
                None,
                None,
            )
        }
        .expect("real native sibling");
        Self(Some(hwnd))
    }
    fn close(&mut self) -> windows::core::Result<()> {
        if let Some(hwnd) = self.0 {
            unsafe { DestroyWindow(hwnd) }?;
            self.0 = None;
        }
        Ok(())
    }
}
impl Drop for Sibling {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::error!(%error,"native sibling cleanup failed");
        }
    }
}

pub(super) fn preserve_external_focus(host: &mut WebViewHost) -> NativeWindow {
    let _phase = tracing::info_span!("act", case = "native sibling and foreign focus").entered();
    let parent = private_parent(host);
    let mut sibling = Sibling::new(host.window.hwnd);
    let handle = sibling.0.expect("native sibling");
    assert!(
        !unsafe { IsChild(parent, handle) }.as_bool(),
        "sibling is outside the controller-owned tree"
    );
    unsafe { SetFocus(Some(handle)) }.expect("real sibling focus");
    host.wait_events(Duration::from_millis(25))
        .expect("native focus pump");
    assert_eq!(
        unsafe { GetFocus() },
        handle,
        "controller does not steal sibling focus"
    );
    sibling.close().expect("native sibling destruction");
    let foreign = NativeWindow::new(
        &WindowConfig::with_visibility(
            "Moirai foreign focus witness",
            128,
            96,
            WindowVisibility::Visible,
        )
        .expect("foreign window configuration"),
    )
    .expect("real foreign window");
    unsafe {
        assert!(SetForegroundWindow(foreign.hwnd).as_bool());
        SetFocus(Some(foreign.hwnd)).expect("real foreign focus");
    }
    host.wait_events(Duration::from_millis(25))
        .expect("foreign focus pump");
    assert_foreign_focus(&foreign);
    foreign
}

pub(super) fn assert_foreign_focus(window: &NativeWindow) {
    assert_eq!(
        unsafe { GetForegroundWindow() },
        window.hwnd,
        "foreign foreground remains owned"
    );
    assert_eq!(
        unsafe { GetFocus() },
        window.hwnd,
        "foreign native focus remains owned"
    );
}
