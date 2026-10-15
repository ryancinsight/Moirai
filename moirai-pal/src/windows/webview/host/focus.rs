//! The stable controller owner for synchronous native parent focus.
//!
//! PROGRAMMATIC restores the previous page element without synthesizing Tab,
//! selecting text or changing Handled. The native window retains only Weak
//! ownership, so moving WebViewHost never invalidates a callback reference.

use std::{io, rc::Rc};

use webview2_com::Microsoft::Web::WebView2::Win32::{
    COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC, ICoreWebView2Controller,
};

use crate::windows::window::{NativeWindow, focus::NativeFocusTarget};

use super::error::callback_context;

pub(super) struct ControllerFocus(ICoreWebView2Controller);

impl ControllerFocus {
    pub(super) fn attach(
        window: &mut NativeWindow,
        controller: &ICoreWebView2Controller,
    ) -> io::Result<Rc<Self>> {
        let owner = Rc::new(Self(controller.clone()));
        let target: Rc<dyn NativeFocusTarget> = owner.clone();
        window.bind_focus_target(Rc::downgrade(&target))?;
        Ok(owner)
    }
}

impl NativeFocusTarget for ControllerFocus {
    fn focus(&self) -> io::Result<()> {
        // SAFETY: the owner and its controller remain on their creating STA.
        // No window-state borrow survives into this reentrant native call.
        unsafe {
            self.0
                .MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC)
        }
        .map_err(|error| {
            io::Error::other(callback_context("WebView2 MoveFocus(PROGRAMMATIC)", error))
        })
    }
}
