//! System-wide hotkeys for the WebView2 host's parent window.

use std::io;

use super::super::super::window::{GlobalHotkey, HotkeyId};
use super::error::closed_error;
use super::view::WebViewHost;

impl WebViewHost {
    /// Registers a system-wide hotkey on the parent window.
    ///
    /// # Errors
    /// Returns an error when the host is closed or the registration fails;
    /// see [`crate::windows::window::NativeWindow::register_hotkey`].
    pub fn register_hotkey(&mut self, id: HotkeyId, hotkey: GlobalHotkey) -> io::Result<()> {
        if self.closed {
            return Err(closed_error());
        }
        self.window.register_hotkey(id, hotkey)
    }

    /// Releases a hotkey; returns whether `id` was registered.
    ///
    /// # Errors
    /// Returns the native error when the system refuses the release.
    pub fn unregister_hotkey(&mut self, id: HotkeyId) -> io::Result<bool> {
        self.window.unregister_hotkey(id)
    }

    /// Drains the presses of registered hotkeys, oldest first.
    pub fn take_hotkey_presses(&mut self) -> Vec<HotkeyId> {
        self.window.take_hotkey_presses()
    }
}
