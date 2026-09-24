//! Tray icon, notifications, popup menus and the menu bar for the WebView2
//! host's parent window.

use std::io;

use super::super::super::window::{MenuBar, MenuCommand, PopupMenu, TrayEvent, TrayIconImage};
use super::error::closed_error;
use super::view::WebViewHost;

impl WebViewHost {
    /// Shows or replaces the parent window's tray icon and tooltip.
    ///
    /// # Errors
    /// Returns an error when the host is closed or the shell refuses; see
    /// [`crate::windows::window::NativeWindow::show_tray_icon`].
    pub fn show_tray_icon(&mut self, image: &TrayIconImage, tooltip: &str) -> io::Result<()> {
        if self.closed {
            return Err(closed_error());
        }
        self.window.show_tray_icon(image, tooltip)
    }

    /// Shows a notification from the parent window's tray icon.
    ///
    /// # Errors
    /// Returns an error when the host is closed, no icon is shown, the text
    /// is invalid or the shell refuses.
    pub fn show_notification(&mut self, title: &str, body: &str) -> io::Result<()> {
        if self.closed {
            return Err(closed_error());
        }
        self.window.show_notification(title, body)
    }

    /// Removes the tray icon; returns whether one was shown.
    ///
    /// # Errors
    /// Returns the shell error.
    pub fn remove_tray_icon(&mut self) -> io::Result<bool> {
        self.window.remove_tray_icon()
    }

    /// Shows a context menu at a screen position and waits for the choice.
    ///
    /// # Errors
    /// Returns an error when the host is closed or the menu cannot be built;
    /// see [`crate::windows::window::NativeWindow::show_popup_menu`].
    pub fn show_popup_menu(
        &mut self,
        menu: &PopupMenu,
        x: i32,
        y: i32,
    ) -> io::Result<Option<usize>> {
        if self.closed {
            return Err(closed_error());
        }
        self.window.show_popup_menu(menu, x, y)
    }

    /// Drains the tray icon's queued activity, oldest first.
    pub fn take_tray_events(&mut self) -> Vec<TrayEvent> {
        self.window.take_tray_events()
    }

    /// Attaches, replaces or removes the parent window's menu bar.
    ///
    /// # Errors
    /// Returns an error when the host is closed or the menu cannot be built;
    /// see [`crate::windows::window::NativeWindow::set_menu_bar`].
    pub fn set_menu_bar(&mut self, bar: Option<&MenuBar>) -> io::Result<()> {
        if self.closed {
            return Err(closed_error());
        }
        self.window.set_menu_bar(bar)
    }

    /// Drains the chosen menu-bar items, oldest first.
    pub fn take_menu_commands(&mut self) -> Vec<MenuCommand> {
        self.window.take_menu_commands()
    }
}
