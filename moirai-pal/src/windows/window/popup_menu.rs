//! Native context menus shown at a screen position.
//!
//! A popup menu is what a tray icon's context request, or a right click in a
//! window, opens. [`NativeWindow::show_popup_menu`] runs the system's modal
//! menu loop on the window's thread and returns the chosen item, so the menu
//! needs no command routing: the caller maps the index to its own command.

use std::io;

use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    AppendMenuW, CreatePopupMenu, DestroyMenu, HMENU, MF_GRAYED, MF_SEPARATOR, MF_STRING,
    PostMessageW, SetForegroundWindow, TPM_NONOTIFY, TPM_RETURNCMD, TPM_RIGHTBUTTON,
    TrackPopupMenu, WM_NULL,
};
use windows::core::PCWSTR;

use super::config::{allocation_error, windows_error};
use super::native::NativeWindow;

/// Maximum items, separators included, in one popup menu.
pub const MAX_POPUP_MENU_ITEMS: usize = 32;
/// Maximum UTF-16 units in one item label.
pub const MAX_POPUP_MENU_LABEL_UNITS: usize = 64;

/// One entry of a popup menu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PopupMenuItem {
    label: Option<Vec<u16>>,
    enabled: bool,
}

impl PopupMenuItem {
    /// A selectable item; a disabled item is shown greyed out.
    ///
    /// # Errors
    /// Returns `InvalidInput` for an empty, NUL-containing or over-long label.
    pub fn action(label: &str, enabled: bool) -> io::Result<Self> {
        Ok(Self {
            label: Some(encode_label(label)?),
            enabled,
        })
    }

    /// A separator line between groups of items.
    #[must_use]
    pub const fn separator() -> Self {
        Self {
            label: None,
            enabled: false,
        }
    }
}

/// A validated list of popup menu items.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PopupMenu {
    items: Vec<PopupMenuItem>,
}

impl PopupMenu {
    /// Validates the items.
    ///
    /// # Errors
    /// Returns `InvalidInput` for more than [`MAX_POPUP_MENU_ITEMS`] items or
    /// a menu without a selectable item.
    pub fn new(items: Vec<PopupMenuItem>) -> io::Result<Self> {
        if items.len() > MAX_POPUP_MENU_ITEMS {
            return Err(invalid("popup menu has too many items"));
        }
        if !items.iter().any(|item| item.label.is_some()) {
            return Err(invalid("popup menu needs a selectable item"));
        }
        Ok(Self { items })
    }

    /// Number of items, separators included.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the menu has no items; a validated menu never is.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Builds the native menu; item `i` reports command `first_id + i`.
    pub(super) fn build(&self, first_id: usize) -> io::Result<OwnedMenu> {
        // SAFETY: CreatePopupMenu has no inputs; the handle is owned below.
        let menu = OwnedMenu(unsafe { CreatePopupMenu() }.map_err(windows_error)?);
        for (index, item) in self.items.iter().enumerate() {
            // SAFETY: the menu handle is live and each label buffer is
            // NUL-terminated and outlives the synchronous call, which copies it.
            unsafe {
                match &item.label {
                    Some(label) => {
                        let flags = if item.enabled {
                            MF_STRING
                        } else {
                            MF_STRING | MF_GRAYED
                        };
                        AppendMenuW(menu.0, flags, first_id + index, PCWSTR(label.as_ptr()))
                    }
                    None => AppendMenuW(menu.0, MF_SEPARATOR, 0, PCWSTR::null()),
                }
            }
            .map_err(windows_error)?;
        }
        Ok(menu)
    }
}

/// A NUL-terminated label of 1 to [`MAX_POPUP_MENU_LABEL_UNITS`] units.
pub(super) fn encode_label(label: &str) -> io::Result<Vec<u16>> {
    let units = label.encode_utf16().count();
    if label.is_empty() || label.contains('\0') || units > MAX_POPUP_MENU_LABEL_UNITS {
        return Err(invalid("menu label must be 1 to 64 NUL-free units"));
    }
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(units + 1)
        .map_err(|_| allocation_error())?;
    encoded.extend(label.encode_utf16());
    encoded.push(0);
    Ok(encoded)
}

/// A menu handle destroyed when dropped, unless ownership passes to a
/// parent menu or window through [`OwnedMenu::into_raw`].
pub(super) struct OwnedMenu(pub(super) HMENU);

impl OwnedMenu {
    /// Releases ownership; the caller's new owner destroys the handle.
    pub(super) fn into_raw(self) -> HMENU {
        let handle = self.0;
        std::mem::forget(self);
        handle
    }
}

impl Drop for OwnedMenu {
    fn drop(&mut self) {
        // SAFETY: the handle came from CreatePopupMenu and is destroyed once.
        let _ = unsafe { DestroyMenu(self.0) };
    }
}

impl NativeWindow {
    /// Shows `menu` at a screen position and waits for the user's choice.
    ///
    /// Returns the index of the chosen item, or `None` when the menu was
    /// dismissed. The call runs the system's modal menu loop on this thread.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a destroyed window, or the native error
    /// while building the menu.
    pub fn show_popup_menu(
        &mut self,
        menu: &PopupMenu,
        x: i32,
        y: i32,
    ) -> io::Result<Option<usize>> {
        if self.is_destroyed() {
            return Err(invalid("cannot show a menu for a destroyed native window"));
        }
        let native = menu.build(1)?;
        // SAFETY: the window and menu handles are live on this thread. The
        // foreground call lets the menu close when the user clicks elsewhere,
        // and the posted null message completes the tray-menu handshake.
        let chosen = unsafe {
            let _ = SetForegroundWindow(self.hwnd);
            let chosen = TrackPopupMenu(
                native.0,
                TPM_RETURNCMD | TPM_NONOTIFY | TPM_RIGHTBUTTON,
                x,
                y,
                None,
                self.hwnd,
                None,
            );
            let _ = PostMessageW(Some(self.hwnd), WM_NULL, WPARAM(0), LPARAM(0));
            chosen
        };
        Ok(usize::try_from(chosen.0)
            .ok()
            .and_then(|id| id.checked_sub(1))
            .filter(|index| *index < menu.len()))
    }
}

pub(super) fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
