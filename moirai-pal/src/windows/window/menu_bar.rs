//! Native menu bars attached to a window.
//!
//! A menu bar is a row of titled [`PopupMenu`]s shown under the title bar,
//! the service `muda` provides to Tauri and Dioxus. Choosing an item queues
//! a [`MenuCommand`] apart from [`super::WindowEvent`], read with
//! [`NativeWindow::take_menu_commands`], so the application maps the menu
//! and item index to its own command.

use std::io;

use windows::Win32::UI::WindowsAndMessaging::{
    AppendMenuW, CreateMenu, DestroyMenu, DrawMenuBar, HMENU, MF_POPUP, MF_STRING, SetMenu,
};
use windows::core::PCWSTR;

use super::config::windows_error;
use super::native::NativeWindow;
use super::popup_menu::{MAX_POPUP_MENU_ITEMS, OwnedMenu, PopupMenu, encode_label, invalid};

/// Most menus one menu bar holds.
pub const MAX_MENU_BAR_MENUS: usize = 16;
/// Most unread menu commands retained; later ones are dropped.
pub const MAX_PENDING_MENU_COMMANDS: usize = 64;

/// Command identifiers of menu `m` start at `(m + 1) * COMMAND_STRIDE`.
const COMMAND_STRIDE: usize = 256;
const _: () = assert!(MAX_POPUP_MENU_ITEMS < COMMAND_STRIDE);
const _: () = assert!((MAX_MENU_BAR_MENUS + 1) * COMMAND_STRIDE <= u16::MAX as usize);

/// A chosen menu-bar item: its menu's and its own position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MenuCommand {
    /// Index of the menu in the bar.
    pub menu: usize,
    /// Index of the item in that menu, separators included.
    pub item: usize,
}

/// A validated row of titled menus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MenuBar {
    menus: Vec<(Vec<u16>, PopupMenu)>,
}

impl MenuBar {
    /// Validates the titled menus; a title may mark its access key with `&`.
    ///
    /// # Errors
    /// Returns `InvalidInput` for no menus, more than
    /// [`MAX_MENU_BAR_MENUS`], or an empty, NUL-containing or over-long
    /// title.
    pub fn new<S: AsRef<str>>(menus: Vec<(S, PopupMenu)>) -> io::Result<Self> {
        if menus.is_empty() || menus.len() > MAX_MENU_BAR_MENUS {
            return Err(invalid("a menu bar holds 1 to 16 menus"));
        }
        let menus = menus
            .into_iter()
            .map(|(title, menu)| Ok((encode_label(title.as_ref())?, menu)))
            .collect::<io::Result<_>>()?;
        Ok(Self { menus })
    }

    fn build(&self) -> io::Result<OwnedMenu> {
        // SAFETY: CreateMenu has no inputs; the handle is owned below.
        let bar = OwnedMenu(unsafe { CreateMenu() }.map_err(windows_error)?);
        for (index, (title, menu)) in self.menus.iter().enumerate() {
            let submenu = menu.build((index + 1) * COMMAND_STRIDE)?;
            // SAFETY: both handles are live; the title buffer is
            // NUL-terminated and copied during the call. On success the bar
            // owns the submenu and destroys it with itself.
            unsafe {
                AppendMenuW(
                    bar.0,
                    MF_POPUP | MF_STRING,
                    submenu.0.0 as usize,
                    PCWSTR(title.as_ptr()),
                )
            }
            .map_err(windows_error)?;
            let _ = submenu.into_raw();
        }
        Ok(bar)
    }

    fn shape(&self) -> Vec<usize> {
        self.menus.iter().map(|(_, menu)| menu.len()).collect()
    }
}

/// Decodes a `WM_COMMAND` identifier against the attached bar's shape.
pub(super) fn decode_command(id: u16, shape: &[usize]) -> Option<MenuCommand> {
    let id = usize::from(id);
    let menu = (id / COMMAND_STRIDE).checked_sub(1)?;
    let item = id % COMMAND_STRIDE;
    (item < *shape.get(menu)?).then_some(MenuCommand { menu, item })
}

impl NativeWindow {
    /// Attaches `bar` under the title bar, replacing any earlier bar, or
    /// removes the bar when `bar` is `None`.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a destroyed window, or the native error;
    /// the earlier bar is then kept.
    pub fn set_menu_bar(&mut self, bar: Option<&MenuBar>) -> io::Result<()> {
        if self.is_destroyed() {
            return Err(invalid("a destroyed native window has no menu bar"));
        }
        let built = bar.map(MenuBar::build).transpose()?;
        let handle = built.as_ref().map(|menu| menu.0);
        // SAFETY: the window and the new menu are live on this thread; on
        // success the window owns the new menu.
        unsafe { SetMenu(self.hwnd, handle) }.map_err(windows_error)?;
        let _ = built.map(OwnedMenu::into_raw);
        if let Some(previous) = std::mem::replace(&mut self.menu_bar, handle) {
            destroy(previous);
        }
        self.menu_shape = bar.map(MenuBar::shape).unwrap_or_default();
        self.state.menu_commands.clear();
        // SAFETY: redraws the live window's non-client area.
        let _ = unsafe { DrawMenuBar(self.hwnd) };
        Ok(())
    }

    /// Drains the chosen menu-bar items, oldest first.
    pub fn take_menu_commands(&mut self) -> Vec<MenuCommand> {
        let shape = &self.menu_shape;
        self.state
            .menu_commands
            .drain(..)
            .filter_map(|id| decode_command(id, shape))
            .collect()
    }
}

fn destroy(menu: HMENU) {
    // SAFETY: the menu was detached from the window and is destroyed once.
    let _ = unsafe { DestroyMenu(menu) };
}
