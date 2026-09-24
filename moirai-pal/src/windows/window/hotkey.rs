//! System-wide hotkeys delivered to one native window.
//!
//! A registered hotkey fires even while another application has focus, the
//! service Tauri's global-shortcut plugin provides. Presses are queued apart
//! from [`super::WindowEvent`] and read with
//! [`NativeWindow::take_hotkey_presses`]; the window's event wait wakes for
//! them like any other input. Registrations belong to the window and are
//! released when it closes.

use std::io;

use windows::Win32::UI::Input::KeyboardAndMouse::{
    HOT_KEY_MODIFIERS, MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, MOD_SHIFT, MOD_WIN, RegisterHotKey,
    UnregisterHotKey,
};

use super::config::{allocation_error, windows_error};
use super::event::ModifierState;
use super::native::NativeWindow;

/// Maximum hotkeys one window may hold at once.
pub const MAX_GLOBAL_HOTKEYS: usize = 32;
/// Maximum unread hotkey presses retained; later presses are dropped.
pub const MAX_PENDING_HOTKEY_PRESSES: usize = 64;
/// Largest identifier an application may register (`0xBFFF`); higher values
/// are reserved for shared libraries.
pub const MAX_HOTKEY_ID: u16 = 0xBFFF;

/// Identifier a window reports when one of its hotkeys is pressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HotkeyId(u16);

impl HotkeyId {
    /// Validates an application hotkey identifier.
    ///
    /// # Errors
    /// Returns `InvalidInput` above [`MAX_HOTKEY_ID`].
    pub fn new(id: u16) -> io::Result<Self> {
        if id > MAX_HOTKEY_ID {
            return Err(invalid("hotkey identifier is in the shared-library range"));
        }
        Ok(Self(id))
    }

    /// The raw identifier.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// A validated modifier chord and virtual key for system-wide registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalHotkey {
    modifiers: ModifierState,
    virtual_key: u32,
}

impl GlobalHotkey {
    /// Validates a hotkey chord.
    ///
    /// At least one modifier is required, so a registration cannot take a
    /// plain key away from every other application, and the key itself must
    /// not be a modifier.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a chord without modifiers, a key outside
    /// `0x01..=0xFE` or a modifier key.
    pub fn new(modifiers: ModifierState, virtual_key: u32) -> io::Result<Self> {
        if modifiers == ModifierState::NONE {
            return Err(invalid("global hotkey requires at least one modifier"));
        }
        let modifier_key = matches!(virtual_key, 0x10..=0x12 | 0x5B | 0x5C | 0xA0..=0xA5);
        if !(0x01..=0xFE).contains(&virtual_key) || modifier_key {
            return Err(invalid(
                "global hotkey key must be a non-modifier virtual key",
            ));
        }
        Ok(Self {
            modifiers,
            virtual_key,
        })
    }

    /// Modifier keys that must be held.
    #[must_use]
    pub const fn modifiers(self) -> ModifierState {
        self.modifiers
    }

    /// Windows virtual-key code of the non-modifier key.
    #[must_use]
    pub const fn virtual_key(self) -> u32 {
        self.virtual_key
    }

    fn native_modifiers(self) -> HOT_KEY_MODIFIERS {
        let mut flags = MOD_NOREPEAT;
        for (held, flag) in [
            (self.modifiers.ctrl(), MOD_CONTROL),
            (self.modifiers.alt(), MOD_ALT),
            (self.modifiers.shift(), MOD_SHIFT),
            (self.modifiers.meta(), MOD_WIN),
        ] {
            if held {
                flags |= flag;
            }
        }
        flags
    }
}

impl NativeWindow {
    /// Registers a system-wide hotkey reported to this window as `id`.
    ///
    /// Auto-repeat is suppressed, so holding the chord reports one press.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a destroyed window, `AlreadyExists` when
    /// this window already uses `id`, `OutOfMemory` past
    /// [`MAX_GLOBAL_HOTKEYS`], or the native error when another application
    /// already owns the chord.
    pub fn register_hotkey(&mut self, id: HotkeyId, hotkey: GlobalHotkey) -> io::Result<()> {
        if self.is_destroyed() {
            return Err(invalid(
                "cannot register a hotkey on a destroyed native window",
            ));
        }
        if self.hotkeys.contains(&id) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "hotkey identifier is already registered on this window",
            ));
        }
        if self.hotkeys.len() >= MAX_GLOBAL_HOTKEYS {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "native window hotkey capacity exceeded",
            ));
        }
        self.hotkeys
            .try_reserve(1)
            .map_err(|_| allocation_error())?;
        // SAFETY: `self.hwnd` is the live handle owned by this thread; the
        // call copies its scalar arguments.
        unsafe {
            RegisterHotKey(
                Some(self.hwnd),
                i32::from(id.0),
                hotkey.native_modifiers(),
                hotkey.virtual_key,
            )
        }
        .map_err(windows_error)?;
        self.hotkeys.push(id);
        Ok(())
    }

    /// Releases a hotkey; returns whether `id` was registered.
    ///
    /// # Errors
    /// Returns the native error when the system refuses the release; the
    /// registration is then kept.
    pub fn unregister_hotkey(&mut self, id: HotkeyId) -> io::Result<bool> {
        let Some(index) = self.hotkeys.iter().position(|held| *held == id) else {
            return Ok(false);
        };
        // SAFETY: the identifier was registered for this live handle.
        unsafe { UnregisterHotKey(Some(self.hwnd), i32::from(id.0)) }.map_err(windows_error)?;
        self.hotkeys.swap_remove(index);
        self.state.hotkey_presses.retain(|pressed| *pressed != id.0);
        Ok(true)
    }

    /// Drains the presses of currently registered hotkeys, oldest first.
    ///
    /// Presses are queued as the window pumps messages, so call this after
    /// [`NativeWindow::poll_events`] or [`NativeWindow::wait_events`].
    pub fn take_hotkey_presses(&mut self) -> Vec<HotkeyId> {
        let hotkeys = &self.hotkeys;
        self.state
            .hotkey_presses
            .drain(..)
            .map(HotkeyId)
            .filter(|id| hotkeys.contains(id))
            .collect()
    }

    /// Releases every registration; used when the window closes.
    pub(super) fn release_hotkeys(&mut self) {
        for id in self.hotkeys.drain(..) {
            // SAFETY: each identifier was registered for this handle. A
            // handle already destroyed by the system makes the call fail
            // harmlessly, and close cannot act on that failure.
            let _ = unsafe { UnregisterHotKey(Some(self.hwnd), i32::from(id.0)) };
        }
        self.state.hotkey_presses.clear();
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
