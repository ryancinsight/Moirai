//! Notification-area (tray) icon and notifications for one native window.
//!
//! The icon and the notifications it raises are the services Tauri's tray
//! and notification plugins provide. A window owns at most one icon; its
//! clicks and notification clicks queue apart from [`super::WindowEvent`]
//! and are read with [`NativeWindow::take_tray_events`]. Notifications are
//! shell balloons, which Windows 10 and later present as toasts, so they need
//! no packaged application identity. Closing the window removes the icon.

use std::io;
use std::mem::size_of;

use windows::Win32::UI::Shell::{
    NIF_ICON, NIF_INFO, NIF_MESSAGE, NIF_SHOWTIP, NIF_TIP, NIIF_INFO, NIM_ADD, NIM_DELETE,
    NIM_MODIFY, NIM_SETVERSION, NIN_BALLOONUSERCLICK, NIN_SELECT, NINF_KEY, NOTIFY_ICON_MESSAGE,
    NOTIFYICON_VERSION_4, NOTIFYICONDATAW, Shell_NotifyIconW,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateIcon, DestroyIcon, HICON, WM_APP, WM_CONTEXTMENU,
};

use super::config::{allocation_error, windows_error};
use super::native::NativeWindow;

/// Icon edge lengths the notification area accepts, in pixels.
pub const TRAY_ICON_SIZES: [u32; 2] = [16, 32];
/// Maximum UTF-16 units in a tray tooltip.
pub const MAX_TRAY_TOOLTIP_UNITS: usize = 127;
/// Maximum UTF-16 units in a notification title.
pub const MAX_NOTIFICATION_TITLE_UNITS: usize = 63;
/// Maximum UTF-16 units in a notification body.
pub const MAX_NOTIFICATION_BODY_UNITS: usize = 255;
/// Maximum unread tray events retained; later events are dropped.
pub const MAX_PENDING_TRAY_EVENTS: usize = 64;

/// Window message the shell sends for tray icon activity.
pub(super) const TRAY_CALLBACK_MESSAGE: u32 = WM_APP + 0x37;
const TRAY_ICON_ID: u32 = 1;

/// Activity reported by a window's tray icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrayEvent {
    /// The icon was clicked or chosen with the keyboard.
    Activated,
    /// A context menu was requested at the given screen position.
    ContextRequested {
        /// Horizontal screen coordinate.
        x: i32,
        /// Vertical screen coordinate.
        y: i32,
    },
    /// The user clicked the most recent notification.
    NotificationClicked,
}

/// Decodes a version-4 tray callback's parameters.
pub(super) fn decode_tray_event(wparam: usize, lparam: isize) -> Option<TrayEvent> {
    let event = (lparam as usize & 0xffff) as u32;
    let x = i32::from((wparam & 0xffff) as u16 as i16);
    let y = i32::from(((wparam >> 16) & 0xffff) as u16 as i16);
    match event {
        NIN_SELECT => Some(TrayEvent::Activated),
        key if key == NIN_SELECT | NINF_KEY => Some(TrayEvent::Activated),
        WM_CONTEXTMENU => Some(TrayEvent::ContextRequested { x, y }),
        NIN_BALLOONUSERCLICK => Some(TrayEvent::NotificationClicked),
        _ => None,
    }
}

/// A validated square ARGB image for the tray icon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrayIconImage {
    size: u32,
    pixels: Vec<u32>,
}

impl TrayIconImage {
    /// Validates a square, non-premultiplied `0xAARRGGBB` image.
    ///
    /// # Errors
    /// Returns `InvalidInput` for an edge outside [`TRAY_ICON_SIZES`] or a
    /// pixel count other than `size * size`.
    pub fn new(size: u32, pixels: &[u32]) -> io::Result<Self> {
        let expected = usize::try_from(size * size).map_err(|_| allocation_error())?;
        if !TRAY_ICON_SIZES.contains(&size) || pixels.len() != expected {
            return Err(invalid("tray icon must be a 16 or 32 pixel square"));
        }
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(expected)
            .map_err(|_| allocation_error())?;
        owned.extend_from_slice(pixels);
        Ok(Self {
            size,
            pixels: owned,
        })
    }

    /// Edge length in pixels.
    #[must_use]
    pub const fn size(&self) -> u32 {
        self.size
    }

    fn create_icon(&self) -> io::Result<OwnedIcon> {
        let colour: Vec<u8> = self
            .pixels
            .iter()
            .flat_map(|pixel| pixel.to_le_bytes())
            .collect();
        // A 32-bit icon takes transparency from alpha; the mask is unused.
        let mask = vec![0_u8; colour.len() / 32];
        let edge = i32::try_from(self.size).map_err(|_| allocation_error())?;
        // SAFETY: both buffers live through the synchronous call and hold
        // `size * size` 32-bit pixels and a one-bit mask of the same extent.
        let icon = unsafe { CreateIcon(None, edge, edge, 1, 32, mask.as_ptr(), colour.as_ptr()) }
            .map_err(windows_error)?;
        Ok(OwnedIcon(icon))
    }
}

/// An icon handle destroyed when dropped.
#[derive(Debug)]
pub(super) struct OwnedIcon(HICON);

impl Drop for OwnedIcon {
    fn drop(&mut self) {
        // SAFETY: the handle came from CreateIcon and is dropped once, after
        // the shell no longer displays it.
        let _ = unsafe { DestroyIcon(self.0) };
    }
}

impl NativeWindow {
    /// Shows or replaces this window's tray icon and tooltip.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a destroyed window or an over-long or
    /// NUL-containing tooltip, or the shell error.
    pub fn show_tray_icon(&mut self, image: &TrayIconImage, tooltip: &str) -> io::Result<()> {
        self.require_live_tray()?;
        let mut data = self.tray_data();
        copy_text(
            &mut data.szTip,
            tooltip,
            MAX_TRAY_TOOLTIP_UNITS,
            "tray tooltip",
        )?;
        let icon = image.create_icon()?;
        data.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP | NIF_SHOWTIP;
        data.uCallbackMessage = TRAY_CALLBACK_MESSAGE;
        data.hIcon = icon.0;
        if self.tray_icon.is_some() {
            notify_shell(NIM_MODIFY, &data)?;
        } else {
            notify_shell(NIM_ADD, &data)?;
            data.Anonymous.uVersion = NOTIFYICON_VERSION_4;
            if let Err(error) = notify_shell(NIM_SETVERSION, &data) {
                let _ = notify_shell(NIM_DELETE, &data);
                return Err(error);
            }
        }
        self.tray_icon = Some(icon);
        Ok(())
    }

    /// Shows a notification from this window's tray icon.
    ///
    /// # Errors
    /// Returns `InvalidInput` without a tray icon, for an empty body, or for
    /// text over its bound or containing NUL, or the shell error.
    pub fn show_notification(&mut self, title: &str, body: &str) -> io::Result<()> {
        self.require_live_tray()?;
        if self.tray_icon.is_none() {
            return Err(invalid("a notification needs a tray icon"));
        }
        if body.is_empty() {
            return Err(invalid("a notification needs a body"));
        }
        let mut data = self.tray_data();
        copy_text(
            &mut data.szInfoTitle,
            title,
            MAX_NOTIFICATION_TITLE_UNITS,
            "notification title",
        )?;
        copy_text(
            &mut data.szInfo,
            body,
            MAX_NOTIFICATION_BODY_UNITS,
            "notification body",
        )?;
        data.uFlags = NIF_INFO;
        data.dwInfoFlags = NIIF_INFO;
        notify_shell(NIM_MODIFY, &data)
    }

    /// Removes the tray icon; returns whether one was shown.
    ///
    /// # Errors
    /// Returns the shell error; the icon is then kept.
    pub fn remove_tray_icon(&mut self) -> io::Result<bool> {
        if self.tray_icon.is_none() {
            return Ok(false);
        }
        notify_shell(NIM_DELETE, &self.tray_data())?;
        self.tray_icon = None;
        self.state.tray_events.clear();
        Ok(true)
    }

    /// Drains the tray icon's queued activity, oldest first.
    pub fn take_tray_events(&mut self) -> Vec<TrayEvent> {
        self.state.tray_events.drain(..).collect()
    }

    /// Removes the icon while the window closes; failures cannot be acted on.
    pub(super) fn release_tray_icon(&mut self) {
        if self.tray_icon.is_some() {
            let _ = notify_shell(NIM_DELETE, &self.tray_data());
            self.tray_icon = None;
        }
        self.state.tray_events.clear();
    }

    fn tray_data(&self) -> NOTIFYICONDATAW {
        NOTIFYICONDATAW {
            cbSize: size_of::<NOTIFYICONDATAW>() as u32,
            hWnd: self.hwnd,
            uID: TRAY_ICON_ID,
            ..Default::default()
        }
    }

    fn require_live_tray(&self) -> io::Result<()> {
        if self.is_destroyed() {
            return Err(invalid("a destroyed native window has no tray icon"));
        }
        Ok(())
    }
}

fn notify_shell(message: NOTIFY_ICON_MESSAGE, data: &NOTIFYICONDATAW) -> io::Result<()> {
    // SAFETY: `data` is initialized with its size field set and names this
    // thread's live window; the shell copies it during the call.
    if unsafe { Shell_NotifyIconW(message, data) }.as_bool() {
        Ok(())
    } else {
        Err(io::Error::other("the shell refused the tray icon request"))
    }
}

/// Copies `text` into a fixed NUL-terminated buffer of `limit` units.
fn copy_text(buffer: &mut [u16], text: &str, limit: usize, label: &'static str) -> io::Result<()> {
    let units = text.encode_utf16().count();
    if text.contains('\0') || units > limit || units >= buffer.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{label} exceeds its bound or contains NUL"),
        ));
    }
    for (slot, unit) in buffer.iter_mut().zip(text.encode_utf16()) {
        *slot = unit;
    }
    buffer[units] = 0;
    Ok(())
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
