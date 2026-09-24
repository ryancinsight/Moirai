//! Restorable native window placement.
//!
//! A placement is the window's restored (non-maximized) outer rectangle plus
//! whether it is maximized, the state desktop applications persist between
//! sessions. Coordinates are the workspace coordinates `GetWindowPlacement`
//! reports, so a value read from one session restores the same rectangle in
//! the next; the system moves a restored rectangle that would land entirely
//! off every monitor back onto the screen.

use std::io;
use std::mem::size_of;

use windows::Win32::Foundation::RECT;
use windows::Win32::UI::WindowsAndMessaging::{
    GetWindowPlacement, SW_HIDE, SW_SHOWMAXIMIZED, SW_SHOWMINIMIZED, SW_SHOWNORMAL,
    SetWindowPlacement, WINDOWPLACEMENT, WINDOWPLACEMENT_FLAGS, WPF_RESTORETOMAXIMIZED,
};

use super::config::{coordinate_error, validate_frame_dimensions, windows_error};
use super::native::NativeWindow;

/// Largest coordinate magnitude accepted for a restored window's corner.
pub const MAX_PLACEMENT_COORDINATE: i32 = 1 << 16;

/// A validated restored rectangle and maximized flag for one window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowPlacement {
    left: i32,
    top: i32,
    width: u32,
    height: u32,
    maximized: bool,
}

impl WindowPlacement {
    /// Validates a restored outer rectangle and maximized flag.
    ///
    /// # Errors
    /// Returns `InvalidInput` when a corner coordinate exceeds
    /// [`MAX_PLACEMENT_COORDINATE`] in magnitude or the dimensions are outside
    /// the provider's bounded frame range.
    pub fn new(left: i32, top: i32, width: u32, height: u32, maximized: bool) -> io::Result<Self> {
        if left.unsigned_abs() > MAX_PLACEMENT_COORDINATE.unsigned_abs()
            || top.unsigned_abs() > MAX_PLACEMENT_COORDINATE.unsigned_abs()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "window placement corner exceeds the bounded coordinate range",
            ));
        }
        validate_frame_dimensions(width, height)?;
        Ok(Self {
            left,
            top,
            width,
            height,
            maximized,
        })
    }

    /// Left edge of the restored outer rectangle.
    #[must_use]
    pub const fn left(&self) -> i32 {
        self.left
    }

    /// Top edge of the restored outer rectangle.
    #[must_use]
    pub const fn top(&self) -> i32 {
        self.top
    }

    /// Width of the restored outer rectangle.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height of the restored outer rectangle.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Whether the window is, or restores to, the maximized state.
    #[must_use]
    pub const fn maximized(&self) -> bool {
        self.maximized
    }

    fn from_native(native: &WINDOWPLACEMENT, pending_maximized: bool) -> io::Result<Self> {
        let rect = native.rcNormalPosition;
        let width = rect
            .right
            .checked_sub(rect.left)
            .ok_or_else(coordinate_error)?;
        let height = rect
            .bottom
            .checked_sub(rect.top)
            .ok_or_else(coordinate_error)?;
        let maximized = native.showCmd == SW_SHOWMAXIMIZED.0 as u32
            || (native.showCmd == SW_SHOWMINIMIZED.0 as u32
                && native.flags.contains(WPF_RESTORETOMAXIMIZED))
            || pending_maximized;
        Self::new(
            rect.left,
            rect.top,
            u32::try_from(width).map_err(|_| coordinate_error())?,
            u32::try_from(height).map_err(|_| coordinate_error())?,
            maximized,
        )
    }

    fn rect(&self) -> io::Result<RECT> {
        let right = i32::try_from(self.width)
            .ok()
            .and_then(|width| self.left.checked_add(width))
            .ok_or_else(coordinate_error)?;
        let bottom = i32::try_from(self.height)
            .ok()
            .and_then(|height| self.top.checked_add(height))
            .ok_or_else(coordinate_error)?;
        Ok(RECT {
            left: self.left,
            top: self.top,
            right,
            bottom,
        })
    }
}

impl NativeWindow {
    /// Reads the window's restored rectangle and maximized state.
    ///
    /// A hidden window reports the maximized state it will show with.
    ///
    /// # Errors
    /// Returns the native error, `InvalidInput` for a destroyed window, or
    /// `InvalidInput` when the system reports a rectangle outside the
    /// placement bounds.
    pub fn placement(&self) -> io::Result<WindowPlacement> {
        self.require_live("read the placement of")?;
        let native = self.native_placement()?;
        WindowPlacement::from_native(&native, !self.visible && self.show_maximized)
    }

    /// Moves the window to a restored rectangle and maximized state.
    ///
    /// A hidden window stays hidden: the rectangle applies immediately and the
    /// maximized state is used by the next [`NativeWindow::show`].
    ///
    /// # Errors
    /// Returns the native error or `InvalidInput` for a destroyed window.
    pub fn set_placement(&mut self, placement: WindowPlacement) -> io::Result<()> {
        self.require_live("set the placement of")?;
        let mut native = self.native_placement()?;
        native.rcNormalPosition = placement.rect()?;
        native.flags = WINDOWPLACEMENT_FLAGS::default();
        native.showCmd = if !self.visible {
            SW_HIDE.0 as u32
        } else if placement.maximized {
            SW_SHOWMAXIMIZED.0 as u32
        } else {
            SW_SHOWNORMAL.0 as u32
        };
        // SAFETY: `native` is initialized storage with its length field set,
        // and `self.hwnd` is the live handle owned by this thread.
        unsafe { SetWindowPlacement(self.hwnd, &native) }.map_err(windows_error)?;
        self.show_maximized = placement.maximized;
        Ok(())
    }

    fn native_placement(&self) -> io::Result<WINDOWPLACEMENT> {
        let mut native = WINDOWPLACEMENT {
            length: size_of::<WINDOWPLACEMENT>() as u32,
            ..Default::default()
        };
        // SAFETY: `native` is writable storage owned by this call with its
        // length field set, and `self.hwnd` is the live handle of this thread.
        unsafe { GetWindowPlacement(self.hwnd, &mut native) }.map_err(windows_error)?;
        Ok(native)
    }

    fn require_live(&self, action: &str) -> io::Result<()> {
        if self.is_destroyed() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("cannot {action} a destroyed native window"),
            ));
        }
        Ok(())
    }
}
