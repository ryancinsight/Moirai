//! Validated Win32 window construction values.

use std::io;

/// Maximum pending events retained for one native window.
pub const MAX_WINDOW_EVENTS: usize = 1_024;
/// Maximum pixels retained by the software presenter.
pub const MAX_FRAME_PIXELS: usize = 16 * 1024 * 1024;
/// Maximum width or height accepted by the software presenter.
pub const MAX_FRAME_DIMENSION: u32 = 16_384;
/// Maximum UTF-16 code units accepted for a window title.
pub const MAX_TITLE_UNITS: usize = 256;
/// Maximum number of messages dispatched during one pump call.
pub const MAX_PUMP_MESSAGES: usize = 1_024;
/// Maximum finite wait accepted by the native event pump.
pub const MAX_WAIT_MILLISECONDS: u32 = 30_000;

/// Initial visibility for a native window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowVisibility {
    /// Create the HWND without showing it.
    Hidden,
    /// Create and show the HWND immediately.
    Visible,
}

/// Validated native-window construction parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowConfig {
    title: Vec<u16>,
    width: u32,
    height: u32,
    visibility: WindowVisibility,
}

impl WindowConfig {
    /// Validates a title and initial client dimensions.
    ///
    /// # Errors
    /// Returns `InvalidInput` for an empty/NUL title or dimensions outside the
    /// provider's bounded coordinate and pixel range.
    pub fn new(title: impl AsRef<str>, width: u32, height: u32) -> io::Result<Self> {
        Self::with_visibility(title, width, height, WindowVisibility::Visible)
    }

    /// Validates a title, dimensions and requested visibility.
    ///
    /// # Errors
    /// Returns `InvalidInput` when the title or dimensions are not representable
    /// by the native provider.
    pub fn with_visibility(
        title: impl AsRef<str>,
        width: u32,
        height: u32,
        visibility: WindowVisibility,
    ) -> io::Result<Self> {
        let title = title.as_ref();
        if title.is_empty() || title.contains('\0') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "window title must be non-empty and NUL-free",
            ));
        }
        let units = title.encode_utf16().count();
        if units > MAX_TITLE_UNITS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "window title exceeds the bounded UTF-16 limit",
            ));
        }
        validate_frame_dimensions(width, height)?;
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(units + 1)
            .map_err(|_| allocation_error())?;
        encoded.extend(title.encode_utf16());
        encoded.push(0);
        Ok(Self {
            title: encoded,
            width,
            height,
            visibility,
        })
    }

    /// Requested client width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Requested client height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Requested title converted from the validated UTF-16 representation.
    #[must_use]
    pub fn title(&self) -> String {
        String::from_utf16_lossy(&self.title[..self.title.len().saturating_sub(1)])
    }

    pub(super) fn title_utf16(&self) -> &[u16] {
        &self.title
    }

    /// Requested initial visibility.
    #[must_use]
    pub const fn visibility(&self) -> WindowVisibility {
        self.visibility
    }
}

pub(super) fn validate_frame_dimensions(width: u32, height: u32) -> io::Result<()> {
    let pixels = u64::from(width) * u64::from(height);
    if width == 0
        || height == 0
        || width > MAX_FRAME_DIMENSION
        || height > MAX_FRAME_DIMENSION
        || pixels > MAX_FRAME_PIXELS as u64
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "frame dimensions exceed the native presentation bound",
        ));
    }
    Ok(())
}

pub(super) fn coordinate_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "window dimensions exceed Win32 coordinate range",
    )
}

pub(super) fn allocation_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::OutOfMemory,
        "native window storage reservation failed",
    )
}

pub(super) fn windows_error(error: windows::core::Error) -> io::Error {
    io::Error::from_raw_os_error(error.code().0)
}
