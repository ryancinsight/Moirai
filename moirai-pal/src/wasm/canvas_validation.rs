//! Pure validation for browser canvas presentation.

use crate::frame::{MAX_RGBA_BYTES, RGBA_CHANNELS, validate_frame_dimensions};
use std::io;

/// A validated non-empty browser canvas extent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanvasSize {
    width: u32,
    height: u32,
    pixels: u64,
}

impl CanvasSize {
    /// Validates a canvas extent against the provider's pixel bound.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] for zero dimensions, arithmetic
    /// overflow, or a pixel count above the provider bound.
    pub fn new(width: u32, height: u32) -> io::Result<Self> {
        if width == 0 || height == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "canvas dimensions must be non-zero",
            ));
        }
        validate_frame_dimensions(width, height)?;
        let pixels = u64::from(width) * u64::from(height);
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Returns the canvas width in device pixels.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Returns the canvas height in device pixels.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }

    /// Returns the number of pixels in the canvas.
    #[must_use]
    pub const fn pixel_count(self) -> u64 {
        self.pixels
    }

    /// Returns the exact number of RGBA8 bytes required by this canvas.
    #[must_use]
    pub const fn rgba_bytes(self) -> u64 {
        self.pixels * RGBA_CHANNELS
    }
}

/// A borrowed, validated RGBA8 frame for one browser canvas upload.
#[derive(Debug, Eq, PartialEq)]
pub struct RgbaFrame<'pixels> {
    size: CanvasSize,
    pixels: &'pixels [u8],
}

impl<'pixels> RgbaFrame<'pixels> {
    /// Validates an RGBA8 frame without taking ownership of its bytes.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when the slice length does not
    /// match the extent or exceeds the provider's upload bound.
    pub fn new(size: CanvasSize, pixels: &'pixels [u8]) -> io::Result<Self> {
        let expected = size.rgba_bytes();
        let actual = u64::try_from(pixels.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "RGBA frame length cannot be represented",
            )
        })?;
        if expected > MAX_RGBA_BYTES || actual != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "RGBA frame length does not match the canvas extent",
            ));
        }
        Ok(Self { size, pixels })
    }

    /// Returns the frame extent.
    #[must_use]
    pub const fn size(&self) -> CanvasSize {
        self.size
    }

    /// Returns the borrowed RGBA8 bytes.
    #[must_use]
    pub const fn pixels(&self) -> &'pixels [u8] {
        self.pixels
    }
}

#[cfg(test)]
mod tests {
    use super::{CanvasSize, RgbaFrame};
    use std::io::ErrorKind;

    #[test]
    fn canvas_size_accepts_boundaries_and_rejects_invalid_extents() {
        let size = CanvasSize::new(16_384, 1_024).expect("the configured boundary is valid");
        assert_eq!(size.width(), 16_384);
        assert_eq!(size.height(), 1_024);
        assert_eq!(size.pixel_count(), 16_777_216);
        assert_eq!(size.rgba_bytes(), 67_108_864);
        for (width, height) in [(0, 1), (1, 0), (16_385, 1), (16_384, 1_025)] {
            assert_eq!(
                CanvasSize::new(width, height)
                    .expect_err("invalid extent must be rejected")
                    .kind(),
                ErrorKind::InvalidInput
            );
        }
    }

    #[test]
    fn rgba_frame_requires_exact_bounded_storage() {
        let size = CanvasSize::new(2, 2).expect("small extent is valid");
        let pixels = [0_u8; 16];
        let frame = RgbaFrame::new(size, &pixels).expect("exact RGBA bytes are valid");
        assert_eq!(frame.size(), size);
        assert_eq!(frame.pixels(), pixels);
        for pixels in [[0_u8; 15].as_slice(), [0_u8; 17].as_slice()] {
            assert_eq!(
                RgbaFrame::new(size, pixels)
                    .expect_err("mismatched RGBA bytes must be rejected")
                    .kind(),
                ErrorKind::InvalidInput
            );
        }
    }
}
