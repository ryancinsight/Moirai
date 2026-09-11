//! Shared bounded presentation-frame limits for platform providers.

use std::io;

/// Maximum pixels retained by one platform presentation surface.
pub const MAX_FRAME_PIXELS: usize = 16 * 1024 * 1024;
/// Maximum width or height accepted by one platform presentation surface.
pub const MAX_FRAME_DIMENSION: u32 = 16_384;
/// Number of channels in an RGBA8 presentation frame.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) const RGBA_CHANNELS: u64 = 4;
/// Maximum bytes transferred by one RGBA8 presentation frame.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) const MAX_RGBA_BYTES: u64 = (MAX_FRAME_PIXELS as u64) * RGBA_CHANNELS;

pub(crate) fn validate_frame_dimensions(width: u32, height: u32) -> io::Result<()> {
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "frame pixel-count arithmetic overflowed",
            )
        })?;
    if width == 0
        || height == 0
        || width > MAX_FRAME_DIMENSION
        || height > MAX_FRAME_DIMENSION
        || pixels > MAX_FRAME_PIXELS as u64
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "frame dimensions exceed the platform presentation bound",
        ));
    }
    Ok(())
}
