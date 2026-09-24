//! The retained software frame: presenting it, in whole or in part, and
//! painting it into the client area.

use std::ffi::c_void;
use std::io;
use std::mem::size_of;

use windows::Win32::Foundation::{HWND, LRESULT, RECT};
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BeginPaint, DIB_RGB_COLORS, EndPaint, HDC,
    InvalidateRect, PAINTSTRUCT, RGBQUAD, SRCCOPY, StretchDIBits,
};
use windows::Win32::UI::WindowsAndMessaging::GetClientRect;

use super::config::{allocation_error, validate_frame_dimensions};
use super::native::NativeWindow;
use super::state::{PresentedFrame, WindowState};

/// A half-open pixel rectangle of a presented frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRegion {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
}

impl FrameRegion {
    /// The columns `[left, right)` of the rows `[top, bottom)`.
    ///
    /// # Errors
    /// Returns `InvalidInput` when an edge precedes its opposite edge.
    pub fn new(left: u32, top: u32, right: u32, bottom: u32) -> io::Result<Self> {
        if right < left || bottom < top {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "frame region edges are out of order",
            ));
        }
        Ok(Self {
            left,
            top,
            right,
            bottom,
        })
    }

    const fn whole(width: u32, height: u32) -> Self {
        Self {
            left: 0,
            top: 0,
            right: width,
            bottom: height,
        }
    }

    const fn is_empty(self) -> bool {
        self.right == self.left || self.bottom == self.top
    }

    fn rect(self) -> io::Result<RECT> {
        let edge = |value: u32| {
            i32::try_from(value).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "frame region exceeds i32")
            })
        };
        Ok(RECT {
            left: edge(self.left)?,
            top: edge(self.top)?,
            right: edge(self.right)?,
            bottom: edge(self.bottom)?,
        })
    }
}

impl NativeWindow {
    /// Retains a bounded ARGB frame and schedules a repaint.
    ///
    /// The input uses the same row-major `0xAARRGGBB` representation as Atlas
    /// software framebuffers. The slice is copied because Windows may repaint
    /// after this method returns.
    ///
    /// # Errors
    /// Rejects mismatched lengths, zero or oversized dimensions, allocation
    /// failure and an invalid native window handle.
    pub fn present_argb8888(&mut self, width: u32, height: u32, pixels: &[u32]) -> io::Result<()> {
        self.present_argb8888_region(width, height, pixels, FrameRegion::whole(width, height))
    }

    /// Retains a frame whose pixels differ from the retained one only inside
    /// `region`, and repaints only that region.
    ///
    /// Only the region's rows are copied and only its rectangle is
    /// invalidated, so a small change costs its own area rather than the
    /// window's. A frame of new dimensions, or a client area of a different
    /// size, is copied and repainted whole; an empty region presents nothing.
    ///
    /// # Errors
    /// Rejects mismatched lengths, zero or oversized dimensions, a region
    /// outside the frame, allocation failure and an invalid native window
    /// handle.
    pub fn present_argb8888_region(
        &mut self,
        width: u32,
        height: u32,
        pixels: &[u32],
        region: FrameRegion,
    ) -> io::Result<()> {
        validate_frame_dimensions(width, height)?;
        let count = usize::try_from(u64::from(width) * u64::from(height))
            .map_err(|_| allocation_error())?;
        if pixels.len() != count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "ARGB frame length does not match dimensions",
            ));
        }
        if region.right > width || region.bottom > height {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "frame region lies outside the frame",
            ));
        }
        let retained = self
            .state
            .frame
            .as_ref()
            .is_some_and(|frame| frame.width == width && frame.height == height);
        let whole = !retained || !self.client_matches(width, height);
        let region = if whole {
            FrameRegion::whole(width, height)
        } else {
            region
        };
        if region.is_empty() {
            return Ok(());
        }
        let frame = self.state.frame.get_or_insert_with(|| PresentedFrame {
            width,
            height,
            pixels: Vec::new(),
        });
        if !retained {
            frame.pixels.clear();
            frame
                .pixels
                .try_reserve_exact(count)
                .map_err(|_| allocation_error())?;
            frame.width = width;
            frame.height = height;
            frame.pixels.resize(count, 0);
        }
        let stride = usize::try_from(width).map_err(|_| allocation_error())?;
        let fits = "invariant: a validated frame coordinate fits usize";
        let (left, right) = (
            usize::try_from(region.left).expect(fits),
            usize::try_from(region.right).expect(fits),
        );
        for row in region.top..region.bottom {
            let start = usize::try_from(row).expect(fits) * stride;
            frame.pixels[start + left..start + right]
                .copy_from_slice(&pixels[start + left..start + right]);
        }
        let rect = region.rect()?;
        // SAFETY: `self.hwnd` is owned by this thread, and `rect` is borrowed
        // only for the synchronous call.
        if !unsafe { InvalidateRect(Some(self.hwnd), Some(&rect), false) }.as_bool() {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Reports whether the client area shows the frame at one frame pixel per
    /// device pixel, so frame coordinates are client coordinates.
    fn client_matches(&self, width: u32, height: u32) -> bool {
        let mut client = RECT::default();
        // SAFETY: `client` is writable storage for this thread's live hwnd.
        if unsafe { GetClientRect(self.hwnd, &mut client) }.is_err() {
            return false;
        }
        i64::from(client.right - client.left) == i64::from(width)
            && i64::from(client.bottom - client.top) == i64::from(height)
    }
}

pub(super) unsafe fn paint(hwnd: HWND, state: &WindowState) -> LRESULT {
    unsafe {
        let mut paint = PAINTSTRUCT::default();
        // SAFETY: `paint` is writable storage and hwnd is the callback's live handle.
        let hdc = BeginPaint(hwnd, &mut paint);
        paint_frame(hwnd, state, hdc);
        // SAFETY: paint was initialized by BeginPaint and belongs to hwnd.
        let _ = EndPaint(hwnd, &paint);
        LRESULT(0)
    }
}

pub(super) unsafe fn paint_frame(hwnd: HWND, state: &WindowState, hdc: HDC) {
    unsafe {
        if hdc.is_invalid() {
            return;
        }
        let Some(frame) = state.frame.as_ref() else {
            return;
        };
        let mut client = RECT::default();
        // SAFETY: `client` is writable storage for this live hwnd.
        if GetClientRect(hwnd, &mut client).is_err() {
            return;
        }
        let dest_width = client.right.saturating_sub(client.left);
        let dest_height = client.bottom.saturating_sub(client.top);
        if dest_width <= 0 || dest_height <= 0 {
            return;
        }
        let info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: frame.width as i32,
                biHeight: -(frame.height as i32),
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                ..Default::default()
            },
            bmiColors: [RGBQUAD::default()],
        };
        // SAFETY: the retained frame remains borrowed for this synchronous
        // GDI call; BITMAPINFO matches the 32-bit row-major ARGB storage and
        // the destination is bounded by GetClientRect.
        let _ = StretchDIBits(
            hdc,
            0,
            0,
            dest_width,
            dest_height,
            0,
            0,
            frame.width as i32,
            frame.height as i32,
            Some(frame.pixels.as_ptr().cast::<c_void>()),
            &info,
            DIB_RGB_COLORS,
            SRCCOPY,
        );
    }
}
