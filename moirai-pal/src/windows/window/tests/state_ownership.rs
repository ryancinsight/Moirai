//! The callback state is shared without aliasing: the window holds its own
//! count until destruction, and re-entrant paint messages neither observe a
//! borrowed state nor lose the retained frame.

use super::super::native::NativeWindow;
use super::super::{WindowConfig, WindowVisibility};
use std::ffi::c_void;
use windows::Win32::Foundation::{LPARAM, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, CreateCompatibleDC, CreateDIBSection, DIB_RGB_COLORS,
    DeleteDC, DeleteObject, HGDIOBJ, RGBQUAD, SelectObject,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetClientRect, SendMessageW, WM_PRINT, WM_PRINTCLIENT,
};

const PRF_CLIENT: isize = 0x4;
const FILL: u32 = 0xFF10_2030;
const RGB_MASK: u32 = 0x00FF_FFFF;

fn window() -> NativeWindow {
    let config =
        WindowConfig::with_visibility("Moirai state test", 64, 48, WindowVisibility::Visible)
            .expect("window configuration");
    NativeWindow::new(&config).expect("native window")
}

fn client_size(window: &NativeWindow) -> (usize, usize) {
    let mut client = RECT::default();
    // SAFETY: the test thread owns the live hwnd; `client` is writable storage.
    unsafe { GetClientRect(window.hwnd, &mut client) }.expect("client rectangle");
    (
        usize::try_from(client.right - client.left).expect("a nonnegative width"),
        usize::try_from(client.bottom - client.top).expect("a nonnegative height"),
    )
}

/// Sends `message` to the window with a zero-initialized memory surface the
/// size of its client area and returns the surface afterwards.
fn print_into_surface(window: &NativeWindow, message: u32, flags: isize) -> Vec<u32> {
    let (width, height) = client_size(window);
    let info = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: i32::try_from(width).expect("width fits"),
            biHeight: -i32::try_from(height).expect("height fits"),
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        },
        bmiColors: [RGBQUAD::default()],
    };
    let mut bits: *mut c_void = std::ptr::null_mut();
    // SAFETY: `info` describes a top-down 32-bit surface; `bits` receives the
    // section's pixel pointer, which stays valid until the bitmap is deleted.
    let bitmap = unsafe { CreateDIBSection(None, &info, DIB_RGB_COLORS, &mut bits, None, 0) }
        .expect("device-independent surface");
    // SAFETY: the compatible context and the selection are released below in
    // reverse order on this thread; the message is synchronous.
    unsafe {
        let context = CreateCompatibleDC(None);
        let previous = SelectObject(context, HGDIOBJ(bitmap.0));
        let _ = SendMessageW(
            window.hwnd,
            message,
            Some(WPARAM(context.0 as usize)),
            Some(LPARAM(flags)),
        );
        // SAFETY: `bits` points at `width * height` initialized u32 pixels
        // owned by `bitmap`, which is still selected and alive.
        let pixels = std::slice::from_raw_parts(bits.cast::<u32>(), width * height).to_vec();
        SelectObject(context, previous);
        let _ = DeleteDC(context);
        let _ = DeleteObject(HGDIOBJ(bitmap.0));
        pixels
    }
}

fn present_uniform_frame(window: &mut NativeWindow) {
    let (width, height) = client_size(window);
    let frame = vec![FILL; width * height];
    window
        .present_argb8888(
            u32::try_from(width).expect("width fits"),
            u32::try_from(height).expect("height fits"),
            &frame,
        )
        .expect("whole present");
}

fn assert_painted(pixels: &[u32]) {
    assert!(!pixels.is_empty());
    for (index, pixel) in pixels.iter().enumerate() {
        assert_eq!(
            pixel & RGB_MASK,
            FILL & RGB_MASK,
            "pixel {index} shows the retained frame"
        );
    }
}

#[test]
fn the_window_holds_its_own_state_count_until_it_is_destroyed() {
    let mut window = window();
    let shared = window.state.clone();
    assert_eq!(
        shared.holders(),
        3,
        "the test clone, the NativeWindow and the HWND each hold a count"
    );
    window.close().expect("close");
    assert_eq!(
        shared.holders(),
        2,
        "WM_NCDESTROY returned the window's count"
    );
    drop(window);
    assert_eq!(shared.holders(), 1);
}

#[test]
fn a_print_client_message_paints_the_frame_and_keeps_it_retained() {
    let mut window = window();
    present_uniform_frame(&mut window);
    assert_painted(&print_into_surface(&window, WM_PRINTCLIENT, PRF_CLIENT));
    assert!(
        window.state.with(|state| state.frame.is_some()),
        "painting restores the frame it moved out for the GDI call"
    );
    assert_painted(&print_into_surface(&window, WM_PRINTCLIENT, PRF_CLIENT));
    window.close().expect("close");
}

#[test]
fn a_print_message_that_re_enters_the_procedure_paints_twice_without_losing_the_frame() {
    let mut window = window();
    present_uniform_frame(&mut window);
    // DefWindowProc handles PRF_CLIENT by sending WM_PRINTCLIENT to the same
    // window while the outer WM_PRINT is still being handled.
    assert_painted(&print_into_surface(&window, WM_PRINT, PRF_CLIENT));
    assert!(window.state.with(|state| state.frame.is_some()));
    window.close().expect("close");
}
