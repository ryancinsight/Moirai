//! Region presents retain the new frame and repaint only the region.

use super::super::native::NativeWindow;
use super::super::{FrameRegion, WindowConfig, WindowVisibility};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Gdi::{GetUpdateRect, ValidateRect};
use windows::Win32::UI::WindowsAndMessaging::GetClientRect;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;

/// A visible window: Windows keeps no update region for a hidden one, and
/// the pending repaint rectangle is what these tests observe.
fn window() -> NativeWindow {
    let config = WindowConfig::with_visibility(
        "Moirai region test",
        WIDTH,
        HEIGHT,
        WindowVisibility::Visible,
    )
    .expect("window configuration");
    NativeWindow::new(&config).expect("native window")
}

/// The pending repaint rectangle, or `None` when nothing is invalid.
fn update_rect(window: &NativeWindow) -> Option<RECT> {
    let mut rect = RECT::default();
    // SAFETY: the test thread owns the live hwnd; `rect` is writable storage.
    unsafe { GetUpdateRect(window.hwnd, Some(&mut rect), false) }
        .as_bool()
        .then_some(rect)
}

/// The client area's size in device pixels, which display scaling can make
/// larger than the logical size the window was created with.
fn client_size(window: &NativeWindow) -> (u32, u32) {
    let mut client = RECT::default();
    // SAFETY: the test thread owns the live hwnd; `client` is writable storage.
    unsafe { GetClientRect(window.hwnd, &mut client) }.expect("client rectangle");
    let extent = |low: i32, high: i32| u32::try_from(high - low).expect("a nonnegative extent");
    (
        extent(client.left, client.right),
        extent(client.top, client.bottom),
    )
}

fn settle(window: &NativeWindow) {
    // SAFETY: the test thread owns the live hwnd; a null rectangle validates
    // the whole client area without painting.
    let _ = unsafe { ValidateRect(Some(window.hwnd), None) };
}

fn retained(window: &NativeWindow) -> Vec<u32> {
    window
        .state
        .frame
        .as_ref()
        .expect("a retained frame")
        .pixels
        .clone()
}

#[test]
fn a_region_present_copies_and_repaints_only_its_region() {
    let mut window = window();
    let (width, height) = client_size(&window);
    let stride = usize::try_from(width).expect("stride");
    let count = stride * usize::try_from(height).expect("rows");
    let first = vec![0xFF10_2030_u32; count];
    window
        .present_argb8888(width, height, &first)
        .expect("whole present");
    settle(&window);

    let mut second = first.clone();
    for row in 5..12 {
        for column in 7..30 {
            second[row * stride + column] = 0xFFA0_B0C0;
        }
    }
    let region = FrameRegion::new(7, 5, 30, 12).expect("ordered edges");
    window
        .present_argb8888_region(width, height, &second, region)
        .expect("region present");
    assert_eq!(retained(&window), second);
    assert_eq!(
        update_rect(&window),
        Some(RECT {
            left: 7,
            top: 5,
            right: 30,
            bottom: 12,
        })
    );

    settle(&window);
    let empty = FrameRegion::new(3, 3, 3, 9).expect("an empty region");
    window
        .present_argb8888_region(width, height, &first, empty)
        .expect("empty present");
    assert_eq!(
        update_rect(&window),
        None,
        "an empty region repaints nothing"
    );
    assert_eq!(retained(&window), second, "an empty region copies nothing");
    window.close().expect("close");
}

#[test]
fn new_dimensions_and_bad_regions_are_handled_whole_or_refused() {
    let mut window = window();
    let small = vec![0xFF00_0000_u32; 32 * 24];
    window
        .present_argb8888(32, 24, &small)
        .expect("frame smaller than the client");
    settle(&window);
    let region = FrameRegion::new(1, 1, 2, 2).expect("ordered edges");
    let grey = vec![0xFF80_8080_u32; 32 * 24];
    window
        .present_argb8888_region(32, 24, &grey, region)
        .expect("stretched present");
    assert_eq!(
        retained(&window),
        grey,
        "a stretched client copies the whole frame"
    );
    let pending = update_rect(&window).expect("a pending repaint");
    assert!(
        pending.right - pending.left > 1 && pending.bottom - pending.top > 1,
        "a stretched client repaints whole: {pending:?}"
    );

    let count = usize::try_from(WIDTH * HEIGHT).expect("pixel count");
    let outside = FrameRegion::new(0, 0, WIDTH + 1, HEIGHT).expect("ordered edges");
    let error = window
        .present_argb8888_region(WIDTH, HEIGHT, &vec![0; count], outside)
        .expect_err("a region past the frame");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    assert_eq!(
        FrameRegion::new(5, 0, 4, 1)
            .expect_err("reversed edges")
            .kind(),
        std::io::ErrorKind::InvalidInput
    );
    window.close().expect("close");
}
