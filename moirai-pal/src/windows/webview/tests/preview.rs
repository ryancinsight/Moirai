//! Installed-runtime preview capture tests.

use std::{io, io::Cursor, sync::Barrier, thread};

use super::{
    NativeWindow, TestPackage, WebViewConfig, WebViewHost, WindowConfig, WindowVisibility,
};

const PAGE: &[u8] = br#"<!doctype html><meta charset="utf-8"><style>html,body{margin:0;width:100%;height:100%;background:#123456}</style><body>preview</body>"#;
const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

#[test]
#[ignore = "requires an installed WebView2 runtime"]
fn installed_runtime_captures_rendered_preview() {
    let package = TestPackage::create_with_script(PAGE);
    let host = preview_host(
        &package,
        "Moirai WebView2 preview test",
        WindowVisibility::Hidden,
    );
    let png = host
        .capture_preview_png()
        .expect("WebView2 preview capture");
    assert_rendered_page(&png);
}

#[test]
#[ignore = "requires an installed WebView2 runtime"]
fn installed_runtime_refuses_capture_while_hidden() {
    let package = TestPackage::create_with_script(PAGE);
    let mut host = preview_host(
        &package,
        "Moirai WebView2 hidden preview test",
        WindowVisibility::Hidden,
    );
    host.set_visible(false).expect("hide controller");
    // WebView2 holds a hidden controller's capture until it is shown again,
    // so an unguarded request ends at the finite wait instead.
    assert_eq!(
        host.capture_preview_png()
            .expect_err("a hidden controller has no frame to capture")
            .kind(),
        io::ErrorKind::InvalidInput
    );
}

#[test]
#[ignore = "requires an installed WebView2 runtime"]
fn installed_runtime_captures_immediately_after_reshow_concurrently() {
    let start = Barrier::new(101);
    thread::scope(|scope| {
        let attempts = (0..100)
            .map(|_| {
                scope.spawn(|| {
                    let package = TestPackage::create_with_script(PAGE);
                    // CapturePreview does not complete when its parent HWND
                    // lacks WS_VISIBLE (WebView2Feedback #579).
                    let mut host = preview_host(
                        &package,
                        "Moirai WebView2 re-show test",
                        WindowVisibility::Visible,
                    );
                    start.wait();
                    host.set_visible(false).expect("hide controller");
                    host.set_visible(true).expect("show rendered controller");
                    let png = host.capture_preview_png().expect("capture after re-show");
                    assert_rendered_page(&png);
                })
            })
            .collect::<Vec<_>>();
        start.wait();
        for attempt in attempts {
            attempt.join().expect("re-show capture thread");
        }
    });
}

fn preview_host(package: &TestPackage, title: &str, visibility: WindowVisibility) -> WebViewHost {
    let config = WebViewConfig::new(package.uri()).expect("packaged URI");
    let window_config = WindowConfig::with_visibility(title, WIDTH, HEIGHT, visibility)
        .expect("window configuration");
    let window = NativeWindow::new(&window_config).expect("native window");
    WebViewHost::new(window, config).expect("installed WebView2 host")
}

fn assert_rendered_page(encoded: &[u8]) {
    let mut decoder = png::Decoder::new(Cursor::new(encoded));
    decoder.set_transformations(
        png::Transformations::normalize_to_color8() | png::Transformations::ALPHA,
    );
    let mut reader = decoder.read_info().expect("valid PNG structure");
    let mut pixels = vec![
        0;
        reader
            .output_buffer_size()
            .expect("bounded preview dimensions")
    ];
    let output = reader.next_frame(&mut pixels).expect("complete PNG frame");
    let pixels = &pixels[..output.buffer_size()];
    assert_eq!((output.width, output.height), (WIDTH, HEIGHT));
    assert_eq!(
        (output.color_type, output.bit_depth),
        (png::ColorType::Rgba, png::BitDepth::Eight)
    );
    let expected = [0x12, 0x34, 0x56, 0xff];
    assert_eq!(pixels.get(..4), Some(expected.as_slice()));
    assert_eq!(
        pixels.get(pixels.len().checked_sub(4).expect("nonempty image")..),
        Some(expected.as_slice())
    );
}
