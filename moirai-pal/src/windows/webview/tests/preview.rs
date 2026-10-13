//! Installed-runtime preview capture tests.

use std::{
    io,
    io::Cursor,
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

use super::{
    NativeWindow, TestPackage, WebViewConfig, WebViewEvent, WebViewHost, WebViewHostEvent,
    WindowConfig, WindowVisibility,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

#[test]
#[ignore = "requires an installed WebView2 runtime"]
fn installed_runtime_captures_rendered_preview() {
    let package = TestPackage::create_with_script(&marker_page("preview", [0x12, 0x34, 0x56]));
    let host = preview_host(
        &package,
        "Moirai WebView2 preview test",
        WindowVisibility::Hidden,
    );
    let png = host
        .capture_preview_png()
        .expect("WebView2 preview capture");
    assert_rendered_page(&png, [0x12, 0x34, 0x56]);
}

#[test]
#[ignore = "requires an installed WebView2 runtime"]
fn installed_runtime_refuses_capture_while_hidden() {
    let package = TestPackage::create_with_script(&marker_page("hidden", [0x12, 0x34, 0x56]));
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
    run_concurrent_reshow_capture(NavigationTiming::WhileHidden);
}

#[test]
#[ignore = "requires an installed WebView2 runtime"]
fn installed_runtime_captures_after_settled_navigation_and_reshow_concurrently() {
    run_concurrent_reshow_capture(NavigationTiming::BeforeHide);
}

#[derive(Clone, Copy)]
enum NavigationTiming {
    BeforeHide,
    WhileHidden,
}

fn run_concurrent_reshow_capture(navigation_timing: NavigationTiming) {
    let start = Arc::new(Barrier::new(101));
    thread::scope(|scope| {
        let attempts = (0..100)
            .map(|host_index| {
                let start = Arc::clone(&start);
                scope.spawn(move || {
                    let initial = marker_color(host_index, 0);
                    let expected = marker_color(host_index, 1);
                    let package = TestPackage::create_with_script(&marker_page(
                        &format!("host-{host_index}-initial"),
                        initial,
                    ));
                    let reshown_uri = write_page(
                        &package,
                        "reshown.html",
                        &marker_page(&format!("host-{host_index}-reshown"), expected),
                    );
                    // CapturePreview does not complete when its parent HWND
                    // lacks WS_VISIBLE (WebView2Feedback #579).
                    let mut host = preview_host(
                        &package,
                        &format!("Moirai WebView2 re-show test {host_index}"),
                        WindowVisibility::Visible,
                    );
                    if matches!(navigation_timing, NavigationTiming::BeforeHide) {
                        host.navigate(reshown_uri.clone())
                            .expect("navigate before hide");
                        wait_for_navigation(&mut host);
                    }
                    start.wait();
                    host.set_visible(false).expect("hide controller");
                    if matches!(navigation_timing, NavigationTiming::WhileHidden) {
                        host.navigate(reshown_uri).expect("navigate while hidden");
                    }
                    host.set_visible(true).expect("show rendered controller");
                    if matches!(navigation_timing, NavigationTiming::WhileHidden) {
                        wait_for_navigation(&mut host);
                    }
                    let png = host.capture_preview_png().expect("capture after re-show");
                    assert_rendered_page(&png, expected);
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

fn marker_page(label: &str, color: [u8; 3]) -> Vec<u8> {
    format!(
        "<!doctype html><meta charset=\"utf-8\"><style>html,body{{margin:0;width:100%;height:100%;background:#{:02x}{:02x}{:02x}}}</style><body>{label}</body>",
        color[0], color[1], color[2]
    )
    .into_bytes()
}

fn marker_color(host_index: usize, phase: u8) -> [u8; 3] {
    let index = u8::try_from(host_index).expect("host marker index fits in a byte");
    [0x20 + index, 0x40 + phase * 0x30, 0x80 + index]
}

fn write_page(package: &TestPackage, name: &str, contents: &[u8]) -> String {
    let path = package.directory.join(name);
    std::fs::write(&path, contents).expect("write preview page");
    let path = path.canonicalize().expect("preview page path");
    super::file_uri(&path)
}

fn wait_for_navigation(host: &mut WebViewHost) {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "re-show navigation did not complete");
        let events = host
            .wait_events(remaining)
            .expect("re-show navigation event");
        if events.iter().any(|event| {
            matches!(
                event,
                WebViewHostEvent::WebView(WebViewEvent::NavigationCompleted { success: true, .. })
            )
        }) {
            return;
        }
    }
}

fn assert_rendered_page(encoded: &[u8], expected_rgb: [u8; 3]) {
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
    let expected = [expected_rgb[0], expected_rgb[1], expected_rgb[2], 0xff];
    assert_eq!(pixels.get(..4), Some(expected.as_slice()));
    assert_eq!(
        pixels.get(pixels.len().checked_sub(4).expect("nonempty image")..),
        Some(expected.as_slice())
    );
}
