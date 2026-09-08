//! Value and native-host tests for the window provider.

use super::config::validate_frame_dimensions;
use super::event::{MouseButton, WindowEvent};
use super::input::{extent_from_lparam, mouse_button, point_from_lparam};
use super::native::NativeWindow;
use super::state::WindowState;
use super::{WindowConfig, WindowVisibility};
use std::io;
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    PostMessageW, SendMessageW, WM_CHAR, WM_DPICHANGED, WM_KEYDOWN, WM_KEYUP, WM_LBUTTONDOWN,
    WM_LBUTTONUP, WM_MOUSEMOVE, WM_SIZE, WM_XBUTTONDOWN,
};

#[test]
fn configuration_and_frame_limits_are_value_checked() {
    assert!(WindowConfig::new("Metis", 800, 600).is_ok());
    for title in ["", "bad\0title"] {
        assert_eq!(
            WindowConfig::new(title, 800, 600)
                .expect_err("invalid title")
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }
    assert!(WindowConfig::new("Metis", 0, 600).is_err());
    assert!(validate_frame_dimensions(4097, 4096).is_err());
    assert!(validate_frame_dimensions(800, 600).is_ok());
}

#[test]
fn utf16_pairing_preserves_scalars_and_rejects_unmatched_units() {
    let mut state = WindowState::new().expect("bounded queue");
    state.push_text_unit('A' as u16);
    state.push_text_unit(0xd83d);
    state.push_text_unit(0xde00);
    state.push_text_unit(0xd800);
    state.finish_text();
    assert_eq!(
        state.events.drain(..).collect::<Vec<_>>(),
        vec![
            WindowEvent::TextInput { character: 'A' },
            WindowEvent::TextInput { character: '😀' },
            WindowEvent::TextInput {
                character: '\u{fffd}'
            },
        ]
    );
}

#[test]
fn pointer_and_extent_decoding_uses_signed_client_coordinates() {
    let point = LPARAM(((0xfff6_u64) << 16 | 0xfffb) as isize);
    assert_eq!(point_from_lparam(point), (-5, -10));
    let extent = LPARAM(((0x0258_u64) << 16 | 0x0320) as isize);
    assert_eq!(extent_from_lparam(extent), (800, 600));
    assert_eq!(mouse_button(WM_XBUTTONDOWN, WPARAM(1_u16 as usize)), None);
    assert_eq!(
        mouse_button(WM_XBUTTONDOWN, WPARAM(1_usize << 16)),
        Some(MouseButton::X1)
    );
}

#[test]
#[cfg(windows)]
fn native_window_lifecycle_and_frame_round_trip() {
    let config = WindowConfig::with_visibility("Moirai test", 320, 240, WindowVisibility::Hidden)
        .expect("config");
    let mut window = NativeWindow::new(&config).expect("native window");
    assert!(!window.is_destroyed());
    let frame = vec![0xff12_3456_u32; 320 * 240];
    window
        .present_argb8888(320, 240, &frame)
        .expect("frame presentation");
    // SAFETY: every message targets the live HWND owned by this test and
    // carries only immediate scalar parameters.
    unsafe {
        PostMessageW(window.hwnd, WM_MOUSEMOVE, WPARAM(0), LPARAM(0)).expect("pointer move");
        PostMessageW(
            window.hwnd,
            WM_LBUTTONDOWN,
            WPARAM(0),
            LPARAM(((24_u32 << 16) | 16) as isize),
        )
        .expect("pointer down");
        PostMessageW(
            window.hwnd,
            WM_LBUTTONUP,
            WPARAM(0),
            LPARAM(((24_u32 << 16) | 16) as isize),
        )
        .expect("pointer up");
        PostMessageW(
            window.hwnd,
            WM_KEYDOWN,
            WPARAM(0x41),
            LPARAM(1_i32 as isize),
        )
        .expect("key down");
        PostMessageW(window.hwnd, WM_KEYUP, WPARAM(0x41), LPARAM(0)).expect("key up");
        PostMessageW(window.hwnd, WM_CHAR, WPARAM(0xd83d), LPARAM(0)).expect("high surrogate");
        PostMessageW(window.hwnd, WM_CHAR, WPARAM(0xde00), LPARAM(0)).expect("low surrogate");
        PostMessageW(
            window.hwnd,
            WM_SIZE,
            WPARAM(0),
            LPARAM(((0x00f0_u32 << 16) | 0x0140) as isize),
        )
        .expect("resize");
        let _ = SendMessageW(window.hwnd, WM_DPICHANGED, WPARAM(144), LPARAM(0));
    }
    let events = window.poll_events().expect("initial messages");
    assert!(events.iter().any(|event| matches!(
        event,
        WindowEvent::Resized {
            width: 320,
            height: 240
        }
    )));
    assert!(events.contains(&WindowEvent::PointerMove { x: 0, y: 0 }));
    assert!(events.contains(&WindowEvent::PointerDown {
        x: 16,
        y: 24,
        button: MouseButton::Left,
    }));
    assert!(events.contains(&WindowEvent::PointerUp {
        x: 16,
        y: 24,
        button: MouseButton::Left,
    }));
    assert!(events.contains(&WindowEvent::KeyDown {
        virtual_key: 0x41,
        repeated: false,
    }));
    assert!(events.contains(&WindowEvent::KeyUp { virtual_key: 0x41 }));
    assert!(events.contains(&WindowEvent::TextInput { character: '😀' }));
    assert!(events.contains(&WindowEvent::DpiChanged { dpi: 144 }));
    window.close().expect("destroy");
    assert!(window.is_destroyed());
}
