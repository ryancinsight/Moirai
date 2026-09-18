//! Native HWND system-key handling (Alt+F4 close) and blocking-wait input
//! delivery.

use super::super::event::WindowEvent;
use super::super::native::NativeWindow;
use super::super::{WindowConfig, WindowVisibility};
use super::key_message_encoding::key_lparam;
use std::time::Duration;
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::Graphics::Gdi::ClientToScreen;
use windows::Win32::UI::Input::KeyboardAndMouse::VK_MENU;
use windows::Win32::UI::WindowsAndMessaging::{
    PostMessageW, SC_CLOSE, SendMessageW, WM_MOUSEMOVE, WM_MOUSEWHEEL, WM_SYSCOMMAND,
    WM_SYSKEYDOWN, WM_SYSKEYUP,
};

#[test]
#[cfg(windows)]
fn native_system_keys_preserve_modifier_sides_and_alt_f4_close() {
    let config =
        WindowConfig::with_visibility("Moirai system-key test", 320, 240, WindowVisibility::Hidden)
            .expect("config");
    let mut window = NativeWindow::new(&config).expect("native window");
    let _ = window.poll_events().expect("initial events");

    // Generic VK_MENU messages carry the side in the scan code and extended
    // bit. Releasing the left key must leave the right Alt modifier active.
    let left_alt = key_lparam(0x38, false, true);
    let right_alt = key_lparam(0x38, true, true);
    // SAFETY: every message targets the live HWND owned by this test and
    // carries only immediate keyboard parameters.
    unsafe {
        let _ = SendMessageW(
            window.hwnd,
            WM_SYSKEYDOWN,
            Some(WPARAM(usize::from(VK_MENU.0))),
            Some(left_alt),
        );
        let _ = SendMessageW(
            window.hwnd,
            WM_SYSKEYDOWN,
            Some(WPARAM(usize::from(VK_MENU.0))),
            Some(right_alt),
        );
        let _ = SendMessageW(
            window.hwnd,
            WM_SYSKEYUP,
            Some(WPARAM(usize::from(VK_MENU.0))),
            Some(left_alt),
        );

        let mut wheel_point = windows::Win32::Foundation::POINT { x: 16, y: 24 };
        if !ClientToScreen(window.hwnd, &mut wheel_point).as_bool() {
            panic!("client point must convert to screen coordinates");
        }
        let wheel_lparam = LPARAM(
            ((u32::try_from(wheel_point.y).expect("test point is positive") << 16)
                | u32::try_from(wheel_point.x).expect("test point is positive"))
                as isize,
        );
        let _ = SendMessageW(
            window.hwnd,
            WM_MOUSEWHEEL,
            Some(WPARAM(
                (usize::from(u16::from_ne_bytes(120_i16.to_ne_bytes())) << 16) | 0x000c,
            )),
            Some(wheel_lparam),
        );

        // Releasing the right Alt leaves no modifier for the next wheel event.
        let _ = SendMessageW(
            window.hwnd,
            WM_SYSKEYUP,
            Some(WPARAM(usize::from(VK_MENU.0))),
            Some(right_alt),
        );
        let _ = SendMessageW(
            window.hwnd,
            WM_MOUSEWHEEL,
            Some(WPARAM(
                usize::from(u16::from_ne_bytes(120_i16.to_ne_bytes())) << 16,
            )),
            Some(wheel_lparam),
        );

        // User32 surfaces Alt+F4 as the top-level system close command. The
        // real HWND path must retain the default WM_SYSCOMMAND -> WM_CLOSE
        // behavior after the PAL records keyboard events.
        let _ = SendMessageW(
            window.hwnd,
            WM_SYSCOMMAND,
            Some(WPARAM(
                usize::try_from(SC_CLOSE).expect("SC_CLOSE fits in WPARAM"),
            )),
            Some(LPARAM(0)),
        );
    }

    let events = window.poll_events().expect("system-key events");
    assert!(events.iter().any(|event| matches!(
        event,
        WindowEvent::PointerWheel { modifiers, .. } if modifiers.alt()
    )));
    let mut wheels = events.iter().filter_map(|event| match event {
        WindowEvent::PointerWheel { modifiers, .. } => Some(modifiers.alt()),
        _ => None,
    });
    assert_eq!(wheels.next(), Some(true));
    assert_eq!(wheels.next(), Some(false));
    assert!(events.contains(&WindowEvent::CloseRequested));
    window.close().expect("destroy");
}

#[test]
#[cfg(windows)]
fn native_window_wait_returns_posted_input_without_busy_polling() {
    let config =
        WindowConfig::with_visibility("Moirai wait test", 320, 240, WindowVisibility::Hidden)
            .expect("config");
    let mut window = NativeWindow::new(&config).expect("native window");
    let error = window
        .wait_events(Duration::from_secs(31))
        .expect_err("bounded wait");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    let initial = window.wait_events(Duration::ZERO).expect("initial events");
    assert!(initial.iter().any(|event| matches!(
        event,
        WindowEvent::Resized {
            width: 320,
            height: 240
        }
    )));
    // SAFETY: the message targets the live HWND owned by this test and carries
    // only immediate scalar parameters.
    unsafe {
        PostMessageW(
            Some(window.hwnd),
            WM_MOUSEMOVE,
            WPARAM(0),
            LPARAM(((18_u32 << 16) | 0x000c) as isize),
        )
        .expect("pointer move");
    }
    let events = window
        .wait_events(Duration::from_secs(1))
        .expect("wait for posted input");
    assert!(events.contains(&WindowEvent::PointerMove { x: 12, y: 18 }));
    assert!(
        window
            .wait_events(Duration::ZERO)
            .expect("immediate wait")
            .is_empty()
    );
    window.close().expect("destroy");
}
