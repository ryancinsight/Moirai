//! Mouse capture across the real window procedure.
//!
//! The tests drive `window_proc` on a real HWND with `SendMessageW`, so the
//! assertions read the thread's actual capture owner, and the decision table
//! pins which button-state words end a capture.

use super::super::event::{MouseButton, WindowEvent};
use super::super::input::{CaptureTransition, capture_transition};
use super::super::native::NativeWindow;
use super::super::{WindowConfig, WindowVisibility};
use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::System::SystemServices::{
    MK_CONTROL, MK_LBUTTON, MK_MBUTTON, MK_RBUTTON, MK_SHIFT, MK_XBUTTON1, MK_XBUTTON2,
};
use windows::Win32::UI::Input::KeyboardAndMouse::GetCapture;
use windows::Win32::UI::WindowsAndMessaging::{
    SendMessageW, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEMOVE,
    WM_RBUTTONDOWN, WM_RBUTTONUP, WM_XBUTTONDOWN, WM_XBUTTONUP,
};

/// Packs signed client coordinates into the `LPARAM` word pair of a mouse message.
pub(super) fn pointer_lparam(x: i32, y: i32) -> LPARAM {
    let word = |value: i32| {
        u32::from(u16::from_ne_bytes(
            i16::try_from(value)
                .expect("test coordinates fit a signed word")
                .to_ne_bytes(),
        ))
    };
    LPARAM(isize::try_from((word(y) << 16) | word(x)).expect("a 32-bit pair fits isize"))
}

/// Builds a button-state word: `flags` in the low word, an X-button id above.
pub(super) fn button_wparam(flags: u32, x_button: u32) -> WPARAM {
    WPARAM(usize::try_from(flags | (x_button << 16)).expect("a 32-bit word fits usize"))
}

pub(super) fn send(window: &NativeWindow, message: u32, wparam: WPARAM, x: i32, y: i32) {
    // SAFETY: the message targets the live HWND owned by this test and carries
    // only immediate scalar parameters.
    unsafe {
        SendMessageW(
            window.hwnd,
            message,
            Some(wparam),
            Some(pointer_lparam(x, y)),
        )
    };
}

pub(super) fn capture_owner() -> HWND {
    // SAFETY: `GetCapture` reads the calling thread's input state only.
    unsafe { GetCapture() }
}

pub(super) fn hidden_window(title: &str) -> NativeWindow {
    let config =
        WindowConfig::with_visibility(title, 320, 240, WindowVisibility::Hidden).expect("config");
    let mut window = NativeWindow::new(&config).expect("native window");
    let _ = window.poll_events().expect("initial events");
    window
}

#[test]
fn capture_follows_the_held_buttons_and_the_outside_release_keeps_its_sign() {
    let mut window = hidden_window("Moirai capture test");
    assert!(capture_owner().is_invalid(), "no capture at start");

    send(
        &window,
        WM_LBUTTONDOWN,
        button_wparam(MK_LBUTTON.0, 0),
        10,
        20,
    );
    assert_eq!(capture_owner(), window.hwnd, "the first press captures");
    send(
        &window,
        WM_RBUTTONDOWN,
        button_wparam(MK_LBUTTON.0 | MK_RBUTTON.0, 0),
        30,
        40,
    );
    assert_eq!(
        capture_owner(),
        window.hwnd,
        "a second press keeps the capture"
    );

    send(
        &window,
        WM_LBUTTONUP,
        button_wparam(MK_RBUTTON.0, 0),
        -5,
        -10,
    );
    assert_eq!(capture_owner(), window.hwnd, "one of two buttons released");
    send(&window, WM_RBUTTONUP, button_wparam(0, 0), -7, -9);
    assert!(
        capture_owner().is_invalid(),
        "the last release ends capture"
    );

    assert_eq!(
        window.poll_events().expect("button events"),
        vec![
            WindowEvent::PointerDown {
                x: 10,
                y: 20,
                button: MouseButton::Left
            },
            WindowEvent::PointerDown {
                x: 30,
                y: 40,
                button: MouseButton::Right
            },
            WindowEvent::PointerUp {
                x: -5,
                y: -10,
                button: MouseButton::Left
            },
            WindowEvent::PointerUp {
                x: -7,
                y: -9,
                button: MouseButton::Right
            },
        ]
    );
    window.close().expect("destroy");
}

#[test]
fn every_button_captures_and_releases_on_its_own() {
    let mut window = hidden_window("Moirai button capture test");
    let buttons = [
        (
            WM_LBUTTONDOWN,
            WM_LBUTTONUP,
            MK_LBUTTON.0,
            0,
            MouseButton::Left,
        ),
        (
            WM_RBUTTONDOWN,
            WM_RBUTTONUP,
            MK_RBUTTON.0,
            0,
            MouseButton::Right,
        ),
        (
            WM_MBUTTONDOWN,
            WM_MBUTTONUP,
            MK_MBUTTON.0,
            0,
            MouseButton::Middle,
        ),
        (
            WM_XBUTTONDOWN,
            WM_XBUTTONUP,
            MK_XBUTTON1.0,
            1,
            MouseButton::X1,
        ),
        (
            WM_XBUTTONDOWN,
            WM_XBUTTONUP,
            MK_XBUTTON2.0,
            2,
            MouseButton::X2,
        ),
    ];
    for (down, up, flag, x_button, button) in buttons {
        send(&window, down, button_wparam(flag, x_button), 3, 4);
        assert_eq!(capture_owner(), window.hwnd, "{button:?} press captures");
        send(&window, up, button_wparam(0, x_button), -3, -4);
        assert!(
            capture_owner().is_invalid(),
            "{button:?} release ends capture"
        );
        assert_eq!(
            window.poll_events().expect("button events"),
            vec![
                WindowEvent::PointerDown { x: 3, y: 4, button },
                WindowEvent::PointerUp {
                    x: -3,
                    y: -4,
                    button
                },
            ]
        );
    }
    window.close().expect("destroy");
}

#[test]
fn motion_neither_captures_nor_releases() {
    let mut window = hidden_window("Moirai motion capture test");
    send(&window, WM_MOUSEMOVE, button_wparam(0, 0), 1, 2);
    assert!(capture_owner().is_invalid());
    send(
        &window,
        WM_LBUTTONDOWN,
        button_wparam(MK_LBUTTON.0, 0),
        1,
        2,
    );
    send(
        &window,
        WM_MOUSEMOVE,
        button_wparam(MK_LBUTTON.0, 0),
        -50,
        -60,
    );
    assert_eq!(
        capture_owner(),
        window.hwnd,
        "a drag outside keeps the capture"
    );
    send(&window, WM_LBUTTONUP, button_wparam(0, 0), -50, -60);
    assert!(capture_owner().is_invalid());
    assert_eq!(
        window.poll_events().expect("events"),
        vec![
            WindowEvent::PointerMove { x: 1, y: 2 },
            WindowEvent::PointerDown {
                x: 1,
                y: 2,
                button: MouseButton::Left
            },
            WindowEvent::PointerMove { x: -50, y: -60 },
            WindowEvent::PointerUp {
                x: -50,
                y: -60,
                button: MouseButton::Left
            },
        ],
        "motion keeps its signed client coordinates, outside the client area too"
    );
    window.close().expect("destroy");
}

#[test]
fn modifier_flags_do_not_count_as_held_buttons() {
    let flags = MK_CONTROL.0 | MK_SHIFT.0;
    assert_eq!(
        capture_transition(WM_LBUTTONUP, button_wparam(flags, 0)),
        Some(CaptureTransition::Release)
    );
    assert_eq!(
        capture_transition(WM_XBUTTONUP, button_wparam(flags, 2)),
        Some(CaptureTransition::Release),
        "the X-button id above the flag word is not a held button"
    );
}

#[test]
fn a_release_with_any_other_button_held_keeps_the_capture() {
    let held = [
        (WM_LBUTTONUP, MK_RBUTTON.0),
        (WM_RBUTTONUP, MK_MBUTTON.0),
        (WM_MBUTTONUP, MK_XBUTTON1.0),
        (WM_XBUTTONUP, MK_XBUTTON2.0),
        (WM_XBUTTONUP, MK_LBUTTON.0 | MK_CONTROL.0),
    ];
    for (up, flags) in held {
        assert_eq!(capture_transition(up, button_wparam(flags, 1)), None);
    }
}

#[test]
fn every_press_acquires_and_unrelated_messages_do_nothing() {
    for down in [
        WM_LBUTTONDOWN,
        WM_RBUTTONDOWN,
        WM_MBUTTONDOWN,
        WM_XBUTTONDOWN,
    ] {
        assert_eq!(
            capture_transition(down, button_wparam(0, 0)),
            Some(CaptureTransition::Acquire)
        );
    }
    assert_eq!(capture_transition(WM_MOUSEMOVE, button_wparam(0, 0)), None);
}
