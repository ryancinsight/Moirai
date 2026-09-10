//! Value and native-host tests for the window provider.

use super::config::validate_frame_dimensions;
use super::event::{CompositionPhase, ModifierState, MouseButton, WindowEvent};
use super::input::{
    extent_from_lparam, mouse_button, point_from_lparam, update_modifier, wheel_deltas,
};
use super::native::NativeWindow;
use super::state::WindowState;
use super::{WindowConfig, WindowVisibility};
use std::io;
use std::time::Duration;
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::Graphics::Gdi::ClientToScreen;
use windows::Win32::UI::Input::KeyboardAndMouse::{VK_CONTROL, VK_LWIN, VK_MENU};
use windows::Win32::UI::WindowsAndMessaging::{
    PostMessageW, SC_CLOSE, SendMessageW, WM_CHAR, WM_DPICHANGED, WM_IME_COMPOSITION,
    WM_IME_ENDCOMPOSITION, WM_IME_STARTCOMPOSITION, WM_KEYDOWN, WM_KEYUP, WM_LBUTTONDOWN,
    WM_LBUTTONUP, WM_MOUSEHWHEEL, WM_MOUSEMOVE, WM_MOUSEWHEEL, WM_SIZE, WM_SYSCOMMAND,
    WM_SYSKEYDOWN, WM_SYSKEYUP, WM_XBUTTONDOWN,
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
fn composition_decoding_is_bounded_and_preserves_unicode() {
    let mut state = WindowState::new().expect("bounded queue");
    state.push_composition(
        CompositionPhase::Updated,
        super::state::decode_composition(&[u16::from(b'A'), 0xd83d, 0xde00])
            .expect("UTF-16 fixture"),
    );
    assert_eq!(
        state.events.pop_front(),
        Some(WindowEvent::TextComposition {
            phase: CompositionPhase::Updated,
            text: "A😀".to_owned(),
        })
    );
    assert!(super::state::decode_composition(&[0xd800]).is_err());
    assert!(
        super::state::decode_composition(&vec![u16::from(b'x'); super::MAX_COMPOSITION_UNITS + 1])
            .is_err()
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
fn wheel_decoding_preserves_signed_axes_and_modifier_state() {
    let encode = |delta: i16, flags: usize| {
        WPARAM((usize::from(u16::from_ne_bytes(delta.to_ne_bytes())) << 16) | flags)
    };
    assert_eq!(
        wheel_deltas(WM_MOUSEWHEEL, encode(120, 0x000c)),
        Some((0, 120))
    );
    assert_eq!(
        wheel_deltas(WM_MOUSEHWHEEL, encode(-240, 0)),
        Some((-240, 0))
    );
    assert_eq!(wheel_deltas(WM_KEYDOWN, encode(120, 0)), None);

    let left_alt = key_lparam(0x38, false, true);
    let right_alt = key_lparam(0x38, true, true);
    let state = update_modifier(ModifierState::NONE, u32::from(VK_MENU.0), left_alt, true);
    let state = update_modifier(state, u32::from(VK_MENU.0), right_alt, true);
    let state = update_modifier(state, u32::from(VK_LWIN.0), LPARAM(0), true);
    let state = state.with_wheel_message_flags(0x000c);
    assert!(state.ctrl());
    assert!(state.shift());
    assert!(state.alt());
    assert!(state.meta());
    let state = update_modifier(state, u32::from(VK_MENU.0), left_alt, false);
    assert!(state.alt());
    assert!(state.meta());

    let left_control = key_lparam(0x1d, false, false);
    let right_control = key_lparam(0x1d, true, false);
    let state = update_modifier(
        ModifierState::NONE,
        u32::from(VK_CONTROL.0),
        left_control,
        true,
    );
    let state = update_modifier(state, u32::from(VK_CONTROL.0), right_control, true);
    let state = update_modifier(state, u32::from(VK_CONTROL.0), left_control, false);
    assert!(state.ctrl());
    let state = update_modifier(state, u32::from(VK_CONTROL.0), right_control, false);
    assert!(!state.ctrl());
}

fn key_lparam(scan_code: u8, extended: bool, context: bool) -> LPARAM {
    let mut raw = u64::from(scan_code) << 16;
    if extended {
        raw |= 1 << 24;
    }
    if context {
        raw |= 1 << 29;
    }
    LPARAM(raw as isize)
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
        PostMessageW(Some(window.hwnd), WM_MOUSEMOVE, WPARAM(0), LPARAM(0)).expect("pointer move");
        PostMessageW(
            Some(window.hwnd),
            WM_LBUTTONDOWN,
            WPARAM(0),
            LPARAM(((24_u32 << 16) | 16) as isize),
        )
        .expect("pointer down");
        PostMessageW(
            Some(window.hwnd),
            WM_LBUTTONUP,
            WPARAM(0),
            LPARAM(((24_u32 << 16) | 16) as isize),
        )
        .expect("pointer up");
        let mut wheel_point = windows::Win32::Foundation::POINT { x: 16, y: 24 };
        if !ClientToScreen(window.hwnd, &mut wheel_point).as_bool() {
            panic!("client point must convert to screen coordinates");
        }
        let wheel_lparam = LPARAM(
            ((u32::try_from(wheel_point.y).expect("test point is positive") << 16)
                | u32::try_from(wheel_point.x).expect("test point is positive"))
                as isize,
        );
        PostMessageW(
            Some(window.hwnd),
            WM_SYSKEYDOWN,
            WPARAM(usize::from(VK_MENU.0)),
            key_lparam(0x38, false, true),
        )
        .expect("Alt down");
        PostMessageW(
            Some(window.hwnd),
            WM_MOUSEWHEEL,
            WPARAM((usize::from(u16::from_ne_bytes(120_i16.to_ne_bytes())) << 16) | 0x000c),
            wheel_lparam,
        )
        .expect("vertical wheel");
        PostMessageW(
            Some(window.hwnd),
            WM_MOUSEHWHEEL,
            WPARAM(usize::from(u16::from_ne_bytes((-240_i16).to_ne_bytes())) << 16),
            wheel_lparam,
        )
        .expect("horizontal wheel");
        PostMessageW(
            Some(window.hwnd),
            WM_SYSKEYUP,
            WPARAM(usize::from(VK_MENU.0)),
            key_lparam(0x38, false, true),
        )
        .expect("Alt up");
        PostMessageW(
            Some(window.hwnd),
            WM_KEYDOWN,
            WPARAM(0x41),
            LPARAM(1_i32 as isize),
        )
        .expect("key down");
        PostMessageW(Some(window.hwnd), WM_KEYUP, WPARAM(0x41), LPARAM(0)).expect("key up");
        PostMessageW(Some(window.hwnd), WM_CHAR, WPARAM(0xd83d), LPARAM(0))
            .expect("high surrogate");
        PostMessageW(Some(window.hwnd), WM_CHAR, WPARAM(0xde00), LPARAM(0)).expect("low surrogate");
        let _ = SendMessageW(
            window.hwnd,
            WM_IME_STARTCOMPOSITION,
            Some(WPARAM(0)),
            Some(LPARAM(0)),
        );
        let _ = SendMessageW(
            window.hwnd,
            WM_IME_COMPOSITION,
            Some(WPARAM(0)),
            Some(LPARAM(0)),
        );
        let _ = SendMessageW(
            window.hwnd,
            WM_IME_ENDCOMPOSITION,
            Some(WPARAM(0)),
            Some(LPARAM(0)),
        );
        PostMessageW(
            Some(window.hwnd),
            WM_SIZE,
            WPARAM(0),
            LPARAM(((0x00f0_u32 << 16) | 0x0140) as isize),
        )
        .expect("resize");
        let _ = SendMessageW(
            window.hwnd,
            WM_DPICHANGED,
            Some(WPARAM(144)),
            Some(LPARAM(0)),
        );
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
    assert!(events.iter().any(|event| matches!(
        event,
        WindowEvent::PointerWheel {
            x: 16,
            y: 24,
            delta_x: 0,
            delta_y: 120,
            modifiers,
        } if modifiers.ctrl() && modifiers.shift() && modifiers.alt() && !modifiers.meta()
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        WindowEvent::PointerWheel {
            x: 16,
            y: 24,
            delta_x: -240,
            delta_y: 0,
            modifiers,
        } if !modifiers.ctrl() && !modifiers.shift() && modifiers.alt() && !modifiers.meta()
    )));
    assert!(events.contains(&WindowEvent::KeyDown {
        virtual_key: 0x41,
        repeated: false,
    }));
    assert!(events.contains(&WindowEvent::KeyUp { virtual_key: 0x41 }));
    assert!(events.contains(&WindowEvent::TextInput { character: '😀' }));
    assert!(events.contains(&WindowEvent::TextComposition {
        phase: CompositionPhase::Started,
        text: String::new(),
    }));
    assert!(events.contains(&WindowEvent::TextComposition {
        phase: CompositionPhase::Canceled,
        text: String::new(),
    }));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                WindowEvent::TextComposition {
                    phase: CompositionPhase::Canceled,
                    text,
                } if text.is_empty()
            ))
            .count(),
        1
    );
    assert!(events.contains(&WindowEvent::DpiChanged { dpi: 144 }));
    window.close().expect("destroy");
    assert!(window.is_destroyed());
}

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
