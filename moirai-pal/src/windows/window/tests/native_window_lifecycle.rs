//! End-to-end native HWND message round-trip: pointer, wheel, keyboard, IME,
//! resize, and DPI events decoded from a live window.

use super::super::event::{CompositionPhase, MouseButton, WindowEvent};
use super::super::native::NativeWindow;
use super::super::{WindowConfig, WindowVisibility};
use super::key_message_encoding::key_lparam;
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::Graphics::Gdi::ClientToScreen;
use windows::Win32::UI::Input::KeyboardAndMouse::{VK_CONTROL, VK_MENU, VK_SHIFT};
use windows::Win32::UI::WindowsAndMessaging::{
    PostMessageW, SendMessageW, WM_CHAR, WM_DPICHANGED, WM_IME_COMPOSITION, WM_IME_ENDCOMPOSITION,
    WM_IME_STARTCOMPOSITION, WM_KEYDOWN, WM_KEYUP, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MOUSEHWHEEL,
    WM_MOUSEMOVE, WM_MOUSEWHEEL, WM_SIZE, WM_SYSKEYDOWN, WM_SYSKEYUP,
};

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
            WPARAM(usize::from(VK_CONTROL.0)),
            key_lparam(0x1d, false, false),
        )
        .expect("Control down");
        PostMessageW(
            Some(window.hwnd),
            WM_KEYDOWN,
            WPARAM(usize::from(VK_SHIFT.0)),
            key_lparam(0x2a, false, false),
        )
        .expect("Shift down");
        PostMessageW(
            Some(window.hwnd),
            WM_KEYDOWN,
            WPARAM(0x41),
            LPARAM(1_i32 as isize),
        )
        .expect("key down");
        PostMessageW(Some(window.hwnd), WM_KEYUP, WPARAM(0x41), LPARAM(0)).expect("key up");
        PostMessageW(
            Some(window.hwnd),
            WM_KEYUP,
            WPARAM(usize::from(VK_SHIFT.0)),
            key_lparam(0x2a, false, false),
        )
        .expect("Shift up");
        PostMessageW(
            Some(window.hwnd),
            WM_KEYUP,
            WPARAM(usize::from(VK_CONTROL.0)),
            key_lparam(0x1d, false, false),
        )
        .expect("Control up");
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
    assert!(events.iter().any(|event| matches!(
        event,
        WindowEvent::KeyDown {
            virtual_key: 0x41,
            repeated: false,
            modifiers,
        } if modifiers.ctrl() && modifiers.shift() && !modifiers.alt() && !modifiers.meta()
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        WindowEvent::KeyUp {
            virtual_key: 0x41,
            modifiers,
        } if modifiers.ctrl() && modifiers.shift() && !modifiers.alt() && !modifiers.meta()
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        WindowEvent::KeyDown {
            virtual_key,
            modifiers,
            ..
        } if *virtual_key == u32::from(VK_CONTROL.0) && modifiers.ctrl()
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        WindowEvent::KeyUp {
            virtual_key,
            modifiers,
        } if *virtual_key == u32::from(VK_CONTROL.0) && !modifiers.ctrl()
    )));
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
