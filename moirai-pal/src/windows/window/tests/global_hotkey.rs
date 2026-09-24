//! System-wide hotkey validation, registration and press delivery.

use super::super::native::NativeWindow;
use super::super::{
    GlobalHotkey, HotkeyId, MAX_HOTKEY_ID, ModifierState, WindowConfig, WindowVisibility,
};
use std::io;
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{PostMessageW, WM_HOTKEY};

const VK_F24: u32 = 0x87;
const CHORD: ModifierState = ModifierState::CONTROL;

fn rare_chord() -> ModifierState {
    CHORD | ModifierState::ALT | ModifierState::SHIFT
}

#[test]
fn hotkey_values_are_validated() {
    assert!(HotkeyId::new(MAX_HOTKEY_ID).is_ok());
    assert!(HotkeyId::new(MAX_HOTKEY_ID + 1).is_err());
    assert!(GlobalHotkey::new(ModifierState::NONE, VK_F24).is_err());
    for modifier_key in [0x10, 0x11, 0x12, 0x5B, 0x5C, 0xA0, 0xA5] {
        assert!(GlobalHotkey::new(CHORD, modifier_key).is_err());
    }
    assert!(GlobalHotkey::new(CHORD, 0).is_err());
    assert!(GlobalHotkey::new(CHORD, 0xFF).is_err());
    let hotkey = GlobalHotkey::new(rare_chord(), VK_F24).expect("hotkey");
    assert_eq!(hotkey.virtual_key(), VK_F24);
    assert_eq!(hotkey.modifiers(), rare_chord());
}

#[test]
fn registered_hotkey_presses_are_delivered_until_released() {
    let config = WindowConfig::with_visibility("Moirai hotkey", 320, 240, WindowVisibility::Hidden)
        .expect("config");
    let mut window = NativeWindow::new(&config).expect("native window");
    let id = HotkeyId::new(7).expect("id");
    let hotkey = GlobalHotkey::new(rare_chord(), VK_F24).expect("hotkey");
    window.register_hotkey(id, hotkey).expect("register");
    let duplicate = window
        .register_hotkey(id, hotkey)
        .expect_err("duplicate id");
    assert_eq!(duplicate.kind(), io::ErrorKind::AlreadyExists);

    let post = |window: &NativeWindow, raw: usize| {
        // SAFETY: the message targets the live HWND owned by this test and
        // carries only scalar parameters.
        unsafe { PostMessageW(Some(window.hwnd), WM_HOTKEY, WPARAM(raw), LPARAM(0)) }
            .expect("post hotkey");
    };
    post(&window, 7);
    post(&window, 9);
    window.poll_events().expect("pump");
    assert_eq!(
        window.take_hotkey_presses(),
        vec![id],
        "unregistered ids are dropped"
    );
    assert!(window.take_hotkey_presses().is_empty());

    post(&window, 7);
    window.poll_events().expect("pump");
    assert!(window.unregister_hotkey(id).expect("unregister"));
    assert!(!window.unregister_hotkey(id).expect("second unregister"));
    assert!(window.take_hotkey_presses().is_empty());

    window.register_hotkey(id, hotkey).expect("register again");
    window.close().expect("close");
    assert!(window.register_hotkey(id, hotkey).is_err());
    let mut other = NativeWindow::new(&config).expect("second window");
    other
        .register_hotkey(id, hotkey)
        .expect("closing released the chord");
    other.close().expect("close");
}
