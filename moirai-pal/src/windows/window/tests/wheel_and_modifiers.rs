//! Wheel-delta decoding and modifier-key side tracking.

use super::super::event::ModifierState;
use super::super::input::{update_modifier, wheel_deltas};
use super::key_message_encoding::key_lparam;
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::UI::Input::KeyboardAndMouse::{VK_CONTROL, VK_LWIN, VK_MENU, VK_SHIFT};
use windows::Win32::UI::WindowsAndMessaging::{WM_KEYDOWN, WM_MOUSEHWHEEL, WM_MOUSEWHEEL};

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

#[test]
fn public_constructors_report_exactly_the_keys_they_name() {
    let keys = [
        ModifierState::CONTROL,
        ModifierState::SHIFT,
        ModifierState::ALT,
        ModifierState::META,
    ];
    let report = |state: ModifierState| [state.ctrl(), state.shift(), state.alt(), state.meta()];
    for (index, key) in keys.into_iter().enumerate() {
        let expected: [bool; 4] = core::array::from_fn(|slot| slot == index);
        assert_eq!(report(key), expected, "{key:?}");
    }
    assert_eq!(report(ModifierState::NONE), [false; 4]);
    let chord = ModifierState::CONTROL | ModifierState::SHIFT;
    assert_eq!(report(chord), [true, true, false, false]);
    let mut state = ModifierState::NONE;
    state |= ModifierState::ALT;
    state |= ModifierState::META;
    assert_eq!(report(state), [false, false, true, true]);
    // A constructed chord equals the state the message decoder records for
    // the same left-hand keys.
    let left_shift = key_lparam(0x2a, false, false);
    let decoded = update_modifier(ModifierState::NONE, u32::from(VK_SHIFT.0), left_shift, true);
    assert_eq!(decoded, ModifierState::SHIFT);
}
