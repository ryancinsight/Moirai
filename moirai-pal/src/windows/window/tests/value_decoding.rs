//! Value validation for window configuration, text, and pointer decoding.

use super::super::WindowConfig;
use super::super::config::validate_frame_dimensions;
use super::super::event::{CompositionPhase, MouseButton, WindowEvent};
use super::super::input::{extent_from_lparam, mouse_button, point_from_lparam};
use super::super::state::WindowState;
use std::io;
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::WM_XBUTTONDOWN;

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
        super::super::state::decode_composition(&[u16::from(b'A'), 0xd83d, 0xde00])
            .expect("UTF-16 fixture"),
    );
    assert_eq!(
        state.events.pop_front(),
        Some(WindowEvent::TextComposition {
            phase: CompositionPhase::Updated,
            text: "A😀".to_owned(),
        })
    );
    assert!(super::super::state::decode_composition(&[0xd800]).is_err());
    assert!(
        super::super::state::decode_composition(&vec![
            u16::from(b'x');
            super::super::MAX_COMPOSITION_UNITS + 1
        ])
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
