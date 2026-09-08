//! Win32 input parameter decoding.

use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MBUTTONDOWN, WM_MBUTTONUP, WM_RBUTTONDOWN, WM_RBUTTONUP,
    WM_XBUTTONDOWN, WM_XBUTTONUP,
};

use super::event::MouseButton;

pub(super) fn mouse_button(message: u32, wparam: WPARAM) -> Option<MouseButton> {
    match message {
        WM_LBUTTONDOWN | WM_LBUTTONUP => Some(MouseButton::Left),
        WM_RBUTTONDOWN | WM_RBUTTONUP => Some(MouseButton::Right),
        WM_MBUTTONDOWN | WM_MBUTTONUP => Some(MouseButton::Middle),
        WM_XBUTTONDOWN | WM_XBUTTONUP => match ((wparam.0 >> 16) & 0xffff) as u16 {
            1 => Some(MouseButton::X1),
            2 => Some(MouseButton::X2),
            _ => None,
        },
        _ => None,
    }
}

pub(super) fn point_from_lparam(lparam: LPARAM) -> (i32, i32) {
    let raw = lparam.0 as u64;
    (
        i32::from(i16::from_ne_bytes([
            (raw & 0xff) as u8,
            ((raw >> 8) & 0xff) as u8,
        ])),
        i32::from(i16::from_ne_bytes([
            ((raw >> 16) & 0xff) as u8,
            ((raw >> 24) & 0xff) as u8,
        ])),
    )
}

pub(super) fn extent_from_lparam(lparam: LPARAM) -> (u32, u32) {
    let raw = lparam.0 as u64;
    ((raw & 0xffff) as u32, ((raw >> 16) & 0xffff) as u32)
}
