//! Win32 input parameter decoding.

use std::io;

use windows::Win32::Foundation::{HWND, LPARAM, POINT, WPARAM};
use windows::Win32::Graphics::Gdi::ScreenToClient;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    VK_CONTROL, VK_LCONTROL, VK_LMENU, VK_LSHIFT, VK_LWIN, VK_MENU, VK_RCONTROL, VK_RMENU,
    VK_RSHIFT, VK_RWIN, VK_SHIFT,
};
use windows::Win32::UI::WindowsAndMessaging::{
    WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEHWHEEL, WM_MOUSEWHEEL,
    WM_RBUTTONDOWN, WM_RBUTTONUP, WM_XBUTTONDOWN, WM_XBUTTONUP,
};

use super::event::{
    ALT_BITS, ALT_LEFT, ALT_RIGHT, CONTROL_BITS, CONTROL_LEFT, CONTROL_RIGHT, META_LEFT,
    META_RIGHT, ModifierState, MouseButton, SHIFT_BITS, SHIFT_LEFT, SHIFT_RIGHT,
};

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

pub(super) fn wheel_deltas(message: u32, wparam: WPARAM) -> Option<(i16, i16)> {
    // The high word is a signed 16-bit value by the Win32 message contract;
    // the mask isolates that wire field before its intentional bit-preserving
    // conversion.
    let raw = ((wparam.0 >> 16) & 0xffff) as u16;
    let delta = i16::from_ne_bytes(raw.to_ne_bytes());
    match message {
        WM_MOUSEWHEEL => Some((0, delta)),
        WM_MOUSEHWHEEL => Some((delta, 0)),
        _ => None,
    }
}

pub(super) fn client_point_from_wheel_lparam(hwnd: HWND, lparam: LPARAM) -> io::Result<(i32, i32)> {
    let (x, y) = point_from_lparam(lparam);
    let mut point = POINT { x, y };
    // SAFETY: `point` is writable storage owned by this call and `hwnd` is the
    // live window whose callback is translating the message synchronously.
    if unsafe { ScreenToClient(hwnd, &mut point) }.as_bool() {
        Ok((point.x, point.y))
    } else {
        Err(io::Error::last_os_error())
    }
}

pub(super) fn modifier_for_key(virtual_key: u32) -> u8 {
    match virtual_key {
        key if key == u32::from(VK_CONTROL.0) => CONTROL_BITS,
        key if key == u32::from(VK_LCONTROL.0) => CONTROL_LEFT,
        key if key == u32::from(VK_RCONTROL.0) => CONTROL_RIGHT,
        key if key == u32::from(VK_SHIFT.0) => SHIFT_BITS,
        key if key == u32::from(VK_LSHIFT.0) => SHIFT_LEFT,
        key if key == u32::from(VK_RSHIFT.0) => SHIFT_RIGHT,
        key if key == u32::from(VK_MENU.0) => ALT_BITS,
        key if key == u32::from(VK_LMENU.0) => ALT_LEFT,
        key if key == u32::from(VK_RMENU.0) => ALT_RIGHT,
        key if key == u32::from(VK_LWIN.0) => META_LEFT,
        key if key == u32::from(VK_RWIN.0) => META_RIGHT,
        _ => 0,
    }
}

pub(super) fn update_modifier(
    state: ModifierState,
    virtual_key: u32,
    pressed: bool,
) -> ModifierState {
    state.set_bits(modifier_for_key(virtual_key), pressed)
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
