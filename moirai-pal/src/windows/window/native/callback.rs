//! Native message translation and IME callback storage ownership.

use super::super::{
    accessibility::ACCESSIBILITY_WAKE_MESSAGE,
    config::{MAX_COMPOSITION_UNITS, allocation_error},
    event::{CompositionPhase, WindowEvent},
    input::{
        client_point_from_wheel_lparam, extent_from_lparam, mouse_button, point_from_lparam,
        wheel_deltas,
    },
    present::{paint, paint_frame},
    state::{SharedWindowState, WindowState, decode_composition},
    tray::TRAY_CALLBACK_MESSAGE,
};
use std::{ffi::c_void, io, mem::size_of};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::Graphics::Gdi::HDC;
use windows::Win32::UI::Input::Ime::{
    GCS_COMPSTR, GCS_RESULTSTR, ImmGetCompositionStringW, ImmGetContext, ImmReleaseContext,
};
use windows::Win32::UI::WindowsAndMessaging::{
    DefWindowProcW, GWLP_USERDATA, GetWindowLongPtrW, SetWindowLongPtrW, WM_CHAR, WM_CLOSE,
    WM_COMMAND, WM_DESTROY, WM_DPICHANGED, WM_ERASEBKGND, WM_HOTKEY, WM_IME_COMPOSITION,
    WM_IME_ENDCOMPOSITION, WM_IME_STARTCOMPOSITION, WM_KEYDOWN, WM_KEYUP, WM_KILLFOCUS,
    WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEHWHEEL, WM_MOUSEMOVE,
    WM_MOUSEWHEEL, WM_NCCREATE, WM_NCDESTROY, WM_PAINT, WM_PRINT, WM_PRINTCLIENT, WM_RBUTTONDOWN,
    WM_RBUTTONUP, WM_SETFOCUS, WM_SIZE, WM_SYSKEYDOWN, WM_SYSKEYUP, WM_XBUTTONDOWN, WM_XBUTTONUP,
};

pub(super) unsafe extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    unsafe {
        if message == WM_NCCREATE {
            // SAFETY: WM_NCCREATE supplies a CREATESTRUCTW pointer for this window;
            // null checks guard malformed caller data before reading it.
            let create = lparam.0 as *const windows::Win32::UI::WindowsAndMessaging::CREATESTRUCTW;
            if create.is_null() {
                return LRESULT(0);
            }
            let create_param = (*create).lpCreateParams.cast_const();
            if create_param.is_null() {
                return LRESULT(0);
            }
            // SAFETY: the creation parameter is the live state that
            // `NativeWindow::new_hidden` lent for the synchronous
            // CreateWindowExW call. The window takes its own count here and
            // returns it in WM_NCDESTROY.
            SetWindowLongPtrW(
                hwnd,
                GWLP_USERDATA,
                SharedWindowState::adopt_create_param(create_param),
            );
            // Default creation stores the caption supplied to CreateWindowExW.
            return DefWindowProcW(hwnd, message, wparam, lparam);
        }

        // SAFETY: GWLP_USERDATA is written only by WM_NCCREATE above for this HWND
        // and cleared in WM_NCDESTROY; zero means a window not created by this
        // module, or one whose state was already released.
        let Some(state) =
            SharedWindowState::borrow_userdata(GetWindowLongPtrW(hwnd, GWLP_USERDATA))
        else {
            return DefWindowProcW(hwnd, message, wparam, lparam);
        };
        // Every access below lends the state to a closure that makes no
        // native call: a message a native call delivers re-entrantly to this
        // procedure therefore always finds the state unborrowed.
        //
        // SendInput delivers a UTF-16 surrogate pair as two WM_CHAR messages
        // with keyboard and repaint messages between them. Keep the high
        // surrogate pending across that interleave so native emoji input
        // remains one scalar instead of being replaced before its low
        // surrogate arrives. Focus and IME transitions are the boundaries
        // that terminate an incomplete text unit.
        if matches!(
            message,
            WM_KILLFOCUS
                | WM_IME_STARTCOMPOSITION
                | WM_IME_COMPOSITION
                | WM_IME_ENDCOMPOSITION
                | WM_NCDESTROY
        ) {
            state.with(WindowState::finish_text);
        }
        match message {
            WM_CLOSE => state.with(|state| state.push(WindowEvent::CloseRequested)),
            WM_HOTKEY => state.with(|state| state.push_hotkey(wparam.0)),
            WM_COMMAND => state.with(|state| state.push_menu_command(wparam.0, lparam.0)),
            TRAY_CALLBACK_MESSAGE => state.with(|state| state.push_tray(wparam.0, lparam.0)),
            WM_DESTROY => {}
            WM_SETFOCUS => {
                state.with(|state| state.push(WindowEvent::FocusGained));
                #[cfg(feature = "webview2")]
                super::super::focus::forward(&state);
            }
            WM_KILLFOCUS => state.with(|state| {
                state.clear_modifiers();
                state.push(WindowEvent::FocusLost);
            }),
            WM_MOUSEMOVE => {
                let (x, y) = point_from_lparam(lparam);
                state.with(|state| state.push(WindowEvent::PointerMove { x, y }));
            }
            WM_LBUTTONDOWN | WM_RBUTTONDOWN | WM_MBUTTONDOWN | WM_XBUTTONDOWN => {
                let (x, y) = point_from_lparam(lparam);
                if let Some(button) = mouse_button(message, wparam) {
                    state.with(|state| state.push(WindowEvent::PointerDown { x, y, button }));
                }
            }
            WM_LBUTTONUP | WM_RBUTTONUP | WM_MBUTTONUP | WM_XBUTTONUP => {
                let (x, y) = point_from_lparam(lparam);
                if let Some(button) = mouse_button(message, wparam) {
                    state.with(|state| state.push(WindowEvent::PointerUp { x, y, button }));
                }
            }
            WM_MOUSEWHEEL | WM_MOUSEHWHEEL => {
                if let Some((delta_x, delta_y)) = wheel_deltas(message, wparam) {
                    let point = client_point_from_wheel_lparam(hwnd, lparam);
                    state.with(|state| match point {
                        Ok((x, y)) => {
                            let modifiers = state.modifiers.with_wheel_message_flags(wparam.0);
                            state.push(WindowEvent::PointerWheel {
                                x,
                                y,
                                delta_x,
                                delta_y,
                                modifiers,
                            });
                        }
                        Err(error) => state.record_error(error),
                    });
                }
            }
            message @ (WM_KEYDOWN | WM_SYSKEYDOWN) => {
                state.with(|state| push_key_event(state, wparam, lparam, true));
                // System-key messages carry Alt/menu and F10/F4 behavior that
                // DefWindowProcW must retain after the PAL records the value event.
                if message == WM_SYSKEYDOWN {
                    return DefWindowProcW(hwnd, message, wparam, lparam);
                }
            }
            message @ (WM_KEYUP | WM_SYSKEYUP) => {
                state.with(|state| push_key_event(state, wparam, lparam, false));
                if message == WM_SYSKEYUP {
                    return DefWindowProcW(hwnd, message, wparam, lparam);
                }
            }
            WM_CHAR => state.with(|state| state.push_text_unit(wparam.0 as u16)),
            WM_IME_STARTCOMPOSITION => state.with(|state| {
                state.push_composition(CompositionPhase::Started, String::new());
            }),
            WM_IME_COMPOSITION => {
                if let Err(error) = composition_message(hwnd, &state, lparam) {
                    state.with(|state| state.record_error(error));
                }
            }
            WM_IME_ENDCOMPOSITION => state.with(cancel_composition),
            WM_SIZE => {
                let (width, height) = extent_from_lparam(lparam);
                state.with(|state| state.push(WindowEvent::Resized { width, height }));
            }
            WM_DPICHANGED => {
                let dpi = (wparam.0 & 0xffff) as u32;
                if dpi != 0 {
                    state.with(|state| state.push(WindowEvent::DpiChanged { dpi }));
                }
            }
            WM_ERASEBKGND => return LRESULT(1),
            WM_PRINT => {
                // DefWindowProcW sends WM_PRINTCLIENT and WM_ERASEBKGND to this
                // window synchronously; no state is lent across the call.
                let result = DefWindowProcW(hwnd, message, wparam, lparam);
                let hdc = HDC(wparam.0 as *mut c_void);
                // SAFETY: WM_PRINT supplies a live destination HDC for the
                // synchronous full-window render.
                paint_frame(hwnd, &state, hdc);
                return result;
            }
            WM_PRINTCLIENT => {
                let hdc = HDC(wparam.0 as *mut c_void);
                // SAFETY: WM_PRINTCLIENT supplies a live destination HDC for
                // the synchronous client-area render.
                paint_frame(hwnd, &state, hdc);
            }
            WM_PAINT => {
                // SAFETY: the callback owns the live HWND for the duration of
                // the synchronous paint operation.
                paint(hwnd, &state);
            }
            WM_NCDESTROY => {
                state.with(|state| state.push(WindowEvent::Destroyed));
                // SAFETY: clearing the module-owned userdata before returning from
                // WM_NCDESTROY prevents later messages from observing stale state.
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
                // Release the caption and other default nonclient resources.
                let result = DefWindowProcW(hwnd, message, wparam, lparam);
                // The last message this window receives: return the count that
                // WM_NCCREATE adopted. `state` is not used afterwards.
                SharedWindowState::release_userdata(state);
                return result;
            }
            ACCESSIBILITY_WAKE_MESSAGE => {}
            _ => return DefWindowProcW(hwnd, message, wparam, lparam),
        }
        LRESULT(0)
    }
}

fn cancel_composition(state: &mut WindowState) {
    if state.composition_active {
        state.push_composition(CompositionPhase::Canceled, String::new());
    }
}

#[inline(always)]
fn push_key_event(state: &mut WindowState, wparam: WPARAM, lparam: LPARAM, pressed: bool) {
    let virtual_key = wparam.0 as u32;
    state.update_modifier(virtual_key, lparam, pressed);
    let modifiers = state.modifiers;
    if pressed {
        state.push(WindowEvent::KeyDown {
            virtual_key,
            repeated: (lparam.0 & (1 << 30)) != 0,
            modifiers,
        });
    } else {
        state.push(WindowEvent::KeyUp {
            virtual_key,
            modifiers,
        });
    }
}

fn composition_message(hwnd: HWND, state: &SharedWindowState, lparam: LPARAM) -> io::Result<()> {
    let flags = u32::try_from(lparam.0).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "native IME composition flags are negative",
        )
    })?;
    let (phase, kind) = if flags & GCS_RESULTSTR.0 != 0 {
        (CompositionPhase::Committed, GCS_RESULTSTR)
    } else if flags & GCS_COMPSTR.0 != 0 {
        (CompositionPhase::Updated, GCS_COMPSTR)
    } else {
        state.with(cancel_composition);
        return Ok(());
    };
    let text = read_composition_text(hwnd, kind)?;
    state.with(|state| state.push_composition(phase, text));
    Ok(())
}

fn read_composition_text(
    hwnd: HWND,
    kind: windows::Win32::UI::Input::Ime::IME_COMPOSITION_STRING,
) -> io::Result<String> {
    // SAFETY: `hwnd` is the live window whose callback is executing; the IME
    // context is acquired and released synchronously on the owning thread.
    let context = unsafe { ImmGetContext(hwnd) };
    if context.is_invalid() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "native IME composition context is unavailable",
        ));
    }
    let result = read_composition_buffer(context, kind);
    // SAFETY: `context` was returned for `hwnd` by ImmGetContext and is released
    // on the same thread before this callback returns.
    let released = unsafe { ImmReleaseContext(hwnd, context) };
    if !released.as_bool() {
        return Err(io::Error::last_os_error());
    }
    result
}

fn read_composition_buffer(
    context: windows::Win32::UI::Input::Ime::HIMC,
    kind: windows::Win32::UI::Input::Ime::IME_COMPOSITION_STRING,
) -> io::Result<String> {
    // SAFETY: the IME context is valid for this synchronous query and the null
    // destination requests only the required byte count.
    let byte_count = unsafe { ImmGetCompositionStringW(context, kind, None, 0) };
    if byte_count < 0 {
        return Err(io::Error::other(
            "native IME composition string query failed",
        ));
    }
    let byte_count = usize::try_from(byte_count).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "native IME composition length is not representable",
        )
    })?;
    if byte_count % size_of::<u16>() != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "native IME composition length is not UTF-16 aligned",
        ));
    }
    let units = byte_count / size_of::<u16>();
    if units > MAX_COMPOSITION_UNITS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "native IME composition exceeds the bounded UTF-16 limit",
        ));
    }
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(units)
        .map_err(|_| allocation_error())?;
    buffer.resize(units, 0);
    if byte_count != 0 {
        let length = u32::try_from(byte_count).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "native IME composition length exceeds the API bound",
            )
        })?;
        // SAFETY: `buffer` has exactly `units` initialized `u16` slots, and the
        // IME API writes exactly `length` bytes into that writable allocation.
        let read = unsafe {
            ImmGetCompositionStringW(context, kind, Some(buffer.as_mut_ptr().cast()), length)
        };
        if read < 0 || usize::try_from(read).ok() != Some(byte_count) {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "native IME composition changed during retrieval",
            ));
        }
    }
    decode_composition(&buffer)
}
