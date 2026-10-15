//! Native inner focus routing and callback-state lifetime for the private HWND.

use std::{io, rc::Rc};
use windows::Win32::{
    Foundation::{GetLastError, HWND, LPARAM, LRESULT, SetLastError, WIN32_ERROR, WPARAM},
    UI::{
        Input::KeyboardAndMouse::{GetFocus, SetFocus},
        WindowsAndMessaging::{
            CREATESTRUCTW, DefWindowProcW, GW_CHILD, GWLP_USERDATA, GetWindow, GetWindowLongPtrW,
            SetWindowLongPtrW, WM_NCCREATE, WM_NCDESTROY, WM_SETFOCUS,
        },
    },
};

use super::{State, context};

pub(super) fn focus_child(hwnd: HWND, state: &State) -> io::Result<()> {
    // Only current native entry to this private container admits forwarding.
    // Native siblings, foreign foreground changes and stale entry do no work.
    if unsafe { GetFocus() } != hwnd {
        return Ok(());
    }
    if state.forwarding.replace(true) {
        return Ok(());
    }
    // One synchronous transfer owns nested native focus notifications. There
    // is no queued retry, SDK LostFocus handler or self-rearming episode.
    let result = (|| {
        // SAFETY: this exclusively owned parent has only the one controller's
        // subtree. No runtime-owned HWND survives this immediate operation.
        let child = unsafe { GetWindow(hwnd, GW_CHILD) }
            .map_err(|e| context("controller container first child", e))?;
        unsafe { SetFocus(Some(child)) }
            .map_err(|e| context("controller container native focus", e))?;
        Ok(())
    })();
    state.forwarding.set(false);
    result
}

pub(super) unsafe extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    unsafe {
        if message == WM_NCCREATE {
            // SAFETY: Win32 supplies CREATESTRUCTW synchronously. Its creation
            // parameter is a live Rc<State> allocation lent by new().
            let create = lparam.0 as *const CREATESTRUCTW;
            if create.is_null() {
                return LRESULT(0);
            }
            let pointer = (*create).lpCreateParams.cast::<State>();
            if pointer.is_null() {
                return LRESULT(0);
            }
            Rc::increment_strong_count(pointer);
            SetLastError(WIN32_ERROR(0));
            let previous = SetWindowLongPtrW(hwnd, GWLP_USERDATA, pointer as isize);
            let error = GetLastError();
            if previous == 0 && error.0 != 0 {
                Rc::decrement_strong_count(pointer);
                SetLastError(error);
                return LRESULT(0);
            }
            (*pointer).alive.set(true);
            let result = DefWindowProcW(hwnd, message, wparam, lparam);
            if result.0 == 0 && (*pointer).alive.replace(false) {
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
                Rc::decrement_strong_count(pointer);
            }
            return result;
        }
        let pointer = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const State;
        if pointer.is_null() {
            return DefWindowProcW(hwnd, message, wparam, lparam);
        }
        // SAFETY: the HWND owns a count until WM_NCDESTROY. This additional
        // local count protects native reentrancy, including nested destruction.
        Rc::increment_strong_count(pointer);
        let state = Rc::from_raw(pointer);
        if message == WM_NCDESTROY {
            state.ready.set(false);
            state.alive.set(false);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
            let result = DefWindowProcW(hwnd, message, wparam, lparam);
            Rc::decrement_strong_count(pointer);
            return result;
        }
        if message == WM_SETFOCUS && state.ready.get() {
            // No state borrow crosses SetFocus's synchronous native messages.
            if let Err(error) = focus_child(hwnd, &state) {
                let mut first = state.error.borrow_mut();
                if first.is_none() {
                    *first = Some(error);
                }
            }
            return LRESULT(0);
        }
        DefWindowProcW(hwnd, message, wparam, lparam)
    }
}
