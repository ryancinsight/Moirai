//! An exclusively owned native parent for one windowed WebView2 controller.
//!
//! Only this owner lends its HWND to CreateCoreWebView2Controller. The native
//! child subtree therefore belongs to that controller by construction, without
//! class-name/PID guesses or caching a runtime-owned HWND. The private parent
//! outlives controller Close and ends before the application window closes.

mod callback;
#[cfg(test)]
mod tests;

use std::{
    cell::{Cell, RefCell},
    io,
    rc::Rc,
};
use windows::{
    Win32::{
        Foundation::{ERROR_CLASS_ALREADY_EXISTS, GetLastError, HINSTANCE, HWND, RECT},
        System::LibraryLoader::GetModuleHandleW,
        UI::WindowsAndMessaging::{
            CreateWindowExW, DestroyWindow, GetClientRect, RegisterClassW, SWP_NOACTIVATE,
            SWP_NOZORDER, SetWindowPos, WINDOW_EX_STYLE, WNDCLASSW, WS_CHILD, WS_CLIPCHILDREN,
            WS_CLIPSIBLINGS, WS_TABSTOP, WS_VISIBLE,
        },
    },
    core::{PCWSTR, w},
};

use super::error::callback_context;

const CLASS_NAME: PCWSTR = w!("MoiraiWebViewParent");

#[derive(Default)]
struct State {
    alive: Cell<bool>,
    ready: Cell<bool>,
    forwarding: Cell<bool>,
    error: RefCell<Option<io::Error>>,
}

pub(super) struct ControllerContainer {
    pub(super) hwnd: HWND,
    state: Rc<State>,
}

impl ControllerContainer {
    pub(super) fn new(parent: HWND) -> io::Result<Self> {
        // SAFETY: immediate owner-thread module/client-rectangle queries; the
        // NativeWindow remains owned by the caller for this complete operation.
        let module =
            unsafe { GetModuleHandleW(None) }.map_err(|e| context("container module", e))?;
        let class = WNDCLASSW {
            lpfnWndProc: Some(callback::window_proc),
            hInstance: HINSTANCE(module.0),
            lpszClassName: CLASS_NAME,
            ..Default::default()
        };
        let atom = unsafe { RegisterClassW(&class) };
        if atom == 0 {
            let error = unsafe { GetLastError() };
            if error != ERROR_CLASS_ALREADY_EXISTS {
                return Err(io::Error::from_raw_os_error(error.0.cast_signed()));
            }
        }
        let mut bounds = RECT::default();
        unsafe { GetClientRect(parent, &mut bounds) }
            .map_err(|e| context("container client rectangle", e))?;
        let state = Rc::new(State::default());
        // SAFETY: the creation call borrows this live Rc allocation. The
        // procedure adopts a separate count at WM_NCCREATE, releases it at
        // WM_NCDESTROY, and stores no borrowed ControllerContainer address.
        let hwnd = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                CLASS_NAME,
                w!(""),
                WS_CHILD | WS_VISIBLE | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_TABSTOP,
                0,
                0,
                bounds.right,
                bounds.bottom,
                Some(parent),
                None,
                Some(HINSTANCE(module.0)),
                Some(Rc::as_ptr(&state).cast()),
            )
        }
        .map_err(|e| context("create controller container", e))?;
        Ok(Self { hwnd, state })
    }

    pub(super) fn activate(&self) -> io::Result<()> {
        self.ensure_live()?;
        self.state.ready.set(true);
        callback::focus_child(self.hwnd, &self.state)
    }

    pub(super) fn resize(&self, width: i32, height: i32) -> io::Result<()> {
        self.ensure_live()?;
        // SAFETY: a live private HWND on its creating thread, fixed client
        // origin, and the dimensions already admitted by WebViewHost::resize.
        unsafe {
            SetWindowPos(
                self.hwnd,
                None,
                0,
                0,
                width,
                height,
                SWP_NOACTIVATE | SWP_NOZORDER,
            )
        }
        .map_err(|e| context("resize controller container", e))
    }

    pub(super) fn deactivate(&self) {
        self.state.ready.set(false);
    }

    pub(super) fn take_error(&self) -> io::Result<()> {
        self.state.error.borrow_mut().take().map_or(Ok(()), Err)
    }

    pub(super) fn ensure_live(&self) -> io::Result<()> {
        if self.state.alive.get() {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "owned controller container is destroyed",
            ))
        }
    }

    #[cfg(test)]
    pub(super) fn liveness_probe(&self) -> impl Fn() -> bool + 'static + use<> {
        let state = Rc::clone(&self.state);
        move || state.alive.get()
    }

    pub(super) fn close(&mut self) -> io::Result<()> {
        self.deactivate();
        let mut first = self.take_error().err();
        if self.state.alive.get() {
            // SAFETY: alive belongs to this allocation and is cleared by its
            // WM_NCDESTROY; an old numeric HWND is never used after destruction.
            if let Err(error) = unsafe { DestroyWindow(self.hwnd) } {
                let error = context("destroy controller container", error);
                if first.is_some() {
                    tracing::error!(%error,"secondary controller container destruction failure");
                }
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }
}

impl Drop for ControllerContainer {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::error!(%error,"controller container teardown failed");
        }
    }
}

fn context(operation: &str, error: windows::core::Error) -> io::Error {
    io::Error::other(callback_context(operation, error))
}
