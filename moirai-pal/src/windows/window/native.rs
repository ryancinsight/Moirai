//! Win32 handle ownership, message translation and software presentation.

use std::ffi::c_void;
use std::io;
use std::mem::size_of;
use std::time::Duration;

use windows::Win32::Foundation::{
    ERROR_CLASS_ALREADY_EXISTS, GetLastError, HINSTANCE, HWND, LPARAM, LRESULT, RECT, WAIT_FAILED,
    WAIT_TIMEOUT, WPARAM,
};
use windows::Win32::Graphics::Gdi::{HDC, UpdateWindow};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetThreadDpiAwarenessContext,
};
use windows::Win32::UI::Input::Ime::{
    GCS_COMPSTR, GCS_RESULTSTR, ImmGetCompositionStringW, ImmGetContext, ImmReleaseContext,
};
use windows::Win32::UI::WindowsAndMessaging::{
    AdjustWindowRectEx, CS_HREDRAW, CS_VREDRAW, CreateWindowExW, DefWindowProcW, DestroyWindow,
    DispatchMessageW, GWLP_USERDATA, GetClientRect, GetWindowLongPtrW, IDC_ARROW, IsWindow,
    LoadCursorW, MWMO_INPUTAVAILABLE, MsgWaitForMultipleObjectsEx, PM_REMOVE, PeekMessageW,
    QS_ALLINPUT, RegisterClassW, SW_SHOW, SW_SHOWMAXIMIZED, SetWindowLongPtrW, ShowWindow,
    TranslateMessage, WINDOW_EX_STYLE, WM_CHAR, WM_CLOSE, WM_DESTROY, WM_DPICHANGED, WM_ERASEBKGND,
    WM_HOTKEY, WM_IME_COMPOSITION, WM_IME_ENDCOMPOSITION, WM_IME_STARTCOMPOSITION, WM_KEYDOWN,
    WM_KEYUP, WM_KILLFOCUS, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MBUTTONDOWN, WM_MBUTTONUP,
    WM_MOUSEHWHEEL, WM_MOUSEMOVE, WM_MOUSEWHEEL, WM_NCCREATE, WM_NCDESTROY, WM_PAINT, WM_PRINT,
    WM_PRINTCLIENT, WM_RBUTTONDOWN, WM_RBUTTONUP, WM_SETFOCUS, WM_SIZE, WM_SYSKEYDOWN, WM_SYSKEYUP,
    WM_XBUTTONDOWN, WM_XBUTTONUP, WNDCLASSW, WS_OVERLAPPEDWINDOW,
};
use windows::core::PCWSTR;

use super::accessibility::{
    ACCESSIBILITY_WAKE_MESSAGE, AccessibilityTree, WindowsAccessibilityAdapter,
};
use super::config::{
    MAX_COMPOSITION_UNITS, MAX_PUMP_MESSAGES, MAX_WAIT_MILLISECONDS, WindowConfig,
    WindowVisibility, allocation_error, coordinate_error, validate_frame_dimensions, windows_error,
};
use super::event::{CompositionPhase, WindowEvent};
use super::hotkey::HotkeyId;
use super::input::{
    client_point_from_wheel_lparam, extent_from_lparam, mouse_button, point_from_lparam,
    wheel_deltas,
};
use super::present::{paint, paint_frame};
use super::state::{WindowState, decode_composition};
use super::tray::{OwnedIcon, TRAY_CALLBACK_MESSAGE};

const WINDOW_CLASS_NAME: &[u16] = &[
    b'M' as u16,
    b'o' as u16,
    b'i' as u16,
    b'r' as u16,
    b'a' as u16,
    b'i' as u16,
    b'W' as u16,
    b'i' as u16,
    b'n' as u16,
    b'd' as u16,
    b'o' as u16,
    b'w' as u16,
    0,
];

/// A thread-owned native Win32 window and bounded software presenter.
pub struct NativeWindow {
    pub(crate) hwnd: HWND,
    #[expect(dead_code, reason = "the guard's Drop restores the thread context")]
    dpi_context: ThreadDpiAwarenessContext,
    pub(super) state: Box<WindowState>,
    accessibility: Option<WindowsAccessibilityAdapter>,
    pub(super) visible: bool,
    pub(super) show_maximized: bool,
    pub(super) hotkeys: Vec<HotkeyId>,
    pub(super) tray_icon: Option<OwnedIcon>,
    destroyed: bool,
}

/// Restores the creating thread's DPI context when the native window leaves it.
struct ThreadDpiAwarenessContext {
    previous: DPI_AWARENESS_CONTEXT,
}

impl ThreadDpiAwarenessContext {
    fn enter() -> io::Result<Self> {
        // SAFETY: the requested context is an operating-system constant and the
        // call affects only the current thread, which owns the NativeWindow.
        let previous =
            unsafe { SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
        if previous.is_invalid() {
            let error = unsafe { GetLastError() };
            return Err(io::Error::from_raw_os_error(error.0 as i32));
        }
        Ok(Self { previous })
    }
}

impl Drop for ThreadDpiAwarenessContext {
    fn drop(&mut self) {
        if !self.previous.is_invalid() {
            // SAFETY: the guard is dropped with its thread-owned NativeWindow;
            // the retained opaque context came from this same thread.
            unsafe {
                SetThreadDpiAwarenessContext(self.previous);
            }
        }
    }
}

impl NativeWindow {
    /// Creates a native window using the current thread's message queue.
    ///
    /// # Errors
    /// Returns the native error when class registration or window creation
    /// fails, or `InvalidInput` for invalid configuration.
    pub fn new(config: &WindowConfig) -> io::Result<Self> {
        let mut window = Self::new_hidden(config)?;
        if let Some(placement) = config.placement() {
            window.set_placement(placement)?;
        }
        if config.visibility() == WindowVisibility::Visible {
            window.show()?;
        }
        Ok(window)
    }

    fn new_hidden(config: &WindowConfig) -> io::Result<Self> {
        let dpi_context = ThreadDpiAwarenessContext::enter()?;
        let instance = register_class()?;
        let (outer_width, outer_height) = outer_dimensions(config.width(), config.height())?;
        let mut state = Box::new(WindowState::new()?);
        let state_ptr: *mut WindowState = &mut *state;
        // SAFETY: `instance`, the class/title UTF-16 buffers and `state_ptr`
        // remain valid for the complete synchronous CreateWindowExW call. The
        // window procedure stores the state pointer only for this HWND, and
        // `NativeWindow` keeps the Box alive until DestroyWindow returns.
        let hwnd = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                PCWSTR(WINDOW_CLASS_NAME.as_ptr()),
                PCWSTR(config.title_utf16().as_ptr()),
                WS_OVERLAPPEDWINDOW,
                0,
                0,
                outer_width,
                outer_height,
                None,
                None,
                Some(HINSTANCE::from(instance)),
                Some(state_ptr.cast::<c_void>()),
            )
        }
        .map_err(windows_error)?;
        let mut window = Self {
            hwnd,
            dpi_context,
            state,
            accessibility: None,
            visible: false,
            show_maximized: false,
            hotkeys: Vec::new(),
            tray_icon: None,
            destroyed: false,
        };
        if !window
            .state
            .events
            .iter()
            .any(|event| matches!(event, WindowEvent::Resized { .. }))
        {
            let (width, height) = client_dimensions(hwnd)?;
            window.state.push(WindowEvent::Resized { width, height });
        }
        Ok(window)
    }

    /// Shows a hidden window after host adapters have been installed.
    ///
    /// # Errors
    /// Returns the native repaint error or an invalid-state error for a closed
    /// window.
    pub fn show(&mut self) -> io::Result<()> {
        if self.destroyed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot show a destroyed native window",
            ));
        }
        if self.visible {
            return Ok(());
        }
        // SAFETY: `self.hwnd` is the live handle created on this thread and the
        // calls are synchronous; no pointer is retained by either operation.
        let updated = unsafe {
            let command = if self.show_maximized {
                SW_SHOWMAXIMIZED
            } else {
                SW_SHOW
            };
            let _ = ShowWindow(self.hwnd, command);
            UpdateWindow(self.hwnd).as_bool()
        };
        if !updated {
            let error = io::Error::last_os_error();
            let _ = self.close();
            return Err(error);
        }
        self.visible = true;
        Ok(())
    }

    /// Installs the Windows accessibility adapter while the HWND is hidden.
    ///
    /// # Errors
    /// Returns an invalid-state error when the window is visible or already has
    /// an adapter, or a validation error for the supplied tree.
    pub fn install_accessibility(&mut self, tree: AccessibilityTree) -> io::Result<()> {
        if self.visible {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "native accessibility must be installed before the window is shown",
            ));
        }
        if self.accessibility.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "native accessibility is already installed",
            ));
        }
        self.accessibility = Some(WindowsAccessibilityAdapter::new(self.hwnd, tree)?);
        Ok(())
    }

    /// Replaces the current native accessibility tree and raises its events.
    ///
    /// # Errors
    /// Returns a validation, queue or native accessibility error.
    pub fn update_accessibility(&mut self, tree: AccessibilityTree) -> io::Result<()> {
        self.accessibility
            .as_mut()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "native accessibility is not installed",
                )
            })?
            .update(tree)
    }

    /// Returns whether the HWND has completed destruction.
    #[must_use]
    pub const fn is_destroyed(&self) -> bool {
        self.destroyed
    }

    /// Pumps at most [`MAX_PUMP_MESSAGES`] messages addressed to this HWND.
    ///
    /// A later call continues a larger native burst, so this method never spins
    /// indefinitely when another producer keeps posting messages.
    ///
    /// # Errors
    /// Returns an error if the bounded event queue overflowed or a native pump
    /// operation reports failure.
    pub fn poll_events(&mut self) -> io::Result<Vec<WindowEvent>> {
        if self.destroyed {
            return Ok(Vec::new());
        }
        let mut message = windows::Win32::UI::WindowsAndMessaging::MSG::default();
        for _ in 0..MAX_PUMP_MESSAGES {
            // SAFETY: `message` is writable storage owned by this call; the HWND
            // filter prevents consuming another window's queue entries.
            let present = unsafe { PeekMessageW(&mut message, Some(self.hwnd), 0, 0, PM_REMOVE) };
            if !present.as_bool() {
                break;
            }
            // SAFETY: `message` was populated by PeekMessageW and remains valid
            // for both synchronous message-dispatch calls.
            unsafe {
                let _ = TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }
        if let Some(error) = self.state.error.take() {
            return Err(error);
        }
        if self.state.overflowed {
            self.state.overflowed = false;
            return Err(io::Error::other(
                "native window event queue capacity exceeded",
            ));
        }
        let mut events: Vec<WindowEvent> = self.state.events.drain(..).collect();
        if let Some(accessibility) = self.accessibility.as_mut() {
            events.extend(
                accessibility
                    .take_actions()?
                    .into_iter()
                    .map(|request| WindowEvent::AccessibilityAction { request }),
            );
        }
        Ok(events)
    }

    /// Waits for native input for a finite duration, then returns one bounded
    /// event batch.
    ///
    /// A zero duration performs an immediate readiness check. The wait is
    /// limited to [`MAX_WAIT_MILLISECONDS`] so a caller cannot turn a window
    /// operation into an unbounded blocking point.
    ///
    /// # Errors
    /// Returns an invalid-duration or native wait error, or the same queue
    /// overflow error as [`Self::poll_events`].
    pub fn wait_events(&mut self, timeout: Duration) -> io::Result<Vec<WindowEvent>> {
        if self.destroyed {
            return Ok(Vec::new());
        }
        let milliseconds = u32::try_from(timeout.as_millis()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "native event wait exceeds the 30 second bound",
            )
        })?;
        if milliseconds > MAX_WAIT_MILLISECONDS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "native event wait exceeds the 30 second bound",
            ));
        }
        if !self.state.events.is_empty()
            || self.state.overflowed
            || self.state.error.is_some()
            || self
                .accessibility
                .as_ref()
                .is_some_and(WindowsAccessibilityAdapter::has_pending_actions)
        {
            return self.poll_events();
        }
        // SAFETY: the call observes only this thread's message queue, accepts
        // no handles, and retains no pointer after returning.
        let result = unsafe {
            MsgWaitForMultipleObjectsEx(None, milliseconds, QS_ALLINPUT, MWMO_INPUTAVAILABLE)
        };
        if result == WAIT_FAILED {
            return Err(io::Error::last_os_error());
        }
        if result == WAIT_TIMEOUT {
            return Ok(Vec::new());
        }
        self.poll_events()
    }

    /// Destroys the window synchronously and drains its callback state.
    ///
    /// # Errors
    /// Returns the native destruction error. Calling this method after a native
    /// destroy message is harmless and reports success.
    pub fn close(&mut self) -> io::Result<()> {
        if self.destroyed {
            return Ok(());
        }
        self.release_hotkeys();
        self.release_tray_icon();
        // Remove the subclass while the HWND is still valid. The adapter's
        // destructor is thread-affine and restores the original window proc.
        self.accessibility.take();
        // SAFETY: the handle belongs to this thread and IsWindow only observes
        // the handle before the synchronous DestroyWindow call.
        if unsafe { IsWindow(Some(self.hwnd)) }.as_bool() {
            unsafe { DestroyWindow(self.hwnd) }.map_err(windows_error)?;
        }
        self.destroyed = true;
        Ok(())
    }
}

impl Drop for NativeWindow {
    fn drop(&mut self) {
        if !self.destroyed {
            self.release_hotkeys();
            self.release_tray_icon();
            self.accessibility.take();
            // Drop cannot report errors. DestroyWindow is the synchronous RAII
            // fallback; the callback remains valid through the call because
            // `state` is dropped only after this method returns.
            // SAFETY: the handle was created on this thread and remains owned by
            // this object until the destructor finishes.
            if unsafe { IsWindow(Some(self.hwnd)) }.as_bool() {
                let _ = unsafe { DestroyWindow(self.hwnd) };
            }
            self.destroyed = true;
        }
    }
}

fn register_class() -> io::Result<windows::Win32::Foundation::HMODULE> {
    // SAFETY: querying the current module with a null name is a process-local
    // read and returns a handle valid for the class registration lifetime.
    let instance = unsafe { GetModuleHandleW(None) }.map_err(windows_error)?;
    // SAFETY: `WINDOW_CLASS_NAME` is a static NUL-terminated UTF-16 buffer and
    // `window_proc` has the system callback ABI required by WNDCLASSW.
    let cursor = unsafe { LoadCursorW(None, IDC_ARROW) }.map_err(windows_error)?;
    let class = WNDCLASSW {
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(window_proc),
        hInstance: HINSTANCE::from(instance),
        hCursor: cursor,
        lpszClassName: PCWSTR(WINDOW_CLASS_NAME.as_ptr()),
        ..Default::default()
    };
    // SAFETY: `class` points to valid static strings and a function pointer for
    // the duration of this synchronous registration call.
    let atom = unsafe { RegisterClassW(&class) };
    if atom == 0 {
        // SAFETY: GetLastError reads the thread-local status set by RegisterClassW.
        let error = unsafe { GetLastError() };
        if error != ERROR_CLASS_ALREADY_EXISTS {
            return Err(io::Error::from_raw_os_error(error.0 as i32));
        }
    }
    Ok(instance)
}

unsafe extern "system" fn window_proc(
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
            let state = (*create).lpCreateParams.cast::<WindowState>();
            if state.is_null() {
                return LRESULT(0);
            }
            // SAFETY: the state pointer came from NativeWindow's live Box and the
            // HWND is being initialized synchronously by CreateWindowExW.
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, state as isize);
            return LRESULT(1);
        }

        // SAFETY: GWLP_USERDATA is written only by WM_NCCREATE above for this HWND;
        // the null check prevents dereferencing a window not created by this module.
        let state_ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut WindowState;
        if state_ptr.is_null() {
            return DefWindowProcW(hwnd, message, wparam, lparam);
        }
        // SAFETY: NativeWindow keeps the Box alive until DestroyWindow returns; the
        // callback is invoked synchronously on the owning window thread.
        let state = &mut *state_ptr;
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
            state.finish_text();
        }
        match message {
            WM_CLOSE => state.push(WindowEvent::CloseRequested),
            WM_HOTKEY => state.push_hotkey(wparam.0),
            TRAY_CALLBACK_MESSAGE => state.push_tray(wparam.0, lparam.0),
            WM_DESTROY => {}
            WM_SETFOCUS => state.push(WindowEvent::FocusGained),
            WM_KILLFOCUS => {
                state.clear_modifiers();
                state.push(WindowEvent::FocusLost);
            }
            WM_MOUSEMOVE => {
                let (x, y) = point_from_lparam(lparam);
                state.push(WindowEvent::PointerMove { x, y });
            }
            WM_LBUTTONDOWN | WM_RBUTTONDOWN | WM_MBUTTONDOWN | WM_XBUTTONDOWN => {
                let (x, y) = point_from_lparam(lparam);
                if let Some(button) = mouse_button(message, wparam) {
                    state.push(WindowEvent::PointerDown { x, y, button });
                }
            }
            WM_LBUTTONUP | WM_RBUTTONUP | WM_MBUTTONUP | WM_XBUTTONUP => {
                let (x, y) = point_from_lparam(lparam);
                if let Some(button) = mouse_button(message, wparam) {
                    state.push(WindowEvent::PointerUp { x, y, button });
                }
            }
            WM_MOUSEWHEEL | WM_MOUSEHWHEEL => {
                if let Some((delta_x, delta_y)) = wheel_deltas(message, wparam) {
                    match client_point_from_wheel_lparam(hwnd, lparam) {
                        Ok((x, y)) => state.push(WindowEvent::PointerWheel {
                            x,
                            y,
                            delta_x,
                            delta_y,
                            modifiers: state.modifiers.with_wheel_message_flags(wparam.0),
                        }),
                        Err(error) => state.record_error(error),
                    }
                }
            }
            WM_KEYDOWN => {
                let virtual_key = wparam.0 as u32;
                state.update_modifier(virtual_key, lparam, true);
                state.push(WindowEvent::KeyDown {
                    virtual_key,
                    repeated: (lparam.0 & (1 << 30)) != 0,
                    modifiers: state.modifiers,
                });
            }
            WM_SYSKEYDOWN => {
                let virtual_key = wparam.0 as u32;
                state.update_modifier(virtual_key, lparam, true);
                state.push(WindowEvent::KeyDown {
                    virtual_key,
                    repeated: (lparam.0 & (1 << 30)) != 0,
                    modifiers: state.modifiers,
                });
                // System-key messages carry Alt/menu and F10/F4 behavior that
                // DefWindowProcW must retain after the PAL records the value event.
                return DefWindowProcW(hwnd, message, wparam, lparam);
            }
            WM_KEYUP => {
                let virtual_key = wparam.0 as u32;
                state.update_modifier(virtual_key, lparam, false);
                state.push(WindowEvent::KeyUp {
                    virtual_key,
                    modifiers: state.modifiers,
                });
            }
            WM_SYSKEYUP => {
                let virtual_key = wparam.0 as u32;
                state.update_modifier(virtual_key, lparam, false);
                state.push(WindowEvent::KeyUp {
                    virtual_key,
                    modifiers: state.modifiers,
                });
                return DefWindowProcW(hwnd, message, wparam, lparam);
            }
            WM_CHAR => state.push_text_unit(wparam.0 as u16),
            WM_IME_STARTCOMPOSITION => {
                state.push_composition(CompositionPhase::Started, String::new());
            }
            WM_IME_COMPOSITION => {
                if let Err(error) = composition_message(hwnd, state, lparam) {
                    state.record_error(error);
                }
            }
            WM_IME_ENDCOMPOSITION => {
                if state.composition_active {
                    state.push_composition(CompositionPhase::Canceled, String::new());
                }
            }
            WM_SIZE => {
                let (width, height) = extent_from_lparam(lparam);
                state.push(WindowEvent::Resized { width, height });
            }
            WM_DPICHANGED => {
                let dpi = (wparam.0 & 0xffff) as u32;
                if dpi != 0 {
                    state.push(WindowEvent::DpiChanged { dpi });
                }
            }
            WM_ERASEBKGND => return LRESULT(1),
            WM_PRINT => {
                let result = DefWindowProcW(hwnd, message, wparam, lparam);
                let hdc = HDC(wparam.0 as *mut c_void);
                // SAFETY: WM_PRINT supplies a live destination HDC for the
                // synchronous full-window render; the retained frame is
                // borrowed only for the duration of the GDI calls.
                paint_frame(hwnd, state, hdc);
                return result;
            }
            WM_PRINTCLIENT => {
                let hdc = HDC(wparam.0 as *mut c_void);
                // SAFETY: WM_PRINTCLIENT supplies a live destination HDC for
                // the synchronous client-area render; the retained frame is
                // borrowed only for the duration of the GDI calls.
                paint_frame(hwnd, state, hdc);
            }
            WM_PAINT => {
                // SAFETY: the callback owns the live HWND and its state for the
                // duration of the synchronous paint operation.
                paint(hwnd, state);
            }
            WM_NCDESTROY => {
                state.finish_text();
                state.push(WindowEvent::Destroyed);
                // SAFETY: clearing the module-owned userdata before returning from
                // WM_NCDESTROY prevents later messages from observing stale state.
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
            }
            ACCESSIBILITY_WAKE_MESSAGE => {}
            _ => return DefWindowProcW(hwnd, message, wparam, lparam),
        }
        LRESULT(0)
    }
}

fn composition_message(hwnd: HWND, state: &mut WindowState, lparam: LPARAM) -> io::Result<()> {
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
        if state.composition_active {
            state.push_composition(CompositionPhase::Canceled, String::new());
        }
        return Ok(());
    };
    let text = read_composition_text(hwnd, kind)?;
    state.push_composition(phase, text);
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

pub(super) fn outer_dimensions(width: u32, height: u32) -> io::Result<(i32, i32)> {
    validate_frame_dimensions(width, height)?;
    let right = i32::try_from(width).map_err(|_| coordinate_error())?;
    let bottom = i32::try_from(height).map_err(|_| coordinate_error())?;
    let mut rect = RECT {
        right,
        bottom,
        ..Default::default()
    };
    // SAFETY: `rect` is writable storage owned by this call; the style and
    // extended style are constants, and no menu is attached to the window.
    unsafe {
        AdjustWindowRectEx(
            &mut rect,
            WS_OVERLAPPEDWINDOW,
            false,
            WINDOW_EX_STYLE::default(),
        )
    }
    .map_err(windows_error)?;
    let outer_width = rect
        .right
        .checked_sub(rect.left)
        .ok_or_else(coordinate_error)?;
    let outer_height = rect
        .bottom
        .checked_sub(rect.top)
        .ok_or_else(coordinate_error)?;
    if outer_width <= 0 || outer_height <= 0 {
        return Err(coordinate_error());
    }
    Ok((outer_width, outer_height))
}

fn client_dimensions(hwnd: HWND) -> io::Result<(u32, u32)> {
    let mut rect = RECT::default();
    // SAFETY: `rect` is writable storage owned by this call and hwnd is the
    // live handle returned by CreateWindowExW.
    unsafe { GetClientRect(hwnd, &mut rect) }.map_err(windows_error)?;
    let width = rect
        .right
        .checked_sub(rect.left)
        .ok_or_else(coordinate_error)?;
    let height = rect
        .bottom
        .checked_sub(rect.top)
        .ok_or_else(coordinate_error)?;
    Ok((
        u32::try_from(width).map_err(|_| coordinate_error())?,
        u32::try_from(height).map_err(|_| coordinate_error())?,
    ))
}
