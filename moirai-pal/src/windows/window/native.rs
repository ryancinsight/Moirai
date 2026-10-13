//! Win32 handle ownership, message translation and software presentation.

use std::io;
use std::time::Duration;

use windows::Win32::Foundation::{
    ERROR_CLASS_ALREADY_EXISTS, GetLastError, HINSTANCE, HWND, RECT, WAIT_FAILED, WAIT_TIMEOUT,
};
use windows::Win32::Graphics::Gdi::UpdateWindow;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetThreadDpiAwarenessContext,
};
use windows::Win32::UI::WindowsAndMessaging::{
    AdjustWindowRectEx, CS_HREDRAW, CS_VREDRAW, CreateWindowExW, DestroyWindow, DispatchMessageW,
    GetClientRect, HMENU, IDC_ARROW, IsWindow, LoadCursorW, MWMO_INPUTAVAILABLE,
    MsgWaitForMultipleObjectsEx, PM_REMOVE, PeekMessageW, QS_ALLINPUT, RegisterClassW, SW_SHOW,
    SW_SHOWMAXIMIZED, ShowWindow, TranslateMessage, WINDOW_EX_STYLE, WNDCLASSW,
    WS_OVERLAPPEDWINDOW,
};
use windows::core::PCWSTR;

use super::accessibility::{AccessibilityTree, WindowsAccessibilityAdapter};
use super::config::{
    MAX_PUMP_MESSAGES, MAX_WAIT_MILLISECONDS, WindowConfig, WindowVisibility, coordinate_error,
    validate_frame_dimensions, windows_error,
};
use super::event::WindowEvent;
use super::hotkey::HotkeyId;
use super::state::{SharedWindowState, WindowState};
use super::tray::OwnedIcon;

mod callback;
use callback::window_proc;

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
    pub(super) state: SharedWindowState,
    accessibility: Option<WindowsAccessibilityAdapter>,
    pub(super) visible: bool,
    pub(super) show_maximized: bool,
    pub(super) hotkeys: Vec<HotkeyId>,
    pub(super) tray_icon: Option<OwnedIcon>,
    /// Owned by the window once attached; destroyed with it.
    pub(super) menu_bar: Option<HMENU>,
    pub(super) menu_shape: Vec<usize>,
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
    /// The configured Unicode title is available even while the window is hidden.
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
        let state = SharedWindowState::new(WindowState::new()?);
        // SAFETY: `instance`, the class/title UTF-16 buffers and the state
        // creation parameter remain valid for the complete synchronous
        // CreateWindowExW call. The window procedure adopts its own strong
        // count of the state during WM_NCCREATE and releases it in
        // WM_NCDESTROY, so the HWND never observes freed state.
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
                Some(state.create_param()),
            )
        }
        .map_err(windows_error)?;
        let window = Self {
            hwnd,
            dpi_context,
            state,
            accessibility: None,
            visible: false,
            show_maximized: false,
            hotkeys: Vec::new(),
            tray_icon: None,
            menu_bar: None,
            menu_shape: Vec::new(),
            destroyed: false,
        };
        let resized = window.state.with(|state| {
            state
                .events
                .iter()
                .any(|event| matches!(event, WindowEvent::Resized { .. }))
        });
        if !resized {
            let (width, height) = client_dimensions(hwnd)?;
            window
                .state
                .with(|state| state.push(WindowEvent::Resized { width, height }));
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
    /// indefinitely when another producer keeps posting messages. After close,
    /// it drains already-produced events, including actual WM_NCDESTROY, without
    /// dispatching messages through a destroyed HWND.
    ///
    /// # Errors
    /// Returns an error if the bounded event queue overflowed or a native pump
    /// operation reports failure.
    pub fn poll_events(&mut self) -> io::Result<Vec<WindowEvent>> {
        let mut message = windows::Win32::UI::WindowsAndMessaging::MSG::default();
        for _ in 0..MAX_PUMP_MESSAGES {
            if self.destroyed {
                break;
            }
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
        let mut events = self.state.with(|state| {
            if let Some(error) = state.error.take() {
                return Err(error);
            }
            if state.overflowed {
                state.overflowed = false;
                return Err(io::Error::other(
                    "native window event queue capacity exceeded",
                ));
            }
            Ok(state.events.drain(..).collect::<Vec<WindowEvent>>())
        })?;
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
            return self.poll_events();
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
        let pending = self
            .state
            .with(|state| !state.events.is_empty() || state.overflowed || state.error.is_some());
        if pending
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

    /// Destroys the window synchronously; polling retains its actual final events.
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
            // fallback. The window holds its own count of the callback state,
            // so a failed call leaves the window with valid state rather than
            // freeing it under a live handle.
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
