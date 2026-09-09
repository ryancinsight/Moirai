//! WebView2 COM lifetime, policy and event integration.

use std::{cell::RefCell, io, rc::Rc, time::Duration};

use webview2_com::Microsoft::Web::WebView2::Win32::{
    ICoreWebView2, ICoreWebView2Controller, ICoreWebView2Environment,
    ICoreWebView2NavigationCompletedEventHandler, ICoreWebView2NavigationStartingEventHandler,
    ICoreWebView2NewWindowRequestedEventHandler, ICoreWebView2WebMessageReceivedEventHandler,
};
use windows::{
    Win32::{
        Foundation::{E_ABORT, RECT},
        UI::WindowsAndMessaging::MSG,
    },
    core::PCWSTR,
};

use super::super::window::NativeWindow;
use super::{
    config::{MAX_WEBVIEW_MESSAGE_UNITS, MAX_WEBVIEW_URI_UNITS, WebViewConfig, validate_message},
    event::{WebViewEvent, WebViewHostEvent},
    pump::{dispatch_pending, wait_for_messages},
    state::WebViewState,
};

use self::{
    callbacks::{
        remove_message, remove_navigation_completed, remove_navigation_starting, remove_new_window,
    },
    com::{ComApartment, create_controller, create_environment},
    text::encode_utf16,
};

mod callbacks;
mod com;
mod lifecycle;
mod text;

/// A thread-affine WebView2 controller hosted inside one [`NativeWindow`].
pub struct WebViewHost {
    window: NativeWindow,
    callbacks: Callbacks,
    webview: Option<ICoreWebView2>,
    controller: Option<ICoreWebView2Controller>,
    environment: Option<ICoreWebView2Environment>,
    state: Rc<RefCell<WebViewState>>,
    current_uri: Rc<RefCell<String>>,
    config: WebViewConfig,
    apartment: ComApartment,
    closed: bool,
}

struct Callbacks {
    navigation_starting: Option<(i64, ICoreWebView2NavigationStartingEventHandler)>,
    navigation_completed: Option<(i64, ICoreWebView2NavigationCompletedEventHandler)>,
    new_window: Option<(i64, ICoreWebView2NewWindowRequestedEventHandler)>,
    message: Option<(i64, ICoreWebView2WebMessageReceivedEventHandler)>,
}

impl Callbacks {
    const fn empty() -> Self {
        Self {
            navigation_starting: None,
            navigation_completed: None,
            new_window: None,
            message: None,
        }
    }
}

impl WebViewHost {
    /// Creates a WebView2 host and loads the configured packaged entry page.
    ///
    /// The caller's thread owns the returned host and must continue pumping it
    /// through [`Self::wait_events`] or [`Self::poll_events`].
    ///
    /// # Errors
    /// Returns a bounded configuration, COM, WebView2, callback, navigation or
    /// native-window error.
    pub fn new(window: NativeWindow, config: WebViewConfig) -> io::Result<Self> {
        if window.is_destroyed() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebView2 host requires a live native window",
            ));
        }
        let apartment = ComApartment::initialize()?;
        let environment = create_environment(config.wait())?;
        let controller = create_controller(&environment, window.hwnd, config.wait())?;
        let webview = unsafe { controller.CoreWebView2() }.map_err(windows_error)?;
        let settings = unsafe { webview.Settings() }.map_err(windows_error)?;
        unsafe {
            settings
                .SetAreDefaultContextMenusEnabled(false)
                .map_err(windows_error)?;
            settings
                .SetAreDevToolsEnabled(false)
                .map_err(windows_error)?;
        }
        let state = Rc::new(RefCell::new(WebViewState::new()?));
        let current_uri = Rc::new(RefCell::new(String::new()));
        let mut host = Self {
            window,
            callbacks: Callbacks::empty(),
            webview: Some(webview),
            controller: Some(controller),
            environment: Some(environment),
            state,
            current_uri,
            config,
            apartment,
            closed: false,
        };
        host.install_callbacks()?;
        host.navigate_and_wait(host.config.start_uri().to_owned())?;
        Ok(host)
    }

    /// Returns whether the controller and parent window have been closed.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    /// Changes the WebView2 controller bounds to a validated client rectangle.
    ///
    /// # Errors
    /// Returns `InvalidInput` for zero or oversized dimensions and a native
    /// error when WebView2 rejects the bounds.
    pub fn resize(&mut self, width: u32, height: u32) -> io::Result<()> {
        let right = i32::try_from(width).map_err(|_| coordinate_error())?;
        let bottom = i32::try_from(height).map_err(|_| coordinate_error())?;
        if width == 0
            || height == 0
            || width > super::super::window::MAX_FRAME_DIMENSION
            || height > super::super::window::MAX_FRAME_DIMENSION
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebView2 bounds exceed the native presentation limit",
            ));
        }
        let controller = self.controller.as_ref().ok_or_else(closed_error)?;
        unsafe {
            controller
                .SetBounds(RECT {
                    left: 0,
                    top: 0,
                    right,
                    bottom,
                })
                .map_err(windows_error)
        }
    }

    /// Sets WebView2 visibility without changing the parent window state.
    ///
    /// # Errors
    /// Returns an error when the controller is closed or the native call fails.
    pub fn set_visible(&mut self, visible: bool) -> io::Result<()> {
        let controller = self.controller.as_ref().ok_or_else(closed_error)?;
        unsafe { controller.SetIsVisible(visible).map_err(windows_error) }
    }

    /// Navigates to an allowlisted packaged resource without waiting for its
    /// completion event.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a disallowed URI or a native WebView2 error.
    pub fn navigate(&mut self, uri: impl AsRef<str>) -> io::Result<()> {
        let uri = uri.as_ref();
        if uri.encode_utf16().count() > MAX_WEBVIEW_URI_UNITS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebView2 navigation URI exceeds the bounded UTF-16 limit",
            ));
        }
        if !self.config.allows(uri) {
            self.record_navigation(uri, false)?;
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "WebView2 navigation is outside the packaged resource prefix",
            ));
        }
        self.navigate_unchecked(uri)
    }

    /// Posts a bounded JSON message to the page.
    ///
    /// # Errors
    /// Returns `InvalidInput` for an oversized or NUL-containing message and a
    /// native error when WebView2 rejects the message.
    pub fn post_json(&mut self, json: impl AsRef<str>) -> io::Result<()> {
        let json = json.as_ref();
        validate_message(json.as_bytes())?;
        let webview = self.webview.as_ref().ok_or_else(closed_error)?;
        let value = encode_utf16(json, MAX_WEBVIEW_MESSAGE_UNITS)?;
        unsafe {
            webview
                .PostWebMessageAsJson(PCWSTR(value.as_ptr()))
                .map_err(windows_error)
        }
    }

    /// Pumps pending owner-thread messages and returns native and WebView2
    /// events in one bounded batch.
    ///
    /// # Errors
    /// Returns queue overflow, callback decoding, native window or controller
    /// errors.
    pub fn poll_events(&mut self) -> io::Result<Vec<WebViewHostEvent>> {
        if self.closed {
            return Ok(Vec::new());
        }
        let mut message = MSG::default();
        dispatch_pending(&mut message);
        self.collect_events()
    }

    /// Waits for owner-thread input or a WebView2 callback for a finite duration.
    ///
    /// # Errors
    /// Returns queue overflow, callback decoding, native window or controller
    /// errors, or an invalid duration.
    pub fn wait_events(&mut self, timeout: Duration) -> io::Result<Vec<WebViewHostEvent>> {
        if self.closed {
            return Ok(Vec::new());
        }
        let events = self.poll_events()?;
        if !events.is_empty() {
            return Ok(events);
        }
        if !wait_for_messages(timeout)? {
            return Ok(Vec::new());
        }
        self.poll_events()
    }

    /// Closes callbacks, the controller and the parent window synchronously.
    ///
    /// All cleanup steps are attempted. The first native error is returned
    /// after the remaining handles have been released.
    ///
    /// # Errors
    /// Returns the first callback-removal, controller-close or window-close
    /// error observed during teardown.
    pub fn close(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        let mut first_error = None;
        if let Some(webview) = self.webview.as_ref() {
            remove_navigation_starting(
                webview,
                &mut self.callbacks.navigation_starting,
                &mut first_error,
            );
            remove_navigation_completed(
                webview,
                &mut self.callbacks.navigation_completed,
                &mut first_error,
            );
            remove_new_window(webview, &mut self.callbacks.new_window, &mut first_error);
            remove_message(webview, &mut self.callbacks.message, &mut first_error);
        }
        if let Some(controller) = self.controller.take()
            && let Err(error) = unsafe { controller.Close() }.map_err(windows_error)
        {
            first_error.get_or_insert(error);
        }
        self.webview.take();
        self.environment.take();
        if let Err(error) = self.window.close() {
            first_error.get_or_insert(error);
        }
        self.closed = true;
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

impl Drop for WebViewHost {
    fn drop(&mut self) {
        // Read the guard field here so its ownership role remains explicit:
        // COM must stay initialized until every WebView2 interface is gone.
        debug_assert!(self.apartment.initialized);
        match self.close() {
            Ok(()) | Err(_) => {}
        }
    }
}

fn push_event(state: &Rc<RefCell<WebViewState>>, event: WebViewEvent) -> windows::core::Result<()> {
    let mut state = state
        .try_borrow_mut()
        .map_err(|_| callback_error("WebView2 event state is already borrowed"))?;
    state.push(event);
    Ok(())
}

fn windows_error(error: windows::core::Error) -> io::Error {
    io::Error::from_raw_os_error(error.code().0)
}

fn callback_error(message: &'static str) -> windows::core::Error {
    windows::core::Error::new(E_ABORT, message)
}

fn closed_error() -> io::Error {
    io::Error::new(io::ErrorKind::NotConnected, "WebView2 host is closed")
}

fn coordinate_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "WebView2 bounds exceed Win32 coordinates",
    )
}
