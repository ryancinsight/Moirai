//! The host handle: its WebView2 interfaces, the operations over them,
//! and the teardown that keeps COM alive until the last one is gone.

use std::{cell::RefCell, io, rc::Rc, sync::mpsc, time::Duration};

use webview2_com::CapturePreviewCompletedHandler;
use webview2_com::Microsoft::Web::WebView2::Win32::{
    COREWEBVIEW2_CAPTURE_PREVIEW_IMAGE_FORMAT_PNG, COREWEBVIEW2_HOST_RESOURCE_ACCESS_KIND_DENY,
    ICoreWebView2, ICoreWebView2_3, ICoreWebView2Controller, ICoreWebView2Environment,
    ICoreWebView2NavigationCompletedEventHandler, ICoreWebView2NavigationStartingEventHandler,
    ICoreWebView2NewWindowRequestedEventHandler, ICoreWebView2PermissionRequestedEventHandler,
    ICoreWebView2WebMessageReceivedEventHandler,
};
use windows::{
    Win32::{
        Foundation::RECT,
        System::Com::{STATFLAG_NONAME, STATSTG, STREAM_SEEK_SET},
        UI::{Shell::SHCreateMemStream, WindowsAndMessaging::MSG},
    },
    core::{HSTRING, Interface, PCWSTR},
};

use super::super::super::window::NativeWindow;
use super::super::{
    config::{
        MAX_WEBVIEW_CAPTURE_BYTES, MAX_WEBVIEW_MESSAGE_UNITS, MAX_WEBVIEW_URI_UNITS, WebViewConfig,
        validate_message,
    },
    event::WebViewHostEvent,
    folder::FolderMapping,
    pump::{dispatch_pending, wait_for, wait_for_messages},
    state::WebViewState,
};

use super::error::{closed_error, coordinate_error, windows_error};
use super::{
    callbacks::{
        remove_message, remove_navigation_completed, remove_navigation_starting, remove_new_window,
        remove_permission_requested,
    },
    com::{ComApartment, create_controller, create_environment},
    text::encode_utf16,
};

/// A thread-affine WebView2 controller hosted inside one [`NativeWindow`].
pub struct WebViewHost {
    pub(super) window: NativeWindow,
    pub(super) callbacks: Callbacks,
    pub(super) webview: Option<ICoreWebView2>,
    pub(super) controller: Option<ICoreWebView2Controller>,
    pub(super) environment: Option<ICoreWebView2Environment>,
    pub(super) state: Rc<RefCell<WebViewState>>,
    pub(super) current_uri: Rc<RefCell<String>>,
    pub(super) config: WebViewConfig,
    pub(super) apartment: ComApartment,
    pub(super) closed: bool,
}

pub(super) struct Callbacks {
    pub(super) navigation_starting: Option<(i64, ICoreWebView2NavigationStartingEventHandler)>,
    pub(super) navigation_completed: Option<(i64, ICoreWebView2NavigationCompletedEventHandler)>,
    pub(super) new_window: Option<(i64, ICoreWebView2NewWindowRequestedEventHandler)>,
    pub(super) permission_requested: Option<(i64, ICoreWebView2PermissionRequestedEventHandler)>,
    pub(super) message: Option<(i64, ICoreWebView2WebMessageReceivedEventHandler)>,
}

impl Callbacks {
    pub(super) const fn empty() -> Self {
        Self {
            navigation_starting: None,
            navigation_completed: None,
            new_window: None,
            permission_requested: None,
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
        if let Some(mapping) = config.folder_mapping() {
            map_folder(&webview, mapping)?;
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
            || width > super::super::super::window::MAX_FRAME_DIMENSION
            || height > super::super::super::window::MAX_FRAME_DIMENSION
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

    /// Captures the rendered page as a bounded PNG from WebView2 itself.
    ///
    /// The preview is read from a COM memory stream after WebView2 completes
    /// the asynchronous capture. It does not depend on the parent window's
    /// compositor or a GDI screenshot API, which keeps evidence valid for
    /// occluded and hardware-composed surfaces.
    ///
    /// # Errors
    /// Returns `NotConnected` for a closed host, `OutOfMemory` when the encoded
    /// preview exceeds [`MAX_WEBVIEW_CAPTURE_BYTES`], a finite-wait error when
    /// WebView2 does not complete, or a native COM/WebView2/stream error.
    pub fn capture_preview_png(&self) -> io::Result<Vec<u8>> {
        let webview = self.webview.as_ref().ok_or_else(closed_error)?;
        // SAFETY: COM is initialized on this owner thread and the returned
        // stream is retained until WebView2 invokes the completion callback.
        let stream = unsafe { SHCreateMemStream(None) }.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                "WebView2 preview stream allocation failed",
            )
        })?;
        let (sender, receiver) = mpsc::sync_channel(1);
        let handler = CapturePreviewCompletedHandler::create(Box::new(move |error_code| {
            sender
                .send(error_code)
                .map_err(|_| super::error::callback_error("WebView2 preview waiter was dropped"))
        }));
        // SAFETY: `webview`, `stream` and `handler` are valid COM interfaces
        // owned by this thread; WebView2 retains the stream until completion.
        unsafe {
            webview
                .CapturePreview(
                    COREWEBVIEW2_CAPTURE_PREVIEW_IMAGE_FORMAT_PNG,
                    &stream,
                    &handler,
                )
                .map_err(windows_error)?;
        }
        wait_for(receiver, self.config.wait())?;

        let mut stat = STATSTG::default();
        // SAFETY: `stat` is writable storage and `stream` is a live COM
        // `IStream` returned by the platform memory-stream factory.
        unsafe {
            stream
                .Stat(&mut stat, STATFLAG_NONAME)
                .map_err(windows_error)?;
            stream
                .Seek(0, STREAM_SEEK_SET, None)
                .map_err(windows_error)?;
        }
        let size = usize::try_from(stat.cbSize).map_err(|_| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                "WebView2 preview length is not representable",
            )
        })?;
        if size > MAX_WEBVIEW_CAPTURE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "WebView2 preview exceeds the bounded capture budget",
            ));
        }
        let mut bytes = vec![0; size];
        let mut offset = 0;
        while offset < size {
            let count = u32::try_from((size - offset).min(u32::MAX as usize)).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "WebView2 preview read length is not representable",
                )
            })?;
            let mut read = 0;
            // SAFETY: the destination is the remaining initialized slice, and
            // the COM stream writes at most the requested byte count.
            unsafe {
                stream
                    .Read(bytes[offset..].as_mut_ptr().cast(), count, Some(&mut read))
                    .ok()
                    .map_err(windows_error)?;
            }
            if read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "WebView2 preview stream ended before its reported length",
                ));
            }
            offset = offset
                .checked_add(usize::try_from(read).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "WebView2 preview read count is not representable",
                    )
                })?)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "WebView2 preview read offset overflowed",
                    )
                })?;
        }
        Ok(bytes)
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
            remove_permission_requested(
                webview,
                &mut self.callbacks.permission_requested,
                &mut first_error,
            );
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

/// Serves `mapping`'s folder under its host; other origins are denied access.
fn map_folder(webview: &ICoreWebView2, mapping: &FolderMapping) -> io::Result<()> {
    let webview: ICoreWebView2_3 = webview.cast().map_err(windows_error)?;
    let host = HSTRING::from(mapping.host());
    let folder = HSTRING::from(mapping.folder().as_os_str());
    // SAFETY: both HSTRINGs are NUL-terminated UTF-16 that outlive the call,
    // and WebView2 copies them before returning.
    unsafe {
        webview.SetVirtualHostNameToFolderMapping(
            &host,
            &folder,
            COREWEBVIEW2_HOST_RESOURCE_ACCESS_KIND_DENY,
        )
    }
    .map_err(windows_error)
}
