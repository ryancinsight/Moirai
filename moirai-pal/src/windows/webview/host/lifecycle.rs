//! WebView2 navigation and event-batch lifecycle.

use std::{cell::Cell, io, rc::Rc, sync::mpsc};

use webview2_com::Microsoft::Web::WebView2::Win32::COREWEBVIEW2_WEB_ERROR_STATUS;
use webview2_com::{NavigationCompletedEventHandler, NavigationStartingEventHandler};
use windows::core::{BOOL, PCWSTR};

use super::super::{
    config::MAX_WEBVIEW_URI_UNITS,
    event::{WebViewEvent, WebViewHostEvent},
    pump::wait_for,
};
use super::error::{callback_error, closed_error, windows_error};
use super::text::encode_utf16;
use super::view::WebViewHost;

impl WebViewHost {
    pub(super) fn navigate_and_wait(&mut self, uri: String) -> io::Result<()> {
        let webview = self.webview.as_ref().ok_or_else(closed_error)?.clone();
        let expected_navigation = Rc::new(Cell::new(None));
        let starting_expected = Rc::clone(&expected_navigation);
        let starting_handler =
            NavigationStartingEventHandler::create(Box::new(move |_sender, args| {
                let args = args.ok_or_else(|| {
                    callback_error("WebView2 navigation callback omitted arguments")
                })?;
                let mut navigation_id = 0;
                unsafe { args.NavigationId(&mut navigation_id)? };
                starting_expected.set(Some(navigation_id));
                Ok(())
            }));
        let mut starting_token = 0;
        unsafe {
            webview
                .add_NavigationStarting(&starting_handler, &mut starting_token)
                .map_err(windows_error)?;
        }
        let (sender, receiver) = mpsc::sync_channel(1);
        let completion_expected = Rc::clone(&expected_navigation);
        let handler = NavigationCompletedEventHandler::create(Box::new(move |_sender, args| {
            let args = args
                .ok_or_else(|| callback_error("WebView2 completion callback omitted arguments"))?;
            let mut navigation_id = 0;
            unsafe { args.NavigationId(&mut navigation_id)? };
            if completion_expected.get() != Some(navigation_id) {
                return Ok(());
            }
            let mut success = BOOL(0);
            unsafe { args.IsSuccess(&mut success)? };
            let mut status = COREWEBVIEW2_WEB_ERROR_STATUS(0);
            unsafe { args.WebErrorStatus(&mut status)? };
            sender
                .send(Ok((success.as_bool(), status.0)))
                .map_err(|_| callback_error("WebView2 navigation waiter was dropped"))
        }));
        let mut token = 0;
        if let Err(error) = unsafe {
            webview
                .add_NavigationCompleted(&handler, &mut token)
                .map_err(windows_error)
        } {
            let _ = unsafe { webview.remove_NavigationStarting(starting_token) };
            return Err(error);
        }
        let value = encode_utf16(&uri, MAX_WEBVIEW_URI_UNITS)?;
        let navigation_result = unsafe {
            webview
                .Navigate(PCWSTR(value.as_ptr()))
                .map_err(windows_error)
        };
        if let Err(error) = navigation_result {
            let _ = unsafe { webview.remove_NavigationCompleted(token) };
            let _ = unsafe { webview.remove_NavigationStarting(starting_token) };
            return Err(error);
        }
        let result = wait_for(receiver, self.config.wait());
        let removal = unsafe { webview.remove_NavigationCompleted(token) }.map_err(windows_error);
        let starting_removal =
            unsafe { webview.remove_NavigationStarting(starting_token) }.map_err(windows_error);
        removal.and(starting_removal)?;
        let (success, status) = result?;
        if success {
            Ok(())
        } else {
            let policy_allowed = self
                .state
                .try_borrow()
                .map_err(|_| io::Error::other("WebView2 event state is already borrowed"))?
                .last_navigation_allowed();
            Err(io::Error::other(format!(
                "WebView2 packaged entry navigation did not succeed (status={status}, policy_allowed={policy_allowed:?})"
            )))
        }
    }

    pub(super) fn navigate_unchecked(&mut self, uri: &str) -> io::Result<()> {
        let webview = self.webview.as_ref().ok_or_else(closed_error)?;
        let value = encode_utf16(uri, MAX_WEBVIEW_URI_UNITS)?;
        unsafe {
            webview
                .Navigate(PCWSTR(value.as_ptr()))
                .map_err(windows_error)
        }
    }

    pub(super) fn record_navigation(&self, uri: &str, allowed: bool) -> io::Result<()> {
        let mut copy = String::new();
        copy.try_reserve_exact(uri.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                "WebView2 URI reservation failed",
            )
        })?;
        copy.push_str(uri);
        let mut current = self
            .current_uri
            .try_borrow_mut()
            .map_err(|_| io::Error::other("WebView2 navigation state is already borrowed"))?;
        current.clear();
        current.push_str(&copy);
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| io::Error::other("WebView2 event state is already borrowed"))?;
        state.push(WebViewEvent::NavigationStarting { uri: copy, allowed });
        Ok(())
    }

    pub(super) fn collect_events(&mut self) -> io::Result<Vec<WebViewHostEvent>> {
        let window_events = self.window.poll_events()?;
        let (webview_events, overflowed) = self
            .state
            .try_borrow_mut()
            .map_err(|_| io::Error::other("WebView2 event state is already borrowed"))?
            .drain();
        if overflowed {
            return Err(io::Error::other("WebView2 event queue capacity exceeded"));
        }
        let mut events = Vec::new();
        events
            .try_reserve_exact(window_events.len().saturating_add(webview_events.len()))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::OutOfMemory,
                    "WebView2 event batch reservation failed",
                )
            })?;
        events.extend(window_events.into_iter().map(WebViewHostEvent::Window));
        events.extend(webview_events.into_iter().map(WebViewHostEvent::WebView));
        Ok(events)
    }
}
