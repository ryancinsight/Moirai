//! WebView2 callback registration and bounded event translation.

use std::{io, rc::Rc};

use webview2_com::Microsoft::Web::WebView2::Win32::{
    ICoreWebView2, ICoreWebView2NavigationCompletedEventHandler,
    ICoreWebView2NavigationStartingEventHandler, ICoreWebView2NewWindowRequestedEventHandler,
    ICoreWebView2WebMessageReceivedEventHandler,
};
use webview2_com::{
    NavigationCompletedEventHandler, NavigationStartingEventHandler,
    NewWindowRequestedEventHandler, WebMessageReceivedEventHandler,
};
use windows::core::{BOOL, PWSTR};

use super::super::event::WebViewEvent;
use super::text::{read_task_mem_message, read_task_mem_uri};
use super::{WebViewHost, callback_error, closed_error, push_event, windows_error};

impl WebViewHost {
    pub(super) fn install_callbacks(&mut self) -> io::Result<()> {
        let webview = self.webview.as_ref().ok_or_else(closed_error)?.clone();
        let state = Rc::clone(&self.state);
        let current_uri = Rc::clone(&self.current_uri);
        let policy = self.config.clone();
        let navigation = NavigationStartingEventHandler::create(Box::new(move |_sender, args| {
            let Some(args) = args else {
                return Err(callback_error(
                    "WebView2 navigation callback omitted arguments",
                ));
            };
            let mut raw_uri = PWSTR::null();
            unsafe { args.Uri(&mut raw_uri)? };
            let uri = read_task_mem_uri(raw_uri).map_err(|_| {
                callback_error("WebView2 navigation callback returned an invalid URI")
            })?;
            let allowed = policy.allows(&uri);
            if let Ok(mut current) = current_uri.try_borrow_mut() {
                current.clear();
                current.push_str(&uri);
            } else {
                return Err(callback_error(
                    "WebView2 navigation state is already borrowed",
                ));
            }
            if !allowed {
                unsafe { args.SetCancel(true)? };
            }
            push_event(&state, WebViewEvent::NavigationStarting { uri, allowed })
        }));
        let mut navigation_token = 0;
        unsafe {
            webview
                .add_NavigationStarting(&navigation, &mut navigation_token)
                .map_err(windows_error)?;
        }
        self.callbacks.navigation_starting = Some((navigation_token, navigation));

        let state = Rc::clone(&self.state);
        let current_uri = Rc::clone(&self.current_uri);
        let completed = NavigationCompletedEventHandler::create(Box::new(move |_sender, args| {
            let Some(args) = args else {
                return Err(callback_error(
                    "WebView2 completion callback omitted arguments",
                ));
            };
            let mut success = BOOL(0);
            unsafe { args.IsSuccess(&mut success)? };
            let uri = current_uri
                .try_borrow()
                .map_err(|_| callback_error("WebView2 navigation state is already borrowed"))?
                .clone();
            push_event(
                &state,
                WebViewEvent::NavigationCompleted {
                    uri,
                    success: success.as_bool(),
                },
            )
        }));
        let mut completed_token = 0;
        unsafe {
            webview
                .add_NavigationCompleted(&completed, &mut completed_token)
                .map_err(windows_error)?;
        }
        self.callbacks.navigation_completed = Some((completed_token, completed));

        let state = Rc::clone(&self.state);
        let new_window = NewWindowRequestedEventHandler::create(Box::new(move |_sender, args| {
            let Some(args) = args else {
                return Err(callback_error(
                    "WebView2 new-window callback omitted arguments",
                ));
            };
            let mut raw_uri = PWSTR::null();
            unsafe { args.Uri(&mut raw_uri)? };
            let uri = read_task_mem_uri(raw_uri).map_err(|_| {
                callback_error("WebView2 new-window callback returned an invalid URI")
            })?;
            unsafe { args.SetHandled(true)? };
            push_event(&state, WebViewEvent::NewWindowDenied { uri })
        }));
        let mut new_window_token = 0;
        unsafe {
            webview
                .add_NewWindowRequested(&new_window, &mut new_window_token)
                .map_err(windows_error)?;
        }
        self.callbacks.new_window = Some((new_window_token, new_window));

        let state = Rc::clone(&self.state);
        let policy = self.config.clone();
        let message = WebMessageReceivedEventHandler::create(Box::new(move |_sender, args| {
            let Some(args) = args else {
                return Err(callback_error(
                    "WebView2 message callback omitted arguments",
                ));
            };
            let mut raw_source = PWSTR::null();
            unsafe { args.Source(&mut raw_source)? };
            let source = read_task_mem_uri(raw_source).map_err(|_| {
                callback_error("WebView2 message callback returned an invalid source URI")
            })?;
            let mut raw_message = PWSTR::null();
            unsafe { args.WebMessageAsJson(&mut raw_message)? };
            let json = read_task_mem_message(raw_message)
                .map_err(|_| callback_error("WebView2 message exceeded the bounded JSON limit"))?;
            if !policy.allows(&source) {
                return push_event(&state, WebViewEvent::MessageRejected { source });
            }
            push_event(&state, WebViewEvent::Message { source, json })
        }));
        let mut message_token = 0;
        unsafe {
            webview
                .add_WebMessageReceived(&message, &mut message_token)
                .map_err(windows_error)?;
        }
        self.callbacks.message = Some((message_token, message));
        Ok(())
    }
}

pub(super) fn remove_navigation_starting(
    webview: &ICoreWebView2,
    callback: &mut Option<(i64, ICoreWebView2NavigationStartingEventHandler)>,
    first_error: &mut Option<io::Error>,
) {
    if let Some((token, _handler)) = callback.take()
        && let Err(error) =
            unsafe { webview.remove_NavigationStarting(token) }.map_err(windows_error)
    {
        first_error.get_or_insert(error);
    }
}

pub(super) fn remove_navigation_completed(
    webview: &ICoreWebView2,
    callback: &mut Option<(i64, ICoreWebView2NavigationCompletedEventHandler)>,
    first_error: &mut Option<io::Error>,
) {
    if let Some((token, _handler)) = callback.take()
        && let Err(error) =
            unsafe { webview.remove_NavigationCompleted(token) }.map_err(windows_error)
    {
        first_error.get_or_insert(error);
    }
}

pub(super) fn remove_new_window(
    webview: &ICoreWebView2,
    callback: &mut Option<(i64, ICoreWebView2NewWindowRequestedEventHandler)>,
    first_error: &mut Option<io::Error>,
) {
    if let Some((token, _handler)) = callback.take()
        && let Err(error) =
            unsafe { webview.remove_NewWindowRequested(token) }.map_err(windows_error)
    {
        first_error.get_or_insert(error);
    }
}

pub(super) fn remove_message(
    webview: &ICoreWebView2,
    callback: &mut Option<(i64, ICoreWebView2WebMessageReceivedEventHandler)>,
    first_error: &mut Option<io::Error>,
) {
    if let Some((token, _handler)) = callback.take()
        && let Err(error) =
            unsafe { webview.remove_WebMessageReceived(token) }.map_err(windows_error)
    {
        first_error.get_or_insert(error);
    }
}
