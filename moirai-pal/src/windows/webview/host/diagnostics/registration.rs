//! Native registration tokens stay owned until removal before controller close.

use super::state::Shared;
use std::io;
use webview2_com::Microsoft::Web::WebView2::Win32::{
    ICoreWebView2, ICoreWebView2_2, ICoreWebView2ContentLoadingEventHandler,
    ICoreWebView2DOMContentLoadedEventHandler,
    ICoreWebView2DevToolsProtocolEventReceivedEventHandler,
    ICoreWebView2DevToolsProtocolEventReceiver, ICoreWebView2NavigationCompletedEventHandler,
    ICoreWebView2NavigationStartingEventHandler, ICoreWebView2ProcessFailedEventHandler,
    ICoreWebView2WebResourceResponseReceivedEventHandler,
};
use windows::core::Interface;

pub(super) struct Registrations {
    pub(super) webview: ICoreWebView2,
    pub(super) resources: ICoreWebView2_2,
    pub(super) content: Option<(i64, ICoreWebView2ContentLoadingEventHandler)>,
    pub(super) dom: Option<(i64, ICoreWebView2DOMContentLoadedEventHandler)>,
    pub(super) process: Option<(i64, ICoreWebView2ProcessFailedEventHandler)>,
    pub(super) response: Option<(i64, ICoreWebView2WebResourceResponseReceivedEventHandler)>,
    pub(super) starting: Option<(i64, ICoreWebView2NavigationStartingEventHandler)>,
    pub(super) completed: Option<(i64, ICoreWebView2NavigationCompletedEventHandler)>,
    pub(super) cdp: Vec<(
        ICoreWebView2DevToolsProtocolEventReceiver,
        i64,
        ICoreWebView2DevToolsProtocolEventReceivedEventHandler,
    )>,
    state: Shared,
    closed: bool,
}

impl Registrations {
    pub(super) fn new(webview: &ICoreWebView2, state: Shared) -> windows::core::Result<Self> {
        Ok(Self {
            webview: webview.clone(),
            resources: webview.cast().map_err(|error| {
                super::super::error::callback_context("Startup ICoreWebView2_2", error)
            })?,
            state,
            content: None,
            dom: None,
            process: None,
            response: None,
            starting: None,
            completed: None,
            cdp: Vec::new(),
            closed: false,
        })
    }

    pub(super) fn close(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        if let Ok(mut state) = self.state.try_borrow_mut() {
            let request = state.slot.snapshot();
            state.record(
                &serde_json::json!({"kind":"close_requested", "pending":request.pending,
                "generation":request.generation.to_string()}),
            );
            state.close_admission();
        }
        let mut first = None;
        if let Some((token, _)) = self.content.take() {
            // SAFETY: this owner retains the same WebView and native token.
            self.removed(
                "remove_ContentLoading",
                unsafe { self.webview.remove_ContentLoading(token) },
                &mut first,
            );
        }
        if let Some((token, _)) = self.dom.take() {
            // SAFETY: the registration belongs to this live versioned interface.
            self.removed(
                "remove_DOMContentLoaded",
                unsafe { self.resources.remove_DOMContentLoaded(token) },
                &mut first,
            );
        }
        if let Some((token, _)) = self.process.take() {
            // SAFETY: the registration belongs to this live WebView.
            self.removed(
                "remove_ProcessFailed",
                unsafe { self.webview.remove_ProcessFailed(token) },
                &mut first,
            );
        }
        if let Some((token, _)) = self.response.take() {
            // SAFETY: the registration belongs to this live versioned interface.
            self.removed(
                "remove_WebResourceResponseReceived",
                unsafe { self.resources.remove_WebResourceResponseReceived(token) },
                &mut first,
            );
        }
        if let Some((token, _)) = self.starting.take() {
            // SAFETY: the registration belongs to this live WebView.
            self.removed(
                "remove_NavigationStarting",
                unsafe { self.webview.remove_NavigationStarting(token) },
                &mut first,
            );
        }
        if let Some((token, _)) = self.completed.take() {
            // SAFETY: the registration belongs to this live WebView.
            self.removed(
                "remove_NavigationCompleted",
                unsafe { self.webview.remove_NavigationCompleted(token) },
                &mut first,
            );
        }
        for (receiver, token, _) in std::mem::take(&mut self.cdp) {
            // SAFETY: receiver/token are retained together from registration.
            self.removed(
                "remove_DevToolsProtocolEventReceived",
                unsafe { receiver.remove_DevToolsProtocolEventReceived(token) },
                &mut first,
            );
        }
        if let Ok(mut state) = self.state.try_borrow_mut() {
            state.metadata("closed", first.is_some());
        }
        first.map_or(Ok(()), Err)
    }

    pub(super) fn counts(&self) -> (usize, usize) {
        let native = [
            self.content.is_some(),
            self.dom.is_some(),
            self.process.is_some(),
            self.response.is_some(),
            self.starting.is_some(),
            self.completed.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count();
        (native, self.cdp.len())
    }

    fn removed(
        &self,
        operation: &str,
        result: windows::core::Result<()>,
        first: &mut Option<io::Error>,
    ) {
        if let Err(error) = result {
            if let Ok(mut state) = self.state.try_borrow_mut() {
                state.error(operation, &error);
            }
            first.get_or_insert_with(|| io::Error::other(error));
        }
    }
}

impl Drop for Registrations {
    fn drop(&mut self) {
        // close already retains each error through the bounded dedicated path
        // and emits the fixed cleanup_failed metadata before this owner drops.
        let _ = self.close();
    }
}
