//! One explicitly selected startup observation session.

use super::super::super::{config::WebViewConfig, pump::wait_for};
use super::state::{Shared, State};
use super::{cdp, completion, dom, native, registration::Registrations};
use crate::webview_startup::{SELECTION_ENV, Selection, TRACE_TARGET};
use serde_json::json;
use std::{cell::RefCell, io, rc::Rc, sync::mpsc, time::Instant};
use webview2_com::Microsoft::Web::WebView2::Win32::ICoreWebView2;

pub(in super::super) struct Startup {
    state: Shared,
    tokens: Registrations,
}

impl Startup {
    pub(in super::super) fn selected(
        webview: &ICoreWebView2,
        config: &WebViewConfig,
    ) -> io::Result<Option<Self>> {
        if !tracing::enabled!(target: TRACE_TARGET, tracing::Level::DEBUG) {
            return Ok(None);
        }
        let encoded = match std::env::var(SELECTION_ENV) {
            Ok(encoded) => encoded,
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(error) => return Err(io::Error::new(io::ErrorKind::InvalidInput, error)),
        };
        let selection = Selection::decode(&encoded, config.start_uri())?;
        let state = Rc::new(RefCell::new(State::new(selection)));
        let mut tokens =
            Registrations::new(webview, Rc::clone(&state)).map_err(io::Error::other)?;
        let result =
            native::install(&mut tokens, &state).and_then(|()| cdp::install(&mut tokens, &state));
        if let Err(error) = result {
            state.borrow_mut().error("observer registration", &error);
            return Err(io::Error::other(error));
        }
        state.borrow_mut().record(&json!({"kind":"registered"}));
        Ok(Some(Self { state, tokens }))
    }

    pub(in super::super) fn activate(&self, deadline: Instant) -> io::Result<()> {
        for method in ["Runtime.enable", "Log.enable"] {
            let (sender, receiver) = mpsc::sync_channel(1);
            if let Err(error) = completion::enable(
                &self.tokens.webview,
                Rc::downgrade(&self.state),
                method,
                sender,
            ) {
                self.state.borrow_mut().error(method, &error);
                return Err(io::Error::other(error));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if let Err(error) = wait_for(receiver, remaining) {
                self.state
                    .borrow_mut()
                    .record(&json!({"kind":"activation_incomplete","operation":method,
                    "error":error.to_string(),"native_code":error.raw_os_error()}));
                return Err(error);
            }
            self.state
                .borrow_mut()
                .record(&json!({"kind":"enabled","method":method}));
        }
        Ok(())
    }

    pub(in super::super) fn tick(&self) {
        dom::tick(&self.tokens.webview, &self.state);
    }

    pub(in super::super) fn close(&mut self) -> io::Result<()> {
        self.tokens.close()
    }
}
