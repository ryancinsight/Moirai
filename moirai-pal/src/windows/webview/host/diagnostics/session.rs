//! One explicitly selected startup observation session.

use super::super::super::{config::WebViewConfig, pump::wait_for};
use super::state::{Shared, State};
use super::{cdp, completion, dom, input, native, registration::Registrations};
use crate::webview_startup::{SELECTION_ENV, Selection, TRACE_TARGET};
use serde_json::json;
use std::{cell::RefCell, io, rc::Rc, sync::mpsc, time::Instant};
use webview2_com::Microsoft::Web::WebView2::Win32::{ICoreWebView2, ICoreWebView2Controller};

pub(in super::super) struct Startup {
    state: Shared,
    tokens: Registrations,
    input: Option<input::Observer>,
}

impl Startup {
    pub(in super::super) fn selected(
        webview: &ICoreWebView2,
        controller: &ICoreWebView2Controller,
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
        let components = selection.components;
        let state = Rc::new(RefCell::new(State::new(selection)));
        let mut tokens =
            Registrations::new(webview, Rc::clone(&state)).map_err(io::Error::other)?;
        let result = if components.native() {
            native::install(&mut tokens, &state)
        } else {
            Ok(())
        }
        .and_then(|()| {
            if components.cdp() {
                cdp::install(&mut tokens, &state)
            } else {
                Ok(())
            }
        });
        if let Err(error) = result {
            state.borrow_mut().error("observer registration", &error);
            return Err(io::Error::other(error));
        }
        let input = components
            .controller()
            .then(|| input::Observer::new(controller, &state))
            .transpose()?;
        let (native_receivers, cdp_receivers) = tokens.counts();
        let controller_receivers = input.as_ref().map_or(0, input::Observer::registered);
        state.borrow_mut().record(
            &json!({"kind":"registered", "components":components.as_str(),
            "native_receivers":native_receivers, "cdp_receivers":cdp_receivers,
            "controller_receivers":controller_receivers}),
        );
        Ok(Some(Self {
            state,
            tokens,
            input,
        }))
    }

    pub(in super::super) fn activate(&self, deadline: Instant) -> io::Result<()> {
        if !self.cdp_enabled() {
            return Ok(());
        }
        let components = self.state.borrow().selection.components;
        for method in ["Runtime.enable", "Log.enable"]
            .into_iter()
            .chain(components.enables_dom().then_some("DOM.enable"))
        {
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

    pub(in super::super) fn tick(&self, admit_new_sample: bool) {
        if let Some(input) = &self.input {
            input.flush();
        }
        if self.sampling_enabled() {
            dom::tick(&self.tokens.webview, &self.state, admit_new_sample);
        }
    }

    pub(in super::super) fn repeating_samples(&self) -> bool {
        // Exactly the policies that admit another sample after the first.
        self.state.borrow().selection.components.admits_sample(1)
    }

    pub(in super::super) fn begin_wait_turn(&self) -> io::Result<()> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| io::Error::other("Startup wait state is borrowed"))?;
        state.wait_turns_attempted = state
            .wait_turns_attempted
            .checked_add(1)
            .ok_or_else(|| io::Error::other("Startup wait turn count exhausted"))?;
        Ok(())
    }

    pub(in super::super) fn close(&mut self) -> io::Result<()> {
        let input = self.input.as_mut().map_or(Ok(()), input::Observer::close);
        let tokens = self.tokens.close();
        input.and(tokens)
    }

    fn cdp_enabled(&self) -> bool {
        // This short borrow ends before any native call or callback dispatch.
        self.state.borrow().selection.components.cdp()
    }

    fn sampling_enabled(&self) -> bool {
        let state = self.state.borrow();
        state.selection.components.admits_sample(state.samples)
    }
}
