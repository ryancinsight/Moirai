//! Test-only owner reuses canonical callbacks without spending the DOM quota.

use super::{
    input::Observer,
    state::{Shared, State},
};
use crate::webview_startup::{Components, Selection};
use serde_json::json;
use std::{cell::RefCell, io, rc::Rc};
use webview2_com::Microsoft::Web::WebView2::Win32::ICoreWebView2Controller;

pub(in super::super) struct FixtureInput {
    observer: Option<Observer>,
    state: Shared,
    closed: bool,
}

impl FixtureInput {
    pub(in super::super) fn checkpoint(&self) -> io::Result<(std::time::Instant, u64)> {
        self.observer
            .as_ref()
            .map(Observer::checkpoint)
            .ok_or_else(|| io::Error::other("controller probe already closed"))
    }

    pub(in super::super) fn new(
        controller: &ICoreWebView2Controller,
        mut selection: Selection,
    ) -> io::Result<Self> {
        // The URI/selectors still describe this actual fixture. This separate
        // owner installs controller receivers only, with no native/CDP tokens.
        selection.components = Components::Controller;
        let mut state = State::new(selection);
        state.fixture_input = true;
        let state = Rc::new(RefCell::new(state));
        let observer = match Observer::new(controller, &state) {
            Ok(observer) => observer,
            Err(error) => {
                let mut state = state.borrow_mut();
                state.failed = true;
                state.slot.close();
                state.metadata("closed", false);
                return Err(error);
            }
        };
        state.borrow_mut().record(&json!({"kind":"registered", "components":"controller",
            "native_receivers":0, "cdp_receivers":0, "controller_receivers":observer.registered()}));
        Ok(Self {
            observer: Some(observer),
            state,
            closed: false,
        })
    }

    pub(in super::super) fn flush(&self) {
        if let Some(observer) = &self.observer {
            observer.flush();
        }
    }

    pub(in super::super) fn close(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        self.state.borrow_mut().slot.close();
        // Canonical close closes its queue, flushes every retained callback,
        // and attempts every actual token removal before returning an error.
        let mut observer = self
            .observer
            .take()
            .ok_or_else(|| io::Error::other("independent controller owner missing before close"))?;
        let removed = observer.close();
        // Release this controller clone before the enclosing host Close.
        // A second canonical close cannot replace the first removal result.
        drop(observer);
        let mut state = self.state.borrow_mut();
        state.metadata("closed", removed.is_err());
        removed?;
        if state.failed || state.cleanup_failed {
            return Err(io::Error::other(
                "independent controller probe has native read, queue or cleanup errors",
            ));
        }
        Ok(())
    }
}

impl Drop for FixtureInput {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::error!(%error, "independent controller observation cleanup failed");
        }
    }
}

#[cfg(test)]
mod tests;
