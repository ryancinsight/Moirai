//! Bounded native observation state; callbacks borrow it only synchronously.

use crate::webview_startup::{RecordAdmission, RequestId, RequestSlot, Selection, TraceBudget};
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

use super::protocol::{Reply, Stage};

pub(super) type Shared = Rc<RefCell<State>>;
pub(super) type Borrowed = Weak<RefCell<State>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Document {
    Outside,
    Loading,
    Available,
}

pub(super) struct State {
    pub(super) selection: Selection,
    pub(super) budget: TraceBudget,
    pub(super) slot: RequestSlot,
    pub(super) reply: Option<(RequestId, Stage, Reply)>,
    pub(super) document: Document,
    pub(super) navigation: Option<u64>,
    pub(super) failed: bool,
    pub(super) cleanup_failed: bool,
    pub(super) samples: u64,
    pub(super) dom_calls_attempted: u64,
    pub(super) activation_calls_attempted: u64,
    pub(super) stale: u64,
    pub(super) last: Vec<Option<Value>>,
    metadata_sent: bool,
}

impl State {
    pub(super) fn new(selection: Selection) -> Self {
        let last = vec![None; selection.selectors.len()];
        Self {
            selection,
            last,
            budget: TraceBudget::default(),
            slot: RequestSlot::default(),
            reply: None,
            document: Document::Outside,
            navigation: None,
            failed: false,
            cleanup_failed: false,
            samples: 0,
            dom_calls_attempted: 0,
            activation_calls_attempted: 0,
            stale: 0,
            metadata_sent: false,
        }
    }

    pub(super) fn record(&mut self, value: &Value) -> bool {
        match self.budget.record(value) {
            Ok(RecordAdmission::Admitted(encoded)) => {
                emit(&encoded);
                true
            }
            Ok(RecordAdmission::Exhausted { first: true }) => {
                self.metadata("quota_exhausted", false);
                false
            }
            Ok(RecordAdmission::Exhausted { first: false }) => false,
            Err(_) => {
                self.failed = true;
                false
            }
        }
    }

    pub(super) fn error(&mut self, operation: &str, error: &windows::core::Error) {
        self.record(&json!({"kind":"native_error","operation":operation,
            "hresult":error.code().0,"message":error.to_string()}));
    }

    pub(super) fn navigation_started(&mut self, id: u64, uri: &str) {
        self.invalidate_nodes();
        self.navigation = Some(id);
        self.document = if uri == self.selection.start_uri {
            Document::Loading
        } else {
            Document::Outside
        };
    }

    pub(super) fn navigation_completed(&mut self, id: u64, success: bool) -> bool {
        let current = self.navigation == Some(id);
        if current {
            self.document = if success && self.document != Document::Outside {
                Document::Available
            } else {
                Document::Outside
            };
        }
        current
    }

    pub(super) fn dom_loaded(&mut self, id: u64) -> bool {
        let current = self.navigation == Some(id);
        if current && self.document != Document::Outside {
            self.document = Document::Available;
        }
        current
    }

    pub(super) fn invalidate_nodes(&mut self) {
        if let Err(error) = self.slot.invalidate() {
            self.failed = true;
            self.record(
                &json!({"kind":"state_error","operation":"invalidate","message":error.to_string()}),
            );
        }
        self.last.fill(None);
    }

    pub(super) fn changed(&mut self, selector: usize, value: Value) {
        if self.last[selector].as_ref() != Some(&value) {
            let observation = json!({"kind":"dom_status","selector":self.selection.selectors[selector],
                "generation":self.slot.snapshot().generation.to_string(),"value":value});
            if self.record(&observation) {
                self.last[selector] = Some(value);
            }
        }
    }

    pub(super) fn metadata(&mut self, kind: &str, cleanup_failed: bool) {
        if kind == "closed" && self.metadata_sent {
            return;
        }
        let budget = self.budget.snapshot();
        let request = self.slot.snapshot();
        let encoded = json!({"kind":kind,"records":budget.records,"bytes":budget.bytes,
            "omitted":budget.omitted.to_string(),"exhausted":budget.exhausted,
            "generation":request.generation.to_string(),"pending":request.pending,
            "closed":request.closed,"samples":self.samples.to_string(),"stale":self.stale.to_string(),
            "dom_calls_attempted":self.dom_calls_attempted.to_string(),
            "activation_calls_attempted":self.activation_calls_attempted.to_string(),
            "failed":self.failed,"cleanup_failed":cleanup_failed || self.cleanup_failed}).to_string();
        emit(&encoded);
        if kind == "closed" {
            self.metadata_sent = true;
        }
    }
}

pub(super) fn observe(
    state: &Borrowed,
    operation: &str,
    action: impl FnOnce(&mut State) -> windows::core::Result<()>,
) -> windows::core::Result<()> {
    let Some(state) = state.upgrade() else {
        return Ok(());
    };
    let mut state = state.try_borrow_mut().map_err(|_| {
        super::super::error::callback_error(
            "WebView2 startup observation state is already borrowed",
        )
    })?;
    if state.slot.snapshot().closed {
        return Ok(());
    }
    if let Err(error) = action(&mut state) {
        state.error(operation, &error);
        return Err(error);
    }
    Ok(())
}

fn emit(encoded: &str) {
    tracing::debug!(target: crate::webview_startup::TRACE_TARGET,
        event = "webview_startup_observation", observation = %encoded);
}

#[cfg(test)]
mod tests;
