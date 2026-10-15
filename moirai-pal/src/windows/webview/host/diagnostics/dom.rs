//! One sequential, generation-bound DOM chain on the existing native pump.

use super::{
    completion,
    protocol::{Failure, Stage},
    state::{Document, DocumentRoot, Shared, State},
};
use crate::webview_startup::{RequestId, Start};
use serde_json::{Value, json};
use std::rc::Rc;
use webview2_com::Microsoft::Web::WebView2::Win32::ICoreWebView2;

struct Command {
    id: RequestId,
    stage: Stage,
    method: &'static str,
    parameters: Value,
}

pub(super) fn tick(webview: &ICoreWebView2, shared: &Shared) {
    let command = {
        let Ok(mut state) = shared.try_borrow_mut() else {
            return;
        };
        if state.slot.snapshot().closed || state.failed || state.budget.snapshot().exhausted {
            if let Some((id, _, _)) = state.reply.take() {
                state.slot.finish(id);
            }
            return;
        }
        if let Some((id, stage, reply)) = state.reply.take() {
            if !state.slot.is_current(id) {
                state.slot.finish(id);
                return;
            }
            match reply.and_then(|value| {
                advance(&mut state, id, stage, &value)
                    .map_err(|error| Failure::protocol(error, &value))
            }) {
                Ok(command) => command,
                Err(error) => {
                    state.slot.finish(id);
                    state.failed = true;
                    state.record(&error.value("DOM sample"));
                    None
                }
            }
        } else if state.document == Document::Available {
            match state.slot.start() {
                Ok(Start::Begun(id)) => match document_command(&state, id) {
                    Ok(command) => Some(command),
                    Err(error) => {
                        state.slot.finish(id);
                        state.failed = true;
                        state.error("DOM.document identity", &error);
                        None
                    }
                },
                Ok(Start::Busy | Start::Closed) => None,
                Err(error) => {
                    state.failed = true;
                    state.record(&json!({"kind":"state_error","operation":"DOM.start","message":error.to_string()}));
                    None
                }
            }
        } else {
            None
        }
    };
    if let Some(command) = command
        && let Err(error) = completion::dom(
            webview,
            Rc::downgrade(shared),
            command.method,
            &command.parameters,
            command.id,
            command.stage,
        )
        && let Ok(mut state) = shared.try_borrow_mut()
    {
        state.slot.finish(command.id);
        state.failed = true;
        state.error(command.method, &error);
    }
}

fn document_command(state: &State, id: RequestId) -> windows::core::Result<Command> {
    let generation = state.slot.snapshot().generation;
    if !state.slot.is_current(id) || state.root.is_some_and(|root| root.generation != generation) {
        return Err(invalid("DOM document root belongs to a stale generation"));
    }
    let existing = state.root.map(|root| root.node);
    let (method, parameters) = match existing {
        Some(root) => (
            "DOM.describeNode",
            json!({"nodeId":root,"depth":0,"pierce":false}),
        ),
        None => ("DOM.getDocument", json!({"depth":0,"pierce":false})),
    };
    Ok(Command {
        id,
        stage: Stage::Document { existing },
        method,
        parameters,
    })
}

fn advance(
    state: &mut State,
    id: RequestId,
    stage: Stage,
    value: &Value,
) -> windows::core::Result<Option<Command>> {
    match stage {
        Stage::Document { existing } => {
            let key = if existing.is_some() { "node" } else { "root" };
            let document = value.get(key);
            let uri = document
                .and_then(|root| root.get("documentURL"))
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("DOM document omits its documentURL"))?;
            if uri != state.selection.start_uri {
                return Err(invalid("DOM documentURL differs from selected startup URI"));
            }
            let root = node(document.and_then(|root| root.get("nodeId")))?;
            if root == 0 {
                return Err(invalid("DOM document has no root node"));
            }
            if existing.is_some_and(|expected| root != expected) {
                return Err(invalid(
                    "DOM described document identity differs from the bound root",
                ));
            }
            // Only the identity is retained. Every chain revalidates the live
            // URL and queries all current selectors/HTML. Actual invalidation
            // revokes this generation before any stale completion is admitted.
            state.root = Some(DocumentRoot {
                node: root,
                generation: state.slot.snapshot().generation,
            });
            Ok(Some(query(state, id, 0, root)))
        }
        Stage::Query { index, root } => {
            let selected = node(value.get("nodeId"))?;
            if selected == 0 {
                state.changed(index, Value::Null);
                next(state, id, index, root)
            } else {
                Ok(Some(Command {
                    id,
                    stage: Stage::Html { index, root },
                    method: "DOM.getOuterHTML",
                    parameters: json!({"nodeId":selected}),
                }))
            }
        }
        Stage::Html { index, root } => {
            let html = value
                .get("outerHTML")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("DOM outerHTML response omits its string"))?;
            state.changed(index, Value::String(html.to_owned()));
            next(state, id, index, root)
        }
    }
}

fn next(
    state: &mut State,
    id: RequestId,
    index: usize,
    root: i64,
) -> windows::core::Result<Option<Command>> {
    if state.failed || state.budget.snapshot().exhausted {
        state.slot.finish(id);
        return Ok(None);
    }
    let index = index
        .checked_add(1)
        .ok_or_else(|| invalid("DOM selector index exhausted"))?;
    if index < state.selection.selectors.len() {
        return Ok(Some(query(state, id, index, root)));
    }
    state.slot.finish(id);
    state.samples = state
        .samples
        .checked_add(1)
        .ok_or_else(|| invalid("DOM sample count exhausted"))?;
    Ok(None)
}

fn query(state: &State, id: RequestId, index: usize, root: i64) -> Command {
    Command {
        id,
        stage: Stage::Query { index, root },
        method: "DOM.querySelector",
        parameters: json!({"nodeId":root,"selector":state.selection.selectors[index]}),
    }
}

fn node(value: Option<&Value>) -> windows::core::Result<i64> {
    value
        .and_then(Value::as_i64)
        .filter(|value| (0..=i64::from(i32::MAX)).contains(value))
        .ok_or_else(|| invalid("DOM response contains an invalid NodeId"))
}

fn invalid(message: &str) -> windows::core::Error {
    windows::core::Error::new(windows::Win32::Foundation::E_ABORT, message)
}

#[cfg(test)]
mod tests;
