//! One sequential, generation-bound DOM chain on the existing native pump.

use super::{
    completion,
    protocol::{Failure, Stage},
    state::{Document, Shared, State},
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

pub(super) fn tick(webview: &ICoreWebView2, shared: &Shared, admit_new_sample: bool) {
    let command = {
        let Ok(mut state) = shared.try_borrow_mut() else {
            return;
        };
        next_command(&mut state, admit_new_sample)
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

fn next_command(state: &mut State, admit_new_sample: bool) -> Option<Command> {
    if state.slot.snapshot().closed || state.failed || state.budget.snapshot().exhausted {
        if let Some((id, _, _)) = state.reply.take() {
            state.slot.finish(id);
        }
        return None;
    }
    if let Some((id, stage, reply)) = state.reply.take() {
        if !state.slot.is_current(id) {
            state.slot.finish(id);
            return None;
        }
        match reply.and_then(|value| {
            advance(state, id, stage, &value).map_err(|error| Failure::protocol(error, &value))
        }) {
            Ok(command) => command,
            Err(error) => {
                state.slot.finish(id);
                state.failed = true;
                state.record(&error.value("DOM sample"));
                None
            }
        }
    } else if admit_new_sample && state.document == Document::Available {
        match state.slot.start() {
            Ok(Start::Begun(id)) => Some(Command {
                id,
                stage: Stage::Document,
                method: "DOM.getDocument",
                parameters: json!({"depth":0,"pierce":false}),
            }),
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
}

fn advance(
    state: &mut State,
    id: RequestId,
    stage: Stage,
    value: &Value,
) -> windows::core::Result<Option<Command>> {
    match stage {
        Stage::Document => {
            let uri = value
                .get("root")
                .and_then(|root| root.get("documentURL"))
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("DOM document omits its documentURL"))?;
            if uri != state.selection.start_uri {
                return Err(invalid("DOM documentURL differs from selected startup URI"));
            }
            let root = node(value.get("root").and_then(|root| root.get("nodeId")))?;
            if root == 0 {
                return Err(invalid("DOM document has no root node"));
            }
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
