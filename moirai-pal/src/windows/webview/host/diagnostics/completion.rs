//! Borrowed COM reply admission before the first Rust payload copy.

use super::{
    protocol::{self, Stage},
    state::Borrowed,
};
use crate::webview_startup::RequestId;
use serde_json::Value;
use std::{cell::Cell, sync::mpsc::SyncSender, time::Instant};
use webview2_com::Microsoft::Web::WebView2::Win32::{
    ICoreWebView2, ICoreWebView2CallDevToolsProtocolMethodCompletedHandler,
    ICoreWebView2CallDevToolsProtocolMethodCompletedHandler_Impl,
};
use windows::core::{HRESULT, HSTRING, PCWSTR, implement};

#[implement(ICoreWebView2CallDevToolsProtocolMethodCompletedHandler)]
struct Completion {
    completed: Cell<bool>,
    state: Borrowed,
    destination: Destination,
}

enum Destination {
    Enable(SyncSender<windows::core::Result<()>>),
    Dom { id: RequestId, stage: Stage },
}

impl ICoreWebView2CallDevToolsProtocolMethodCompletedHandler_Impl for Completion_Impl {
    fn Invoke(&self, errorcode: HRESULT, result: &PCWSTR) -> windows::core::Result<()> {
        if self.completed.replace(true) {
            return Err(super::super::error::callback_error(
                "WebView2 startup completion invoked twice",
            ));
        }
        let Some(owner) = self.state.upgrade() else {
            return Ok(());
        };
        {
            let mut state = owner.try_borrow_mut().map_err(|_| {
                super::super::error::callback_error(
                    "WebView2 startup reply state is already borrowed",
                )
            })?;
            let current = match &self.destination {
                Destination::Enable(_) => !state.slot.snapshot().closed,
                Destination::Dom { id, .. } => state.slot.is_current(*id),
            };
            if !current {
                if let Destination::Dom { id, .. } = &self.destination {
                    state.slot.finish(*id);
                    state.stale = state.stale.checked_add(1).ok_or_else(|| {
                        super::super::error::callback_error("WebView2 stale reply count exhausted")
                    })?;
                }
                return Ok(());
            }
        }
        // SAFETY: the SDK owns this terminated reply until Invoke returns;
        // protocol admission bounds scans/copies and retains failed status/body.
        let reply = unsafe { protocol::read(errorcode, result) };
        match &self.destination {
            Destination::Enable(sender) => {
                if let Some(state) = self.state.upgrade()
                    && let Ok(mut state) = state.try_borrow_mut()
                    && let Err(error) = &reply
                {
                    state.record(&error.value("CDP.enable completion"));
                }
                // A timed-out owner has already recorded incomplete activation;
                // its late callback cannot revive the closed host.
                let _ = sender.try_send(reply.map(|_| ()).map_err(|failure| failure.error));
            }
            Destination::Dom { id, stage } => {
                if let Some(state) = self.state.upgrade() {
                    let mut state = state.try_borrow_mut().map_err(|_| {
                        super::super::error::callback_error(
                            "WebView2 startup reply state is already borrowed",
                        )
                    })?;
                    if state.slot.is_current(*id) {
                        state.reply = Some((*id, *stage, reply));
                    } else {
                        state.slot.finish(*id);
                        state.stale = state.stale.checked_add(1).ok_or_else(|| {
                            super::super::error::callback_error(
                                "WebView2 stale reply count exhausted",
                            )
                        })?;
                    }
                }
            }
        }
        Ok(())
    }
}

pub(super) fn enable(
    webview: &ICoreWebView2,
    state: Borrowed,
    method: &str,
    sender: SyncSender<windows::core::Result<()>>,
    deadline: Instant,
) -> windows::core::Result<()> {
    let admitted = issue(
        webview,
        state,
        method,
        &Value::Object(serde_json::Map::new()),
        Destination::Enable(sender),
        || Instant::now() < deadline,
    )?;
    if admitted {
        Ok(())
    } else {
        Err(windows::core::Error::new(
            windows::Win32::Foundation::ERROR_TIMEOUT.to_hresult(),
            "WebView2 activation not admitted before the original deadline",
        ))
    }
}

pub(super) fn dom(
    webview: &ICoreWebView2,
    state: Borrowed,
    method: &str,
    parameters: &Value,
    id: RequestId,
    stage: Stage,
    can_issue: impl Fn() -> bool,
) -> windows::core::Result<bool> {
    issue(
        webview,
        state,
        method,
        parameters,
        Destination::Dom { id, stage },
        can_issue,
    )
}

fn issue(
    webview: &ICoreWebView2,
    state: Borrowed,
    method: &str,
    parameters: &Value,
    destination: Destination,
    can_issue: impl Fn() -> bool,
) -> windows::core::Result<bool> {
    let request = match &destination {
        Destination::Dom { id, .. } => Some(*id),
        Destination::Enable(_) => None,
    };
    // No diagnostic state borrow crosses setup or the actual SDK call.
    let callback: ICoreWebView2CallDevToolsProtocolMethodCompletedHandler = Completion {
        completed: Cell::new(false),
        state: state.clone(),
        destination,
    }
    .into();
    let native_method = HSTRING::from(method);
    let parameters = HSTRING::from(parameters.to_string());
    if !admit(&state, method, request, can_issue)? {
        return Ok(false);
    }
    // SAFETY: the SDK copies strings and retains callback. No tracing,
    // dispatch, serialization or retained state borrow follows admission.
    unsafe { webview.CallDevToolsProtocolMethod(&native_method, &parameters, &callback)? };
    Ok(true)
}

/// The predicate is evaluated after all native argument/callback preparation.
/// Denial neither counts an SDK attempt nor consumes the caller's command.
fn admit(
    state: &Borrowed,
    method: &str,
    request: Option<RequestId>,
    can_issue: impl Fn() -> bool,
) -> windows::core::Result<bool> {
    if !can_issue() {
        return Ok(false);
    }
    let owner = state.upgrade().ok_or_else(|| {
        super::super::error::callback_error("WebView2 command observation owner expired")
    })?;
    let mut owner = owner.try_borrow_mut().map_err(|_| {
        super::super::error::callback_error("WebView2 command counter state is borrowed")
    })?;
    match request {
        Some(id) if !owner.slot.is_current(id) => {
            return Ok(false);
        }
        None if owner.slot.snapshot().closed => {
            return Err(super::super::error::callback_error(
                "WebView2 activation observation owner closed",
            ));
        }
        _ => {}
    }
    owner.command_attempt(method, request.is_some())?;
    Ok(true)
}

#[cfg(test)]
mod tests;
