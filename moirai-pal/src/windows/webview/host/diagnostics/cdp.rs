//! CDP passive event receivers register before their domains are enabled.

use super::super::{
    error::{callback_context, callback_error},
    text::read_task_mem_message,
};
use super::{
    registration::Registrations,
    state::{Shared, observe},
};
use serde_json::json;
use std::rc::Rc;
use webview2_com::DevToolsProtocolEventReceivedEventHandler;
use windows::core::{HSTRING, PWSTR};

pub(super) fn install(tokens: &mut Registrations, shared: &Shared) -> windows::core::Result<()> {
    for name in [
        "Runtime.exceptionThrown",
        "Log.entryAdded",
        "DOM.documentUpdated",
    ] {
        let name_wide = HSTRING::from(name);
        // SAFETY: name_wide is valid terminated text copied by the SDK.
        let receiver = unsafe {
            tokens
                .webview
                .GetDevToolsProtocolEventReceiver(&name_wide)
                .map_err(|error| {
                    callback_context("Startup GetDevToolsProtocolEventReceiver", error)
                })?
        };
        let weak = Rc::downgrade(shared);
        // The SDK's native callback boundary requires runtime-erased ownership.
        let handler =
            DevToolsProtocolEventReceivedEventHandler::create(Box::new(move |_, args| {
                observe(&weak, name, |state| {
                    let args =
                        args.ok_or_else(|| callback_error("Startup CDP event arguments missing"))?;
                    let mut raw = PWSTR::null();
                    // SAFETY: native getter transfers one terminated CoTaskMem string.
                    unsafe {
                        args.ParameterObjectAsJson(&mut raw)?;
                    }
                    // SAFETY: converter owns/frees the transferred buffer and admits
                    // its UTF16/UTF8 representation before making a Rust copy.
                    let encoded = unsafe { read_task_mem_message(raw) }?;
                    let value: serde_json::Value =
                        serde_json::from_str(&encoded).map_err(|error| {
                            windows::core::Error::new(
                                windows::Win32::Foundation::E_ABORT,
                                format!("WebView2 CDP event JSON: {error}"),
                            )
                        })?;
                    if name == "DOM.documentUpdated" {
                        state.invalidate_nodes();
                    }
                    state.record(&json!({"kind":"cdp_event","name":name,"parameters":value}));
                    Ok(())
                })
            }));
        let mut token = 0;
        // SAFETY: callback and receiver are retained together until removal.
        unsafe {
            receiver
                .add_DevToolsProtocolEventReceived(&handler, &mut token)
                .map_err(|error| {
                    callback_context("Startup add_DevToolsProtocolEventReceived", error)
                })?;
        }
        tokens.cdp.push((receiver, token, handler));
    }
    Ok(())
}
