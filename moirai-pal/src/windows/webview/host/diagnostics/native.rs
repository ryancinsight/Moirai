//! Passive native lifecycle and resource observations; no response body access.

use super::super::{
    error::{callback_context, callback_error},
    text::{read_task_mem_message, read_task_mem_uri},
};
use super::{
    registration::Registrations,
    state::{Shared, observe},
};
use serde_json::json;
use std::rc::Rc;
use webview2_com::Microsoft::Web::WebView2::Win32::COREWEBVIEW2_PROCESS_FAILED_KIND;
use webview2_com::{
    ContentLoadingEventHandler, DOMContentLoadedEventHandler, NavigationCompletedEventHandler,
    NavigationStartingEventHandler, ProcessFailedEventHandler,
    WebResourceResponseReceivedEventHandler,
};
use windows::core::{BOOL, HSTRING, PWSTR};

pub(super) fn install(tokens: &mut Registrations, shared: &Shared) -> windows::core::Result<()> {
    navigation(tokens, shared)?;
    content(tokens, shared)?;
    resources(tokens, shared)?;
    process(tokens, shared)
}

fn navigation(tokens: &mut Registrations, shared: &Shared) -> windows::core::Result<()> {
    let weak = Rc::downgrade(shared);
    // The SDK's native callback boundary requires runtime-erased closure ownership.
    let handler = NavigationStartingEventHandler::create(Box::new(move |_, args| {
        observe(&weak, "NavigationStarting", |state| {
            let args =
                args.ok_or_else(|| callback_error("Startup navigation arguments missing"))?;
            let mut id = 0;
            // SAFETY: args is valid during its SDK callback; id is writable.
            unsafe {
                args.NavigationId(&mut id)
                    .map_err(|error| callback_context("Startup NavigationId", error))?;
            }
            let mut raw = PWSTR::null();
            // SAFETY: the getter transfers a terminated task-memory URI.
            unsafe {
                args.Uri(&mut raw)
                    .map_err(|error| callback_context("Startup NavigationStarting.Uri", error))?;
            }
            // SAFETY: converter owns and frees that transferred allocation.
            let uri = unsafe { read_task_mem_uri(raw) }?;
            state.navigation_started(id, &uri);
            state.record(&json!({"kind":"navigation_starting","navigation_id":id.to_string()}));
            Ok(())
        })
    }));
    let mut token = 0;
    // SAFETY: the SDK retains the handler and returns its owned removal token.
    unsafe {
        tokens
            .webview
            .add_NavigationStarting(&handler, &mut token)
            .map_err(|error| callback_context("Startup add_NavigationStarting", error))?;
    }
    tokens.starting = Some((token, handler));
    let weak = Rc::downgrade(shared);
    // The SDK's native callback boundary requires runtime-erased closure ownership.
    let handler = NavigationCompletedEventHandler::create(Box::new(move |_, args| {
        observe(&weak, "NavigationCompleted", |state| {
            let args =
                args.ok_or_else(|| callback_error("Startup completion arguments missing"))?;
            let mut success = BOOL(0);
            let mut id = 0;
            // SAFETY: args is valid during this callback; outputs are writable.
            unsafe {
                args.IsSuccess(&mut success).map_err(|error| {
                    callback_context("Startup NavigationCompleted.IsSuccess", error)
                })?;
                args.NavigationId(&mut id)
                    .map_err(|error| callback_context("Startup NavigationId", error))?;
            }
            let current = state.navigation_completed(id, success.as_bool());
            state.record(&json!({"kind":"navigation_completed","success":success.as_bool(),"navigation_id":id.to_string(),"current":current}));
            Ok(())
        })
    }));
    // SAFETY: handler/token have the same lifetime and ownership as above.
    unsafe {
        tokens
            .webview
            .add_NavigationCompleted(&handler, &mut token)
            .map_err(|error| callback_context("Startup add_NavigationCompleted", error))?;
    }
    tokens.completed = Some((token, handler));
    Ok(())
}

fn content(tokens: &mut Registrations, shared: &Shared) -> windows::core::Result<()> {
    let weak = Rc::downgrade(shared);
    // The SDK's native callback boundary requires runtime-erased closure ownership.
    let handler = ContentLoadingEventHandler::create(Box::new(move |_, args| {
        observe(&weak, "ContentLoading", |state| {
            let args = args.ok_or_else(|| callback_error("Startup content arguments missing"))?;
            let mut error_page = BOOL(0);
            let mut id = 0;
            // SAFETY: native args is live and both outputs are writable.
            unsafe {
                args.IsErrorPage(&mut error_page).map_err(|error| {
                    callback_context("Startup ContentLoading.IsErrorPage", error)
                })?;
                args.NavigationId(&mut id)
                    .map_err(|error| callback_context("Startup NavigationId", error))?;
            }
            state.record(&json!({"kind":"content_loading","error_page":error_page.as_bool(),"navigation_id":id.to_string()}));
            Ok(())
        })
    }));
    let mut token = 0;
    // SAFETY: the SDK retains the handler until removal by this owner.
    unsafe {
        tokens
            .webview
            .add_ContentLoading(&handler, &mut token)
            .map_err(|error| callback_context("Startup add_ContentLoading", error))?;
    }
    tokens.content = Some((token, handler));
    let weak = Rc::downgrade(shared);
    // The SDK's native callback boundary requires runtime-erased closure ownership.
    let handler = DOMContentLoadedEventHandler::create(Box::new(move |_, args| {
        observe(&weak, "DOMContentLoaded", |state| {
            let args = args.ok_or_else(|| callback_error("Startup DOM arguments missing"))?;
            let mut id = 0;
            // SAFETY: args is live for this callback and id is writable.
            unsafe {
                args.NavigationId(&mut id)
                    .map_err(|error| callback_context("Startup NavigationId", error))?;
            }
            let current = state.dom_loaded(id);
            state.record(&json!({"kind":"dom_content_loaded","navigation_id":id.to_string(),"current":current}));
            Ok(())
        })
    }));
    // SAFETY: the SDK retains the handler and token until owned removal.
    unsafe {
        tokens
            .resources
            .add_DOMContentLoaded(&handler, &mut token)
            .map_err(|error| callback_context("Startup add_DOMContentLoaded", error))?;
    }
    tokens.dom = Some((token, handler));
    Ok(())
}

fn resources(tokens: &mut Registrations, shared: &Shared) -> windows::core::Result<()> {
    let weak = Rc::downgrade(shared);
    // The SDK's native callback boundary requires runtime-erased closure ownership.
    let handler = WebResourceResponseReceivedEventHandler::create(Box::new(move |_, args| {
        observe(&weak, "WebResourceResponseReceived", |state| {
            let args = args.ok_or_else(|| callback_error("Startup response arguments missing"))?;
            // SAFETY: getters read actual request/response metadata; no body is read.
            let (request, response) = unsafe {
                (
                    args.Request().map_err(|error| {
                        callback_context("Startup WebResourceResponseReceived.Request", error)
                    })?,
                    args.Response().map_err(|error| {
                        callback_context("Startup WebResourceResponseReceived.Response", error)
                    })?,
                )
            };
            let mut raw = PWSTR::null();
            // SAFETY: the getter transfers one terminated CoTaskMem allocation.
            unsafe {
                request
                    .Uri(&mut raw)
                    .map_err(|error| callback_context("Startup WebResourceRequest.Uri", error))?;
            }
            // SAFETY: read_task_mem_uri owns and frees that transferred allocation.
            let uri = unsafe { read_task_mem_uri(raw) }?;
            let mut status = 0;
            // SAFETY: status is writable and the response remains live.
            unsafe {
                response.StatusCode(&mut status).map_err(|error| {
                    callback_context("Startup WebResourceResponseView.StatusCode", error)
                })?;
            }
            // SAFETY: response remains live; only the Content-Type header is read.
            let headers = unsafe {
                response.Headers().map_err(|error| {
                    callback_context("Startup WebResourceResponseView.Headers", error)
                })?
            };
            let name = HSTRING::from("Content-Type");
            let mut present = BOOL(0);
            // SAFETY: native outputs are writable and name outlives both calls.
            unsafe {
                headers.Contains(&name, &mut present).map_err(|error| {
                    callback_context("Startup HttpResponseHeaders.Contains(Content-Type)", error)
                })?;
            }
            let mime = if present.as_bool() {
                let mut raw = PWSTR::null();
                // SAFETY: this getter transfers a terminated CoTaskMem allocation.
                unsafe {
                    headers.GetHeader(&name, &mut raw).map_err(|error| {
                        callback_context(
                            "Startup HttpResponseHeaders.GetHeader(Content-Type)",
                            error,
                        )
                    })?;
                }
                // SAFETY: this conversion owns the transferred header allocation.
                Some(unsafe { read_task_mem_message(raw) }?)
            } else {
                None
            };
            state.record(
                &json!({"kind":"resource_response","uri":uri,"status":status,"content_type":mime}),
            );
            Ok(())
        })
    }));
    let mut token = 0;
    // SAFETY: this passive observer neither intercepts nor modifies requests.
    unsafe {
        tokens
            .resources
            .add_WebResourceResponseReceived(&handler, &mut token)
            .map_err(|error| callback_context("Startup add_WebResourceResponseReceived", error))?;
    }
    tokens.response = Some((token, handler));
    Ok(())
}

fn process(tokens: &mut Registrations, shared: &Shared) -> windows::core::Result<()> {
    let weak = Rc::downgrade(shared);
    // The SDK's native callback boundary requires runtime-erased closure ownership.
    let handler = ProcessFailedEventHandler::create(Box::new(move |_, args| {
        observe(&weak, "ProcessFailed", |state| {
            let args = args.ok_or_else(|| callback_error("Startup process arguments missing"))?;
            let mut kind = COREWEBVIEW2_PROCESS_FAILED_KIND(0);
            // SAFETY: args is live and kind is writable for this call.
            unsafe {
                args.ProcessFailedKind(&mut kind)
                    .map_err(|error| callback_context("Startup ProcessFailedKind", error))?;
            }
            state.record(&json!({"kind":"process_failed","process_kind":kind.0}));
            Ok(())
        })
    }));
    let mut token = 0;
    // SAFETY: SDK retains the callback until its owned token is removed.
    unsafe {
        tokens
            .webview
            .add_ProcessFailed(&handler, &mut token)
            .map_err(|error| callback_context("Startup add_ProcessFailed", error))?;
    }
    tokens.process = Some((token, handler));
    Ok(())
}
