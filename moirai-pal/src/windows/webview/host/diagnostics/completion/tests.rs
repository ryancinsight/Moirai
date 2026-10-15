//! The actual COM callback boundary admits borrowed bytes before retaining JSON.

use super::super::state::{Shared, State};
use super::*;
use crate::{
    webview_limits::MAX_WEBVIEW_MESSAGE_BYTES,
    webview_startup::{Selection, Start},
};
use serde_json::json;
use std::{cell::RefCell, rc::Rc};

mod admission;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .without_time()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

fn request() -> (
    Shared,
    RequestId,
    ICoreWebView2CallDevToolsProtocolMethodCompletedHandler,
) {
    let state = Rc::new(RefCell::new(State::new(Selection {
        components: crate::webview_startup::Components::Full,
        start_uri: "https://startup.invalid/index.html".to_owned(),
        selectors: vec!["#status".to_owned()],
    })));
    let Start::Begun(id) = state.borrow_mut().slot.start().expect("request identity") else {
        panic!("new slot admits one request")
    };
    let handler = Completion {
        completed: Cell::new(false),
        state: Rc::downgrade(&state),
        destination: Destination::Dom {
            id,
            stage: Stage::Document,
        },
    }
    .into();
    (state, id, handler)
}

#[test]
fn native_completion_admits_inclusive_bytes_and_preserves_unicode_values() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_native_reply_bound").entered();
    let _apartment = tracing::info_span!("arrange").in_scope(|| {
        super::super::super::com::ComApartment::initialize().expect("native COM apartment")
    });
    for bytes in [
        MAX_WEBVIEW_MESSAGE_BYTES - 1,
        MAX_WEBVIEW_MESSAGE_BYTES,
        MAX_WEBVIEW_MESSAGE_BYTES + 1,
    ] {
        let (state, id, callback, units, expected) = tracing::info_span!("arrange", bytes)
            .in_scope(|| {
                let value = json!({"value":format!("患者🦀{}", "x".repeat(bytes - 22))});
                let encoded = value.to_string();
                assert_eq!(encoded.len(), bytes);
                let units = encoded.encode_utf16().chain([0]).collect::<Vec<_>>();
                let (state, id, callback) = request();
                (state, id, callback, units, value)
            });
        tracing::info_span!("act").in_scope(|| {
            // SAFETY: the vector lends its complete terminated UTF16 payload
            // through the real SDK callback interface invocation.
            unsafe { callback.Invoke(HRESULT(0), PCWSTR(units.as_ptr())) }
                .expect("callback delivery");
        });
        tracing::info_span!("assert").in_scope(|| {
            let mut state = state.borrow_mut();
            let (actual_id, _, reply) = state.reply.take().expect("owned reply");
            assert_eq!(actual_id, id);
            if bytes <= MAX_WEBVIEW_MESSAGE_BYTES {
                assert_eq!(reply.expect("inclusive JSON"), expected);
            } else {
                assert_eq!(
                    reply.expect_err("excess byte rejected").error.message(),
                    "WebView2 JSON message exceeds the bounded byte limit"
                );
            }
            assert!(
                state.slot.is_current(id),
                "the full chain still owns its slot"
            );
        });
    }
}

#[test]
fn native_completion_preserves_failures_and_refuses_a_second_delivery() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_native_reply_failures").entered();
    let _apartment = tracing::info_span!("arrange").in_scope(|| {
        super::super::super::com::ComApartment::initialize().expect("native COM apartment")
    });
    for (input, expected) in [
        ("[]", "WebView2 CDP response is not an object"),
        (
            r#"{"error":{"code":-32000,"message":"node disappeared"}}"#,
            "WebView2 CDP error response",
        ),
    ] {
        let (state, _, callback, units) = tracing::info_span!("arrange").in_scope(|| {
            let (state, id, callback) = request();
            (
                state,
                id,
                callback,
                input.encode_utf16().chain([0]).collect::<Vec<_>>(),
            )
        });
        let repeated = tracing::info_span!("act").in_scope(|| {
            // SAFETY: first call borrows a terminated vector; the second is
            // rejected by the once-only owner before reading its null pointer.
            unsafe { callback.Invoke(HRESULT(0), PCWSTR(units.as_ptr())) }.expect("first delivery");
            unsafe { callback.Invoke(HRESULT(0), PCWSTR::null()) }.expect_err("second delivery")
        });
        tracing::info_span!("assert").in_scope(|| {
            let (_, _, reply) = state.borrow_mut().reply.take().expect("first owned reply");
            let failure = reply.expect_err("actual protocol rejection");
            assert_eq!(failure.error.message(), expected);
            assert_eq!(failure.value("test")["response"], input);
            assert_eq!(
                repeated.message(),
                "WebView2 startup completion invoked twice"
            );
        });
    }
    let (state, _, callback) = tracing::info_span!("arrange").in_scope(request);
    tracing::info_span!("act").in_scope(|| {
        // SAFETY: the actual failure HRESULT short-circuits before the SDK
        // result pointer is read; no successful payload is manufactured.
        unsafe { callback.Invoke(windows::Win32::Foundation::E_ACCESSDENIED, PCWSTR::null()) }
            .expect("native failure delivery");
    });
    tracing::info_span!("assert").in_scope(|| {
        let (_, _, reply) = state
            .borrow_mut()
            .reply
            .take()
            .expect("owned native failure");
        assert_eq!(
            reply.expect_err("native failure preserved").error.code(),
            windows::Win32::Foundation::E_ACCESSDENIED
        );
    });
    let (state, _, callback, encoded, units) = tracing::info_span!("arrange").in_scope(|| {
        let encoded = json!({"error":{"code":5,"message":"λ denied 患者"}}).to_string();
        let units = encoded.encode_utf16().chain([0]).collect::<Vec<_>>();
        let (state, id, callback) = request();
        (state, id, callback, encoded, units)
    });
    tracing::info_span!("act").in_scope(|| {
        // SAFETY: the actual failure status and complete bounded error reply
        // both cross the real SDK callback interface; neither replaces the other.
        unsafe {
            callback.Invoke(
                windows::Win32::Foundation::E_ACCESSDENIED,
                PCWSTR(units.as_ptr()),
            )
        }
        .expect("native status and error payload");
    });
    tracing::info_span!("assert").in_scope(|| {
        let (_, _, reply) = state
            .borrow_mut()
            .reply
            .take()
            .expect("owned native failure");
        let failure = reply.expect_err("failed native status cannot continue the chain");
        assert_eq!(
            failure.error.code(),
            windows::Win32::Foundation::E_ACCESSDENIED
        );
        assert_eq!(failure.value("actual error")["response"], encoded);
        assert_eq!(failure.value("actual error")["decoding_error"], Value::Null);
    });
}

#[test]
fn cancelled_and_ownerless_completions_do_not_read_or_retain_late_payloads() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_native_reply_cancel").entered();
    let _apartment = tracing::info_span!("arrange").in_scope(|| {
        super::super::super::com::ComApartment::initialize().expect("native COM apartment")
    });
    let (state, _, callback) = tracing::info_span!("arrange").in_scope(request);
    tracing::info_span!("act").in_scope(|| {
        state.borrow_mut().slot.close();
        // SAFETY: closed requests return before inspecting the absent payload.
        unsafe { callback.Invoke(HRESULT(0), PCWSTR::null()) }.expect("cancelled completion");
    });
    tracing::info_span!("assert").in_scope(|| {
        let state = state.borrow();
        assert_eq!(state.stale, 1);
        assert_eq!(
            state.slot.snapshot(),
            crate::webview_startup::RequestSnapshot {
                generation: 1,
                pending: false,
                closed: true
            }
        );
        assert!(state.reply.is_none());
    });
    let (enable, receiver, before) = tracing::info_span!("arrange").in_scope(|| {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let before = state.borrow().budget.snapshot();
        let enable: ICoreWebView2CallDevToolsProtocolMethodCompletedHandler = Completion {
            completed: Cell::new(false),
            state: Rc::downgrade(&state),
            destination: Destination::Enable(sender),
        }
        .into();
        (enable, receiver, before)
    });
    tracing::info_span!("act").in_scope(|| {
        // SAFETY: closed activation returns before the null SDK payload is
        // read and never appends a failure or sends another activation result.
        unsafe { enable.Invoke(HRESULT(0), PCWSTR::null()) }.expect("closed activation completion");
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            receiver
                .try_recv()
                .expect_err("closed activation sends no result"),
            std::sync::mpsc::TryRecvError::Empty
        );
        assert_eq!(state.borrow().budget.snapshot(), before);
        assert_eq!(state.borrow().stale, 1);
        assert!(state.borrow().slot.snapshot().closed);
    });
    let (owner, _, callback) = tracing::info_span!("arrange").in_scope(request);
    tracing::info_span!("act").in_scope(|| {
        drop(owner);
        // SAFETY: an expired Weak owner returns before inspecting this pointer.
        unsafe { callback.Invoke(HRESULT(0), PCWSTR::null()) }.expect("expired owner completion");
    });
    tracing::info_span!("assert").in_scope(|| assert_eq!(Rc::strong_count(&state), 1));
}
