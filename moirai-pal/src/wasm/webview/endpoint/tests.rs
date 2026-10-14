//! Real browser values exercise structural checks, not a fabricated WebView.

use std::{cell::RefCell, io::ErrorKind, rc::Rc};

use js_sys::Reflect;
use wasm_bindgen::{JsValue, closure::Closure};
use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::{Event, EventTarget};

use super::{EventOperation, NativeMessageTarget, callable, property};
use crate::wasm::webview::WebViewBridge;

#[wasm_bindgen_test]
fn ordinary_browser_reports_the_actual_missing_native_member() {
    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("webview_actual_browser_absence").entered();
    let member = tracing::info_span!("arrange").in_scope(|| {
        let window = web_sys::window().expect("actual browser Window");
        let chrome = Reflect::get(window.as_ref(), &JsValue::from_str("chrome"))
            .expect("actual native property lookup");
        if chrome.is_null_or_undefined() {
            "chrome"
        } else {
            let webview = Reflect::get(&chrome, &JsValue::from_str("webview"))
                .expect("actual Chrome property lookup");
            assert!(webview.is_null_or_undefined(), "ordinary browser fixture");
            "webview"
        }
    });
    let actual = tracing::info_span!("act").in_scope(WebViewBridge::current);
    tracing::info_span!("assert").in_scope(|| {
        let error = match actual {
            Err(error) => error,
            Ok(_) => panic!("ordinary browser unexpectedly supplies a native bridge"),
        };
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.to_string(),
            format!("WebView2 bridge has no {member}")
        );
    });
}

#[wasm_bindgen_test]
fn real_window_preserves_receiver_and_three_native_callable_identities() {
    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("webview_real_window_structural_methods").entered();
    let receiver: JsValue = tracing::info_span!("arrange")
        .in_scope(|| web_sys::window().expect("actual Window").into());
    let (target, post) = tracing::info_span!("act").in_scope(|| {
        (
            NativeMessageTarget::new(receiver.clone()).expect("real native event methods"),
            callable(&receiver, "postMessage").expect("real Window postMessage method"),
        )
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(target.receiver, receiver);
        let members: [(&str, &JsValue); 3] = [
            ("addEventListener", target.add.as_ref()),
            ("removeEventListener", target.remove.as_ref()),
            ("postMessage", post.as_ref()),
        ];
        for (member, cached) in members {
            let actual = Reflect::get(&receiver, &JsValue::from_str(member))
                .expect("actual unchanged native method");
            assert!(actual.is_function(), "native {member} is callable");
            assert_eq!(cached, &actual, "cached {member} has the same identity");
        }
    });
}

#[wasm_bindgen_test]
fn actual_nonobject_values_reject_before_target_admission() {
    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("webview_nonobject_receiver_domain").entered();
    let cases = tracing::info_span!("arrange").in_scope(|| {
        let window = web_sys::window().expect("actual Window");
        let function = callable(window.as_ref(), "postMessage").expect("native callable");
        [
            JsValue::NULL,
            JsValue::UNDEFINED,
            JsValue::TRUE,
            JsValue::from(37.5),
            JsValue::from_str("native receiver λ🦀"),
            function.into(),
        ]
    });
    for value in cases {
        let actual = tracing::info_span!("act").in_scope(|| NativeMessageTarget::new(value));
        tracing::info_span!("assert").in_scope(|| {
            let error = match actual {
                Err(error) => error,
                Ok(_) => panic!("nonobject receiver unexpectedly passes admission"),
            };
            assert_eq!(error.kind(), ErrorKind::InvalidData);
            assert_eq!(error.to_string(), "WebView2 bridge is not an object");
        });
    }
}

#[wasm_bindgen_test]
fn actual_dom_event_target_does_not_supply_native_post_message() {
    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("webview_missing_required_post_method").entered();
    let receiver: JsValue = tracing::info_span!("arrange")
        .in_scope(|| EventTarget::new().expect("actual DOM EventTarget").into());
    let (target, actual) = tracing::info_span!("act").in_scope(|| {
        (
            NativeMessageTarget::new(receiver.clone()).expect("real DOM event operations"),
            callable(&receiver, "postMessage"),
        )
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(target.receiver, receiver);
        let error = actual.expect_err("DOM event methods do not imply native posting");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(error.to_string(), "WebView2 bridge has no postMessage");
    });
}

#[wasm_bindgen_test]
fn native_noncallable_event_properties_preserve_member_context() {
    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("webview_native_noncallable_properties").entered();
    let event = tracing::info_span!("arrange")
        .in_scope(|| Event::new("native-shape-λ🦀").expect("actual browser Event"));
    // DOM's constructed Event has its supplied type, bubbles=false, and the
    // published NONE phase before dispatch. These are actual browser values.
    let cases = [
        ("type", JsValue::from_str("native-shape-λ🦀")),
        ("bubbles", JsValue::FALSE),
        ("eventPhase", JsValue::from(0)),
    ];
    for (member, expected) in cases {
        let (value, actual) = tracing::info_span!("act").in_scope(|| {
            (
                Reflect::get(event.as_ref(), &JsValue::from_str(member))
                    .expect("actual native event property"),
                callable(event.as_ref(), member),
            )
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(value, expected, "actual {member} value");
            let error = actual.expect_err("native data properties are not callables");
            assert_eq!(error.kind(), ErrorKind::InvalidData);
            assert_eq!(
                error.to_string(),
                format!("WebView2 {member} is not callable")
            );
        });
    }
}

#[wasm_bindgen_test]
fn actual_primitive_lookup_exceptions_retain_native_type_and_operation() {
    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("webview_native_lookup_exceptions").entered();
    let cases = tracing::info_span!("arrange").in_scope(|| {
        [
            JsValue::from_str("lookup 李🦀"),
            JsValue::from(41),
            JsValue::FALSE,
        ]
    });
    for value in cases {
        for member in ["postMessage", "addEventListener", "removeEventListener"] {
            let actual = tracing::info_span!("act").in_scope(|| property(&value, member));
            tracing::info_span!("assert").in_scope(|| {
                let error = actual.expect_err("Reflect.get rejects a primitive target");
                assert_eq!(error.kind(), ErrorKind::Other);
                assert!(
                    error
                        .to_string()
                        .starts_with(&format!("WebView2 {member} lookup: TypeError:")),
                    "{error}"
                );
                assert!(error.to_string().contains("Reflect.get"), "{error}");
            });
        }
    }
}

#[wasm_bindgen_test]
fn cached_native_event_methods_attach_and_detach_the_same_real_callback() {
    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("webview_native_cached_event_invocation").entered();
    let (events, native, target, callback) = tracing::info_span!("arrange").in_scope(|| {
        let events = Rc::new(RefCell::new(Vec::new()));
        let observed = Rc::clone(&events);
        let native = EventTarget::new().expect("actual browser EventTarget");
        let target = NativeMessageTarget::new(native.clone().into()).expect("native methods");
        // Runtime type erasure is the browser callback ABI (interop condition 2).
        let callback = Closure::<dyn FnMut(Event)>::new(move |event: Event| {
            observed
                .borrow_mut()
                .push((event.type_(), event.event_phase()));
        });
        (events, native, target, callback)
    });
    let (first, second) = tracing::info_span!("act").in_scope(|| {
        target
            .event(EventOperation::Attach, callback.as_ref())
            .expect("actual attach");
        let first = native
            .dispatch_event(&Event::new("message").expect("first real event"))
            .expect("native first dispatch");
        target
            .event(EventOperation::Detach, callback.as_ref())
            .expect("actual detach");
        let second = native
            .dispatch_event(&Event::new("message").expect("second real event"))
            .expect("native second dispatch");
        (first, second)
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!((first, second), (true, true), "neither event is cancelled");
        assert_eq!(*events.borrow(), [("message".to_owned(), Event::AT_TARGET)]);
    });
}
