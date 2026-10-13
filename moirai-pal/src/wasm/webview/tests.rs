//! Browser-native JSON and lossless representation acceptance.

use std::io::ErrorKind;

use js_sys::JSON;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

use super::{WebViewBridge, message};

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn actual_browser_object_and_json_string_preserve_values() {
    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("webview_browser_json_values").entered();
    let text = r#"{"patient":"Zoë / 李 / 🦀","sequence":"18446744073709551615"}"#;
    let values = tracing::info_span!("arrange").in_scope(|| {
        [
            JSON::parse(text).expect("actual JSON object"),
            JsValue::from_str(text),
        ]
    });
    for value in values {
        let actual = tracing::info_span!("act").in_scope(|| message::receive(value));
        tracing::info_span!("assert").in_scope(|| assert_eq!(actual.expect("valid JSON"), text));
    }
}

#[wasm_bindgen_test]
fn actual_browser_json_errors_retain_syntax_context() {
    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("webview_browser_json_errors").entered();
    let values = tracing::info_span!("arrange").in_scope(|| {
        [
            JsValue::NULL,
            JsValue::TRUE,
            JsValue::from_str("{\"incomplete\":"),
        ]
    });
    for (index, value) in values.into_iter().enumerate() {
        let actual = tracing::info_span!("act").in_scope(|| message::receive(value));
        tracing::info_span!("assert").in_scope(|| {
            let error = actual.expect_err("invalid browser value");
            assert_eq!(error.kind(), ErrorKind::InvalidData);
            if index < 2 {
                assert_eq!(
                    error.to_string(),
                    "WebView2 message must contain a JSON object or JSON string"
                );
            } else {
                assert!(
                    error
                        .to_string()
                        .contains("WebView2 received JSON.parse: SyntaxError:"),
                    "{error}"
                );
            }
        });
    }
}

#[wasm_bindgen_test]
fn actual_browser_serialization_obeys_the_inclusive_byte_budget() {
    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("webview_browser_byte_boundaries").entered();
    for character in ["a", "é", "🦀"] {
        // A JSON string has two quote bytes. Its repeated scalar payload has
        // floor((budget - 2)/width) scalars; the next scalar exceeds the budget.
        let inputs = tracing::info_span!("arrange").in_scope(|| {
            let count = (WebViewBridge::message_byte_limit() - 2) / character.len();
            [
                format!("\"{}\"", character.repeat(count)),
                format!("\"{}\"", character.repeat(count + 1)),
            ]
        });
        let [valid, excessive] = inputs;
        let actual = tracing::info_span!("act").in_scope(|| {
            [
                message::receive(JsValue::from_str(&valid)),
                message::receive(JsValue::from_str(&excessive)),
            ]
        });
        tracing::info_span!("assert").in_scope(|| {
            let [valid_result, excessive_result] = actual;
            assert_eq!(valid_result.expect("inclusive scalar budget"), valid);
            let error = excessive_result.expect_err("next scalar exceeds budget");
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
            let expected = if excessive.encode_utf16().count()
                > crate::webview_limits::MAX_WEBVIEW_MESSAGE_UNITS
            {
                "WebView2 UTF-16 payload exceeds its bound"
            } else {
                "WebView2 JSON message exceeds the bounded byte limit"
            };
            assert_eq!(error.to_string(), expected);
        });
    }
}

#[wasm_bindgen_test]
fn actual_browser_object_serialization_checks_the_allocated_representation() {
    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("webview_browser_object_bounds").entered();
    let texts = tracing::info_span!("arrange").in_scope(|| {
        let overhead = r#"{"text":""}"#.len();
        let count = WebViewBridge::message_byte_limit() - overhead;
        [
            format!(r#"{{"text":"{}"}}"#, "a".repeat(count)),
            format!(r#"{{"text":"{}"}}"#, "b".repeat(count + 1)),
        ]
    });
    let actual = tracing::info_span!("act").in_scope(|| {
        texts
            .iter()
            .map(|text| message::receive(JSON::parse(text).expect("actual browser JSON object")))
            .collect::<Vec<_>>()
    });
    tracing::info_span!("assert").in_scope(|| {
        let mut actual = actual.into_iter();
        assert_eq!(
            actual
                .next()
                .expect("exact budget result")
                .expect("inclusive object bytes"),
            texts[0]
        );
        let error = actual
            .next()
            .expect("excessive result")
            .expect_err("serialized object exceeds both bounds");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "WebView2 UTF-16 payload exceeds its bound"
        );
    });
}

#[wasm_bindgen_test]
fn real_javascript_surrogate_and_cyclic_object_fail_without_changed_values() {
    use js_sys::Reflect;

    let _trace = crate::wasm::test_trace::install();
    let _test = tracing::info_span!("webview_browser_adversarial_values").entered();
    let (surrogate, cyclic) = tracing::info_span!("arrange").in_scope(|| {
        // RFC8259 section8.2 permits escaped unpaired surrogates syntactically;
        // JSON.parse creates an actual JS UTF16 value that cannot become UTF8.
        let surrogate = JSON::parse(r#""\ud800""#).expect("native JSON surrogate value");
        let cyclic = JSON::parse(r#"{"payload":"actual object"}"#).expect("native JSON object");
        assert!(
            Reflect::set(&cyclic, &JsValue::from_str("self"), &cyclic)
                .expect("actual object cycle")
        );
        (surrogate, cyclic)
    });
    let actual = tracing::info_span!("act")
        .in_scope(|| [message::receive(surrogate), message::receive(cyclic)]);
    tracing::info_span!("assert").in_scope(|| {
        let [surrogate, cyclic] = actual;
        let error = surrogate.expect_err("lossy UTF16 conversion is rejected");
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert_eq!(
            error.to_string(),
            "WebView2 payload contains unmatched UTF-16"
        );
        let error = cyclic.expect_err("actual JSON.stringify cannot encode cycles");
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert!(
            error
                .to_string()
                .contains("WebView2 JSON.stringify: TypeError:"),
            "{error}"
        );
    });
}
