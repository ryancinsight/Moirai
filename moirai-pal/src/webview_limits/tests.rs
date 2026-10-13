use std::io::ErrorKind;

use super::{
    MAX_WEBVIEW_EVENTS, MAX_WEBVIEW_MESSAGE_BYTES, MAX_WEBVIEW_MESSAGE_UNITS,
    MAX_WEBVIEW_WAIT_MILLISECONDS, message_utf8_length, validate_message,
};

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

#[test]
fn native_resource_limits_preserve_the_established_values() {
    let _trace = trace();
    let _test = tracing::info_span!("webview_resource_contract").entered();
    let expected = tracing::info_span!("arrange")
        .in_scope(|| (1_usize << 16, 1_usize << 16, 1_usize << 8, 30_u32 * 1_000));
    let limits = tracing::info_span!("act").in_scope(|| {
        (
            MAX_WEBVIEW_MESSAGE_BYTES,
            MAX_WEBVIEW_MESSAGE_UNITS,
            MAX_WEBVIEW_EVENTS,
            MAX_WEBVIEW_WAIT_MILLISECONDS,
        )
    });
    tracing::info_span!("assert").in_scope(|| assert_eq!(limits, expected));
}

#[test]
fn byte_admission_is_inclusive_and_preserves_invalid_input_context() {
    let _trace = trace();
    let _test = tracing::info_span!("webview_message_byte_budget").entered();
    let (valid, excessive, control) = tracing::info_span!("arrange").in_scope(|| {
        (
            vec![b'a'; MAX_WEBVIEW_MESSAGE_BYTES],
            vec![b'b'; MAX_WEBVIEW_MESSAGE_BYTES + 1],
            b"{\"text\":\"a\0b\"}".as_slice(),
        )
    });
    let actual = tracing::info_span!("act").in_scope(|| {
        [
            validate_message(&valid),
            validate_message(&excessive),
            validate_message(control),
        ]
    });
    tracing::info_span!("assert").in_scope(|| {
        let [valid, excessive, control] = actual;
        assert_eq!(valid.expect("inclusive byte budget"), ());
        for (result, expected) in [
            (
                excessive,
                "WebView2 JSON message exceeds the bounded byte limit",
            ),
            (control, "WebView2 JSON message contains NUL"),
        ] {
            let error = result.expect_err("rejected payload");
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
            assert_eq!(error.to_string(), expected);
        }
    });
}

#[test]
fn unicode_length_matches_the_actual_utf8_byte_oracle_without_copying() {
    let _trace = trace();
    let _test = tracing::info_span!("webview_unicode_representation").entered();
    for text in [
        "",
        "{\"patient\":\"Zoë / 李 / 🦀\"}",
        "\"\\u0000\"",
        "a",
        "é",
        "李",
        "🦀",
    ] {
        let units =
            tracing::info_span!("arrange").in_scope(|| text.encode_utf16().collect::<Vec<_>>());
        let actual =
            tracing::info_span!("act").in_scope(|| message_utf8_length(units.iter().copied()));
        tracing::info_span!("assert")
            .in_scope(|| assert_eq!(actual.expect("well-formed UTF16"), text.len()));
    }
}

#[test]
fn utf8_and_utf16_bounds_use_their_actual_representation_sizes() {
    let _trace = trace();
    let _test = tracing::info_span!("webview_representation_bounds").entered();
    let inputs = tracing::info_span!("arrange").in_scope(|| {
        [
            "a".repeat(MAX_WEBVIEW_MESSAGE_BYTES),
            "é".repeat(MAX_WEBVIEW_MESSAGE_BYTES / 2),
            "🦀".repeat(MAX_WEBVIEW_MESSAGE_BYTES / 4),
        ]
    });
    for text in inputs {
        let actual = tracing::info_span!("act")
            .in_scope(|| message_utf8_length(text.encode_utf16().collect::<Vec<_>>().into_iter()));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(
                actual.expect("exact byte budget"),
                MAX_WEBVIEW_MESSAGE_BYTES
            )
        });
    }
    let inputs = tracing::info_span!("arrange").in_scope(|| {
        [
            (
                "a".repeat(MAX_WEBVIEW_MESSAGE_UNITS + 1),
                "WebView2 UTF-16 payload exceeds its bound",
            ),
            (
                "é".repeat(MAX_WEBVIEW_MESSAGE_BYTES / 2 + 1),
                "WebView2 JSON message exceeds the bounded byte limit",
            ),
            (
                "🦀".repeat(MAX_WEBVIEW_MESSAGE_BYTES / 4 + 1),
                "WebView2 JSON message exceeds the bounded byte limit",
            ),
        ]
    });
    for (text, expected) in inputs {
        let actual = tracing::info_span!("act")
            .in_scope(|| message_utf8_length(text.encode_utf16().collect::<Vec<_>>().into_iter()));
        tracing::info_span!("assert").in_scope(|| {
            let error = actual.expect_err("representation exceeds budget");
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
            assert_eq!(error.to_string(), expected);
        });
    }
}

#[test]
fn malformed_utf16_and_embedded_control_do_not_change_the_message() {
    let _trace = trace();
    let _test = tracing::info_span!("webview_lossless_unicode_contract").entered();
    let inputs = tracing::info_span!("arrange").in_scope(|| {
        [
            (
                vec![0xd800],
                ErrorKind::InvalidData,
                "WebView2 payload contains unmatched UTF-16",
            ),
            (
                vec![0xdc00],
                ErrorKind::InvalidData,
                "WebView2 payload contains unmatched UTF-16",
            ),
            (
                vec![0xd800, 0x0061],
                ErrorKind::InvalidData,
                "WebView2 payload contains unmatched UTF-16",
            ),
            (
                vec![0x0061, 0x0000, 0x0062],
                ErrorKind::InvalidInput,
                "WebView2 JSON message contains NUL",
            ),
        ]
    });
    for (units, kind, expected) in inputs {
        let actual = tracing::info_span!("act").in_scope(|| message_utf8_length(units.into_iter()));
        tracing::info_span!("assert").in_scope(|| {
            let error = actual.expect_err("no lossy conversion");
            assert_eq!(error.kind(), kind);
            assert_eq!(error.to_string(), expected);
        });
    }
}
