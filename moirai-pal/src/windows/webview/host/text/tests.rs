//! Actual bounded native pointer conversion and computed representation oracles.

use windows::core::PCWSTR;

use super::super::com::ComApartment;
use super::{MAX_WEBVIEW_MESSAGE_UNITS, MessageText, read_bounded_pcwstr};
use crate::webview_limits::MAX_WEBVIEW_MESSAGE_BYTES;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

#[test]
fn bounded_native_pointer_preserves_exact_ascii_and_supplementary_scalars() {
    let _trace = trace();
    let _test = tracing::info_span!("native_message_pointer_boundary").entered();
    let _apartment = tracing::info_span!("arrange")
        .in_scope(|| ComApartment::initialize().expect("actual native COM apartment"));
    let inputs = tracing::info_span!("arrange").in_scope(|| {
        [
            "ab".repeat(MAX_WEBVIEW_MESSAGE_BYTES / 2),
            "🦀".repeat(MAX_WEBVIEW_MESSAGE_BYTES / 4),
        ]
    });
    for expected in inputs {
        let units = expected.encode_utf16().chain([0]).collect::<Vec<_>>();
        let pointer = PCWSTR(units.as_ptr());
        // SAFETY: the actual vector contains the complete terminated UTF16
        // allocation and stays borrowed through the production conversion.
        let actual = tracing::info_span!("act")
            .in_scope(|| unsafe { read_bounded_pcwstr::<MessageText>(&pointer) });
        tracing::info_span!("assert").in_scope(|| {
            let actual = actual.expect("inclusive native message bound");
            assert_eq!(actual.len(), MAX_WEBVIEW_MESSAGE_BYTES);
            assert_eq!(actual, expected);
        });
    }
}

#[test]
fn bounded_native_scan_rejects_unit_and_byte_overflow_before_copying() {
    let _trace = trace();
    let _test = tracing::info_span!("native_message_pointer_rejection").entered();
    let _apartment = tracing::info_span!("arrange")
        .in_scope(|| ComApartment::initialize().expect("actual native COM apartment"));
    let inputs = tracing::info_span!("arrange").in_scope(|| {
        [
            (
                format!("{}c", "ab".repeat(MAX_WEBVIEW_MESSAGE_UNITS / 2)),
                "WebView2 UTF-16 payload exceeds its bound",
            ),
            (
                "🦀".repeat(MAX_WEBVIEW_MESSAGE_BYTES / 4 + 1),
                "WebView2 JSON message exceeds the bounded byte limit",
            ),
        ]
    });
    for (text, expected) in inputs {
        let units = text.encode_utf16().chain([0]).collect::<Vec<_>>();
        let pointer = PCWSTR(units.as_ptr());
        // SAFETY: the borrowed vector contains every scanned unit and its
        // terminator; the provider stops after at most bound+1 units.
        let actual = tracing::info_span!("act")
            .in_scope(|| unsafe { read_bounded_pcwstr::<MessageText>(&pointer) });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(
                actual.expect_err("oversized representation").message(),
                expected
            );
        });
    }
}

#[test]
fn native_pointer_and_surrogate_errors_preserve_the_actual_failure() {
    let _trace = trace();
    let _test = tracing::info_span!("native_message_pointer_validity").entered();
    let _apartment = tracing::info_span!("arrange")
        .in_scope(|| ComApartment::initialize().expect("actual native COM apartment"));
    let units = tracing::info_span!("arrange").in_scope(|| [0xd800, 0]);
    for (pointer, expected) in [
        (PCWSTR::null(), "WebView2 returned a null UTF-16 pointer"),
        (
            PCWSTR(units.as_ptr()),
            "WebView2 payload contains unmatched UTF-16",
        ),
    ] {
        // SAFETY: null is rejected before a read; the second pointer borrows
        // an actual terminated allocation throughout the conversion.
        let actual = tracing::info_span!("act")
            .in_scope(|| unsafe { read_bounded_pcwstr::<MessageText>(&pointer) });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(
                actual.expect_err("invalid native representation").message(),
                expected
            );
        });
    }
}
