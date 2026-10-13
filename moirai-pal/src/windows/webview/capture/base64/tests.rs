use std::io::ErrorKind;

use super::super::MAX_WEBVIEW_CAPTURE_BYTES;
use super::decode;

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn tracing() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

// RFC 4648 section 4: one big-endian 24-bit group becomes four 6-bit indices.
// This encoder supplies the independent byte oracle; the published vectors
// below anchor its alphabet and terminal padding before the boundary test.
fn encode(bytes: impl Iterator<Item = u8>, length: usize) -> String {
    let mut bytes = bytes;
    let mut encoded = String::with_capacity(length.div_ceil(3) * 4);
    for _ in 0..length.div_ceil(3) {
        let first = bytes.next().expect("first byte of group");
        let second = bytes.next();
        let third = bytes.next();
        let packed = (u32::from(first) << 16)
            | (u32::from(second.unwrap_or(0)) << 8)
            | u32::from(third.unwrap_or(0));
        encoded.push(char::from(ALPHABET[((packed >> 18) & 63) as usize]));
        encoded.push(char::from(ALPHABET[((packed >> 12) & 63) as usize]));
        encoded.push(second.map_or('=', |_| char::from(ALPHABET[((packed >> 6) & 63) as usize])));
        encoded.push(third.map_or('=', |_| char::from(ALPHABET[(packed & 63) as usize])));
    }
    assert_eq!(bytes.next(), None, "encoder consumes the complete oracle");
    encoded
}

#[test]
fn published_vectors_and_complete_byte_alphabet_decode_exactly() {
    let _trace = tracing();
    let _test = tracing::info_span!("canonical_base64_values").entered();
    for (bytes, text) in [
        (b"f".as_slice(), "Zg=="),
        (b"fo".as_slice(), "Zm8="),
        (b"foo".as_slice(), "Zm9v"),
        (b"foob".as_slice(), "Zm9vYg=="),
        (b"fooba".as_slice(), "Zm9vYmE="),
        (b"foobar".as_slice(), "Zm9vYmFy"),
    ] {
        let encoded =
            tracing::info_span!("arrange").in_scope(|| encode(bytes.iter().copied(), bytes.len()));
        let decoded =
            tracing::info_span!("act").in_scope(|| decode::<6>(text).expect("RFC 4648 vector"));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(encoded, text);
            assert_eq!(decoded, bytes);
        });
    }
    let (bytes, encoded) = tracing::info_span!("arrange").in_scope(|| {
        let bytes: Vec<_> = (0..=u8::MAX).collect();
        let encoded = encode(bytes.iter().copied(), bytes.len());
        (bytes, encoded)
    });
    let decoded =
        tracing::info_span!("act").in_scope(|| decode::<256>(&encoded).expect("all byte values"));
    tracing::info_span!("assert").in_scope(|| assert_eq!(decoded, bytes));
}

#[test]
fn exact_small_budgets_account_for_both_padding_lengths() {
    let _trace = tracing();
    let _test = tracing::info_span!("exact_padding_budgets").entered();
    let inputs = tracing::info_span!("arrange")
        .in_scope(|| [("Zg==", b"f".as_slice()), ("Zm8=", b"fo".as_slice())]);
    let actual = tracing::info_span!("act")
        .in_scope(|| [decode::<1>(inputs[0].0), decode::<2>(inputs[1].0)]);
    tracing::info_span!("assert").in_scope(|| {
        for (result, (_, bytes)) in actual.into_iter().zip(inputs) {
            assert_eq!(result.expect("inclusive exact byte budget"), bytes);
        }
    });
}

#[test]
fn malformed_encoding_preserves_specific_error_context() {
    let _trace = tracing();
    let _test = tracing::info_span!("base64_rejection_context").entered();
    let cases = tracing::info_span!("arrange").in_scope(|| {
        [
            ("", "WebView2 presented-frame data is not base64"),
            ("Zg=", "WebView2 presented-frame data is not base64"),
            ("Zg===", "WebView2 presented-frame data is not base64"),
            ("=g==", "invalid base64"),
            ("Z===", "invalid base64"),
            ("Zm-v", "invalid base64"),
            ("Zm_v", "invalid base64"),
            ("Zm v", "invalid base64"),
            ("Zm\nv", "invalid base64"),
            ("Zm\0v", "invalid base64"),
            ("Zm\u{7f}v", "invalid base64"),
            ("Zm\u{e9}", "invalid base64"),
            ("Zg=A", "invalid base64 padding"),
            ("Zg==Zm9v", "invalid base64 padding"),
            ("Zm8=Zm9v", "invalid base64 padding"),
            ("Zh==", "nonzero base64 padding bits"),
            ("Zm9=", "nonzero base64 padding bits"),
        ]
    });
    for (text, context) in cases {
        let error = tracing::info_span!("act", text)
            .in_scope(|| decode::<12>(text).expect_err("malformed base64"));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(error.kind(), ErrorKind::InvalidData);
            assert_eq!(error.to_string(), context);
        });
    }
}

#[test]
fn valid_payload_exceeding_budget_reports_resource_rejection() {
    let _trace = tracing();
    let _test = tracing::info_span!("base64_budget_rejection").entered();
    let inputs = tracing::info_span!("arrange").in_scope(|| ["Zg==", "Zm8=", "Zm9v"]);
    let results = tracing::info_span!("act").in_scope(|| {
        [
            decode::<0>(inputs[0]),
            decode::<1>(inputs[1]),
            decode::<2>(inputs[2]),
        ]
    });
    tracing::info_span!("assert").in_scope(|| {
        for result in results {
            let error = result.expect_err("actual decoded length exceeds limit");
            assert_eq!(error.kind(), ErrorKind::InvalidData);
            assert_eq!(
                error.to_string(),
                "WebView2 presented-frame data exceeds the bounded image budget"
            );
        }
    });
}

#[test]
fn full_capture_bound_preserves_each_byte_and_rejects_first_excess_byte() {
    let _trace = tracing();
    let _test = tracing::info_span!("capture_byte_boundary").entered();
    // 64 MiB modulo 3 is 1: max−2 has one '=', max−1 has none,
    // max has two '=', and max+1 is the first disallowed byte count.
    for length in [
        MAX_WEBVIEW_CAPTURE_BYTES - 2,
        MAX_WEBVIEW_CAPTURE_BYTES - 1,
        MAX_WEBVIEW_CAPTURE_BYTES,
        MAX_WEBVIEW_CAPTURE_BYTES + 1,
    ] {
        let _case = tracing::info_span!("boundary", length).entered();
        let encoded = tracing::info_span!("arrange")
            .in_scope(|| encode((0..=u8::MAX).cycle().take(length), length));
        let result =
            tracing::info_span!("act").in_scope(|| decode::<MAX_WEBVIEW_CAPTURE_BYTES>(&encoded));
        tracing::info_span!("assert").in_scope(|| {
            if length > MAX_WEBVIEW_CAPTURE_BYTES {
                let error = result.expect_err("first byte exceeding the budget");
                assert_eq!(error.kind(), ErrorKind::InvalidData);
                assert_eq!(
                    error.to_string(),
                    "WebView2 presented-frame data exceeds the bounded image budget"
                );
            } else {
                let decoded = result.expect("inclusive capture byte budget");
                assert_eq!(decoded.len(), length);
                let mismatch = decoded
                    .iter()
                    .copied()
                    .zip((0..=u8::MAX).cycle())
                    .position(|(actual, expected)| actual != expected);
                assert_eq!(mismatch, None, "every byte equals index modulo 256");
                assert_eq!(decoded.first(), Some(&0));
                assert_eq!(decoded.last(), Some(&(((length - 1) % 256) as u8)));
            }
        });
    }
}
