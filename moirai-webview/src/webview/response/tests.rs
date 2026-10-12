use super::*;
use objc2_foundation::NSHTTPURLResponse;

fn capture() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

#[test]
fn successful_resources_expose_fetch_status_and_exact_metadata() {
    let _trace = capture();
    let _test = tracing::info_span!("packaged_fetch_response").entered();
    objc2::rc::autoreleasepool(|_| {
        for (path, mime, length) in [
            ("index.html", "text/html", b"<!doctype html>".len()),
            (
                "main.js",
                "text/javascript",
                b"export default function() {}".len(),
            ),
            ("module.wasm", "application/wasm", b"\0asm\x01\0\0\0".len()),
            ("empty.bin", "application/octet-stream", 0),
            (
                "boundary.wasm",
                "application/wasm",
                usize::try_from(crate::MAX_RESOURCE_BYTES).expect("resource bound"),
            ),
        ] {
            let uri = format!("metis://app/{path}");
            let url = tracing::info_span!("arrange", %uri, mime, length).in_scope(|| {
                NSURL::URLWithString(&NSString::from_str(&uri)).expect("resource URL")
            });
            let response = tracing::info_span!("act")
                .in_scope(|| new(&url, mime, length).expect("successful response"));
            tracing::info_span!("assert").in_scope(|| {
                let http = response
                    .downcast_ref::<NSHTTPURLResponse>()
                    .expect("Fetch needs a response carrying HTTP status and headers");
                assert_eq!(http.statusCode(), 200);
                assert_eq!(
                    http.valueForHTTPHeaderField(&NSString::from_str("Content-Type"))
                        .expect("MIME header")
                        .to_string(),
                    mime
                );
                assert_eq!(
                    http.valueForHTTPHeaderField(&NSString::from_str("Content-Length"))
                        .expect("length header")
                        .to_string(),
                    length.to_string()
                );
                assert_eq!(
                    response.MIMEType().expect("MIME metadata").to_string(),
                    mime
                );
                assert_eq!(
                    response.expectedContentLength(),
                    i64::try_from(length).expect("native response length")
                );
                assert_eq!(
                    response
                        .URL()
                        .expect("response URL")
                        .absoluteString()
                        .expect("URL string")
                        .to_string(),
                    uri
                );
                assert_eq!(http.allHeaderFields().len(), 2);
            });
        }
    });
}

#[test]
fn unrepresentable_native_length_fails_before_response_creation() {
    let _trace = capture();
    let _test = tracing::info_span!("response_length_overflow").entered();
    let (url, length) = tracing::info_span!("arrange").in_scope(|| {
        (
            NSURL::URLWithString(&NSString::from_str("metis://app/module.wasm"))
                .expect("resource URL"),
            usize::try_from(isize::MAX).expect("signed length fits") + 1,
        )
    });
    let error = tracing::info_span!("act")
        .in_scope(|| new(&url, "application/wasm", length).expect_err("length exceeds native ABI"));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "out of range integral type conversion attempted"
        );
    });
}
