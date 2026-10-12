use super::{MAX_HTTP_BYTES, content_length, count_chunk, validate_limits};
use std::{io::ErrorKind, time::Duration};

fn tracing() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(tracing_subscriber::fmt().with_test_writer().finish())
}

#[test]
fn decimal_lengths_preserve_values_and_reject_invalid_headers() {
    let _trace = tracing();
    let _span = tracing::info_span!("http_length_contract").entered();
    for (text, expected) in [("0", 0), ("16", 16), ("016", 16), ("16384", 16384)] {
        assert_eq!(
            content_length(text, 16384).expect("decimal length"),
            expected
        );
    }
    for text in [
        "",
        "-1",
        "+1",
        "1.5",
        " 1",
        "1 ",
        "1,1",
        "16385",
        "18446744073709551616",
    ] {
        assert_eq!(
            content_length(text, 16384).expect_err(text).kind(),
            ErrorKind::InvalidData
        );
    }
}

#[test]
fn streamed_byte_count_uses_remaining_capacity_without_overflow() {
    let _trace = tracing();
    let _span = tracing::info_span!("http_stream_bound").entered();
    for bound in [1, 16, 16384, MAX_HTTP_BYTES, usize::MAX] {
        assert_eq!(count_chunk(0, bound, bound).expect("exact bound"), bound);
        assert_eq!(count_chunk(bound, 0, bound).expect("EOF"), bound);
        assert_eq!(
            count_chunk(bound, 1, bound).expect_err("excess").kind(),
            ErrorKind::InvalidData
        );
    }
    assert_eq!(count_chunk(3, 5, 8).expect("two chunks"), 8);
    assert_eq!(
        count_chunk(9, 0, 8).expect_err("invalid count").kind(),
        ErrorKind::InvalidData
    );
}

#[test]
fn budgets_and_deadlines_reject_before_starting_a_browser_request() {
    let _trace = tracing();
    let _span = tracing::info_span!("http_request_limits").entered();
    for bytes in [1, 16384, MAX_HTTP_BYTES] {
        assert_eq!(
            validate_limits(bytes, bytes, Duration::from_secs(5)).expect("limits"),
            bytes
        );
    }
    for (body, bound, deadline) in [
        (0, 0, Duration::from_secs(5)),
        (MAX_HTTP_BYTES + 1, 1, Duration::from_secs(5)),
        (0, MAX_HTTP_BYTES + 1, Duration::from_secs(5)),
        (0, 1, Duration::ZERO),
        (0, 1, Duration::from_secs(61)),
    ] {
        assert_eq!(
            validate_limits(body, bound, deadline)
                .expect_err("limits")
                .kind(),
            ErrorKind::InvalidInput
        );
    }
}
